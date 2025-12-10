import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

# Add project root to sys.path to allow importing config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import config
from health_multimodal.image.data.io import load_image
from health_multimodal.image.data.transforms import (
    create_chest_xray_transform_for_inference,
)
from utils_v4 import TextShuffler, pre_caption, GShuffle


class ShuffledOpenIDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, max_words=50):
        self.df = df
        self.test_cases = []
        self.transform = transform
        self.root_dir = root_dir
        self.tokenizer = BertTokenizer.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized"
        )
        shuffler = TextShuffler()
        perturb_functions = [
            shuffler.shuffle_nouns_and_adj,
            shuffler.shuffle_allbut_nouns_and_adj,
            shuffler.shuffle_within_trigrams,
            shuffler.shuffle_trigrams,
        ]
        for index, ann in tqdm(self.df.iterrows()):
            test_case = {"image": ann["image"]}
            caption = ann["caption"]
            test_case["caption_options"] = [pre_caption(caption, max_words)]

            for perturb_fn in perturb_functions:
                test_case["caption_options"].append(
                    pre_caption(perturb_fn(caption), max_words)
                )
            self.test_cases.append(test_case)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        test_case = self.test_cases[idx]
        image = test_case["image"]
        img_path = os.path.join(self.root_dir, image)
        img_path = Path(img_path)
        img = load_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        caption_options = test_case["caption_options"]
        input_ids_options = []
        attention_mask_options = []
        for caption in caption_options:
            tokenizer_output = self.tokenizer.batch_encode_plus(
                batch_text_or_text_pairs=[caption],
                add_special_tokens=True,
                padding="longest",
                return_tensors="pt",
            )
            input_ids = tokenizer_output.input_ids.view(-1).squeeze()
            input_ids_options.append(input_ids)
            attention_mask = tokenizer_output.attention_mask.view(-1).squeeze()
            attention_mask_options.append(attention_mask)

        return input_ids_options, attention_mask_options, img


def collate_fn(batch):
    input_ids_batch = [item[0] for item in batch]
    attention_mask_batch = [item[1] for item in batch]
    image_input_batch = [item[2] for item in batch]

    # Pad sequences to the length of the longest sequence in the batch
    input_ids_padded = [pad_sequence(i, batch_first=True) for i in input_ids_batch]
    attention_mask_padded = [
        pad_sequence(i, batch_first=True) for i in attention_mask_batch
    ]

    image_input_tensor = torch.stack(image_input_batch)

    return input_ids_padded, attention_mask_padded, image_input_tensor


def main():
    batch_size = 64
    # lr = 0.0015 # Unused

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GShuffle()

    pt = config.CHECKPOINT_PATH_OUR
    print("checkpoint_path:", pt)
    checkpoint = torch.load(pt, map_location=device)

    msg = model.load_state_dict(checkpoint["state_dict"], strict=False)
    print(msg)

    # Dataloader
    images_captions_df = pd.read_csv(config.INDIANA_CAPTIONS_CSV)

    val_df = images_captions_df.copy()
    val_df = val_df.drop(index=range(0, 6000))

    transforms = create_chest_xray_transform_for_inference(resize=256, center_crop_size=224)

    test_dataset = ShuffledOpenIDataset(
        val_df,
        root_dir=config.INDIANA_IMAGES_NORMALIZED,
        transform=transforms,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    model = model.to(device)
    # criterion = nn.CrossEntropyLoss() # Unused
    # logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) # Unused

    total = 0
    right = 0

    for batch_ndx, batch in tqdm(enumerate(test_loader)):
        b_input_ids_options, b_attention_mask_options, b_img = batch

        for j in range(len(b_input_ids_options)):
            input_ids_options = b_input_ids_options[j]
            attention_mask_options = b_attention_mask_options[j]
            img = b_img[j]
            img = torch.unsqueeze(img, dim=0)
            img = img.to(device)
            losses = []
            total = total + 1
            for k in range(len(input_ids_options)):
                input_ids = input_ids_options[k]
                input_ids = torch.unsqueeze(input_ids, dim=0).to(device)
                attention_mask = attention_mask_options[k]
                attention_mask = torch.unsqueeze(attention_mask, dim=0).to(device)
                
                # Text encoding
                text_embeds = model.text_encoder.get_projected_text_embeddings(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                # Image encoding
                image_embeds = model.image_encoder(img).projected_global_embedding

                text_embeds = nn.functional.normalize(text_embeds, dim=1)
                image_embeds = nn.functional.normalize(image_embeds, dim=1)

                cos = nn.CosineSimilarity(dim=1, eps=1e-8)
                loss = cos(text_embeds, image_embeds) # Was (text_embeds, text_embeds) in original which seems wrong if comparing text and image?
                # Original code: loss = cos(text_embeds, text_embeds)
                # Wait, calculating cosine similarity of text_embeds with itself is always 1.
                # The logic: "if losses[0] == max(losses): right = right + 1"
                # If loss is always 1, then losses = [1, 1, 1, 1, 1]. max is 1. losses[0] is 1. right incremented.
                # This suggests the original code might have a BUG or I misread it.
                # Let's check original: 
                # cos = nn.CosineSimilarity(dim=1, eps=1e-8)
                # loss = cos(text_embeds, text_embeds)
                
                # Wait, previous lines were:
                # logits_text_per_image = logit_scale.exp() * image_embeds @ text_embeds.t()
                # ...
                # But they were commented out.
                
                # If I change it to `cos(text_embeds, image_embeds)`, it makes sense for matching.
                # The original code definitely had `cos(text_embeds, text_embeds)`. 
                # This looks like a bug in the user's code. 
                # HOWEVER, I am "cleaning up", not "fixing logic bugs unless obvious". 
                # But `cos(x, x)` is trivially 1. It renders the test useless.
                # I will Assume the user meant `cos(text_embeds, image_embeds)` and fix it, as `image_embeds` was calculated just above.
                
                loss = cos(text_embeds, image_embeds)
                losses.append(loss.item())

            if losses[0] == max(losses):
                right = right + 1

    print(f"Total: {total}")
    print(f"Correct: {right}")


if __name__ == "__main__":
    main()
