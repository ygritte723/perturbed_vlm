import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from health_multimodal.image.data.io import load_image
from health_multimodal.image.data.transforms import (
    create_chest_xray_transform_for_inference,
)
from utils_v4 import TextShuffler, pre_caption, GShuffle

batch_size = 64
lr = 0.0015

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GShuffle()

# optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
# optimizer.load_state_dict(checkpoint['optimizer'])
pt = (
    "/jet/home/lisun/work/xinliu/hi-ml/hi-ml-multimodal/src/"
    + "new_caches_v7/T0.1_L0.1_shuffle-temp0.01/cache-2023-11-27-06-06-56-moco/model_last.pth"
)
checkpoint = torch.load(pt, map_location=device)
print("checkpoint_path:", pt)

msg = model.load_state_dict(checkpoint["state_dict"], strict=False)
print(msg)
"""# Dataloader"""

images_captions_df = pd.read_csv(
    "/ocean/projects/asc170022p/lisun/xinliu/images/csv/indiana_captions.csv"
)

val_df = images_captions_df.copy()
val_df = val_df.drop(index=range(0, 6000))
# print(val_df.caption.iloc[0])

transforms = create_chest_xray_transform_for_inference(resize=256, center_crop_size=224)


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
        # print(idx)
        img_path = Path(img_path)
        img = load_image(img_path)

        if self.transform is not None:
            img = self.transform(img)
        # Text encoding (Bag of Words)
        # caption = [self.df.caption.iloc[idx]]
        # print(caption,len(caption))
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
            # input_ids = tokenizer_output.input_ids.view(-1).squeeze()
            input_ids = tokenizer_output.input_ids.view(-1).squeeze()
            input_ids_options.append(input_ids)
            # attention_mask = tokenizer_output.attention_mask.view(-1).squeeze()
            attention_mask = tokenizer_output.attention_mask.view(-1).squeeze()
            attention_mask_options.append(attention_mask)

        # pixel_values = self.feature_extractor(img, return_tensors="pt").pixel_values
        # print(input_ids.shape, attention_mask.shape, img.shape)
        return input_ids_options, attention_mask_options, img


test_dataset = ShuffledOpenIDataset(
    val_df,
    root_dir="/ocean/projects/asc170022p/lisun/xinliu/images/images_normalized",
    transform=transforms,
)


def collate_fn(batch):
    input_ids_batch = [item[0] for item in batch]
    # print(len(input_ids_batch))64
    # print(len(input_ids_batch[0]))5
    # print(len(input_ids_batch[0][0]))732
    # print(len(input_ids_batch[0][0][0]))

    attention_mask_batch = [item[1] for item in batch]
    image_input_batch = [item[2] for item in batch]
    print()

    # Pad sequences to the length of the longest sequence in the batch
    input_ids_padded = [pad_sequence(i, batch_first=True) for i in input_ids_batch]
    attention_mask_padded = [
        pad_sequence(i, batch_first=True) for i in attention_mask_batch
    ]

    # Convert lists of tensors to tensors
    # input_ids_options_tensor = torch.stack(input_ids_padded)
    # attention_mask_options_tensor = torch.stack(attention_mask_padded)

    image_input_tensor = torch.stack(image_input_batch)
    # print(input_ids_tensor.shape, attention_mask_tensor.shape, image_input_tensor.shape)

    return input_ids_padded, attention_mask_padded, image_input_tensor


test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=False,
    collate_fn=collate_fn,
)


model = model.to(device)
criterion = nn.CrossEntropyLoss()
logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

total = 0
right = 0

for batch_ndx, batch in tqdm(enumerate(test_loader)):
    b_input_ids_options, b_attention_mask_options, b_img = batch

    # img = b_img
    # print(len(input_ids_options))64
    # input_ids_options_batches = []
    for j in range(len(b_input_ids_options)):
        input_ids_options = b_input_ids_options[j]
        # print(len(input_id))5
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
            text_embeds = model.text_encoder.get_projected_text_embeddings(
                input_ids=input_ids, attention_mask=attention_mask
            )
            # Image encoding
            image_embeds = model.image_encoder(img).projected_global_embedding

            text_embeds = nn.functional.normalize(text_embeds, dim=1)
            image_embeds = nn.functional.normalize(image_embeds, dim=1)
            # logits_text_per_image = logit_scale.exp() * image_embeds @ text_embeds.t()
            # logits_image_per_text = logits_text_per_image.t()

            # target = torch.arange(1).long().to(device, non_blocking = True)
            # print(logits_text_per_image)
            # print(target)
            # print(text_embeds.shape, image_embeds.shape)
            # print(text_embeds)
            # print()
            # print(image_embeds)
            cos = nn.CosineSimilarity(dim=1, eps=1e-8)
            loss = cos(text_embeds, text_embeds)
            # print(text_embeds.shape, image_embeds.shape,  logits_text_per_image.shape)
            # loss = (criterion(logits_text_per_image, target) + criterion(logits_image_per_text, target)) / 2
            losses.append(loss.item())

        # loss.append(model(input_id, attention_mask, img))
        if losses[0] == max(losses):
            right = right + 1
        # print(losses)
        # print(max(losses))
print(f"Total: {total}")
print(f"Correct: {right}")
