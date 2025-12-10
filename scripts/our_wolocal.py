# -*- coding: utf-8 -*-

# add more text pertubations


# WANDB_API_KEY=425c813e4ad3283798084d341b069aad7184735b
"""# Set arguments"""

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import wandb
from torch import autocast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from evaluation.shuffled_matching.utils import TextShuffler, pre_caption
from health_multimodal.image.data.io import load_image
from health_multimodal.image.data.transforms import (
    create_chest_xray_transform_for_inference,
)
from health_multimodal.image.model import ImageModel
from health_multimodal.image.model.pretrained import (
    BIOMED_VLP_CXR_BERT_SPECIALIZED,
    CXR_BERT_COMMIT_TAG,
)
from health_multimodal.image.model.types import ImageEncoderType
from health_multimodal.text.model import CXRBertModel, CXRBertTokenizer

# from health_multimodal.image.model.pretrained import get_imagenet_init_encoder
# from health_multimodal.image.model.resnet import resnet50
# from health_multimodal.text.inference_engine import TextInferenceEngine
# from enum import Enum, unique

if torch.cuda.is_available():
    print("You have a GPU")
else:
    print("Warning! No GPU available ")

parser = argparse.ArgumentParser(description="Train MoCo+BIOVIL+newloss_v4")

parser.add_argument("-a", "--arch", default="resnet50")

# lr: 0.06 for batch 512 (or 0.03 for batch 256)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.0015,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--epochs", default=100, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--schedule",
    default=[120, 160],
    nargs="*",
    type=int,
    help="learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on",
)
parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")

parser.add_argument(
    "--batch-size", default=12, type=int, metavar="N", help="mini-batch size"
)
parser.add_argument("--wd", default=5e-4, type=float, metavar="W", help="weight decay")

# moco specific configs:
parser.add_argument("--loss-t", default=1, type=float, help="loss temperature")
parser.add_argument("--shuffle-temp", default=0.07, type=float, help="loss temperature")

# utils
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--results-dir",
    default="",
    type=str,
    metavar="PATH",
    help="path to cache (default: none)",
)
parser.add_argument(
    "--wandb", default=False, action="store_true", help="whether to use wandb"
)

args = parser.parse_args()  # running in command line

# args = parser.parse_args('')  # running in ipynb

# set command line arguments here when running in ipynb
args.epochs = 150
args.cos = True
args.schedule = []  # cos in use
# args.symmetric = False
if args.results_dir == "":
    args.results_dir = (
            "./new_caches_v6"
            + "/shuffle-temp"
            + str(args.shuffle_temp)
            + "/cache-"
            + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-moco")
    )

print(args)

# 🐝 1️⃣ Start a new run to track this script
if args.wandb:
    wandb.login(key="425c813e4ad3283798084d341b069aad7184735b")
    wandb.init(
        # Set the project where this run will be logged
        project="0909_v5",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name="experiment_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-moco"),
        # Track hyperparameters and run metadata
        config=args,
    )

"""# Dataloader"""

images_captions_df = pd.read_csv(
    "/ocean/projects/asc170022p/lisun/xinliu/images/csv/indiana_captions.csv"
)

# import swifter
new_df = images_captions_df.copy()
new_df = new_df.drop(index=range(6000, 6469))


# val_df = images_captions_df.copy()
# val_df = val_df.drop(index = range(0,6000))
# print(val_df.head())


class ShuffledOpenIDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, max_words=50):
        self.df = df
        self.test_cases = []
        self.transform = transform
        self.root_dir = root_dir
        self.tokenizer = CXRBertTokenizer.from_pretrained(
            BIOMED_VLP_CXR_BERT_SPECIALIZED, revision=CXR_BERT_COMMIT_TAG
        )
        shuffler = TextShuffler()
        perturb_functions = [
            shuffler.shuffle_nouns_and_adj,
            shuffler.shuffle_allbut_nouns_and_adj,
            shuffler.shuffle_within_trigrams,
            shuffler.shuffle_trigrams,
            shuffler.shuffle_all_words,
            shuffler.reverse_sentence,
            shuffler.shuffle_nouns_verbs_adj,
            shuffler.replace_adjectives_with_antonyms,
            shuffler.swap_adjacent_words,
        ]
        for index, ann in tqdm(self.df.iterrows()):
            test_case = {"image": ann["image"]}
            caption = ann["caption"]
            test_case["caption_options"] = [pre_caption(caption, max_words)]

            for perturb_fn in perturb_functions:
                test_case["caption_options"].append(
                    pre_caption(perturb_fn(caption), max_words)
                )
            # print(len(test_case["caption_options"]))
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


def collate_fn(batch):
    input_ids_batch = [item[0] for item in batch]
    # print(len(input_ids_batch))64
    # print(len(input_ids_batch[0]))5
    # print(len(input_ids_batch[0][0]))732
    # print(len(input_ids_batch[0][0][0]))

    attention_mask_batch = [item[1] for item in batch]
    image_input_batch = [item[2] for item in batch]
    # print()

    # Pad sequences to the length of the longest sequence in the batch
    # input_ids_padded = [pad_sequence(i, batch_first=True) for i in input_ids_batch ]
    # attention_mask_padded = [pad_sequence(i, batch_first=True) for i in attention_mask_batch]
    input_ids_padded = [
        pad_sequence([seq for seq in option_set], batch_first=True)
        for option_set in zip(*input_ids_batch)
    ]
    attention_mask_padded = [
        pad_sequence([seq for seq in option_set], batch_first=True)
        for option_set in zip(*attention_mask_batch)
    ]

    # input_ids_padded = pad_sequence(input_ids_padded, batch_first=True)
    # attention_mask_padded = pad_sequence(attention_mask_padded, batch_first=True)
    # input_ids_padded = torch.stack(input_ids_padded)
    # attention_mask_padded = torch.stack(attention_mask_padded)

    # Convert lists of tensors to tensors
    # input_ids_options_tensor = torch.stack(input_ids_padded)
    # attention_mask_options_tensor = torch.stack(attention_mask_padded)

    image_input_tensor = torch.stack(image_input_batch)
    # print(input_ids_tensor.shape, attention_mask_tensor.shape, image_input_tensor.shape)

    return input_ids_padded, attention_mask_padded, image_input_tensor


transforms = create_chest_xray_transform_for_inference(
    # resize=512,
    # center_crop_size=448
    resize=256,
    center_crop_size=224,
)

train_dataset = ShuffledOpenIDataset(
    new_df,
    root_dir="/ocean/projects/asc170022p/lisun/xinliu/images/images_normalized",
    transform=transforms,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=False,
    drop_last=True,
    collate_fn=collate_fn,
)

# batch = next(iter(train_loader))

# print('input_ids_options, attention_mask_options, img')
# print()
# print(batch[0], len(batch[1]), batch[2].shape)
# print(len(batch[0][0]), len(batch[1][0]), batch[2].shape)

"""# Identical Shuffled Contrastive model"""


class IShuffledContrastiveModel(nn.Module):
    def __init__(self, T=0.5, shuffle_temp=0.07):
        super(IShuffledContrastiveModel, self).__init__()

        self.T = T
        print("T: ", T)
        print("shuffle_temp: ", shuffle_temp)

        # Text Encoder CXRBert
        self.text_encoder = CXRBertModel.from_pretrained(
            BIOMED_VLP_CXR_BERT_SPECIALIZED, revision=CXR_BERT_COMMIT_TAG
        )
        # Image Encoder ResNet50
        self.image_encoder = ImageModel(
            img_encoder_type=ImageEncoderType.RESNET50.value, joint_feature_size=128
        )

        self.criterion = nn.CrossEntropyLoss()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_shuffle = nn.Parameter(
            torch.ones([]) * np.log(1 / shuffle_temp)
        )

    def contrastive_loss(self, input_ids, attention_mask, image_input):
        # print('image_input:', image_input.shape, image_input.min(), image_input.max())
        batch_size = input_ids.shape[0]

        # Text encoding
        text_embeds = self.text_encoder.get_projected_text_embeddings(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # Image encoding
        image_embeds = self.image_encoder(image_input).projected_global_embedding

        text_embeds = nn.functional.normalize(text_embeds, dim=1)
        image_embeds = nn.functional.normalize(image_embeds, dim=1)

        logits_text_per_image = self.logit_scale.exp() * image_embeds @ text_embeds.t()
        logits_image_per_text = logits_text_per_image.t()

        target = (
            torch.arange(batch_size).long().to(text_embeds.device, non_blocking=True)
        )
        # print(text_embeds.device, image_embeds.device, self.device, logits_text_per_image.device)
        loss = (
                       self.criterion(logits_text_per_image, target)
                       + self.criterion(logits_image_per_text, target)
               ) / 2

        return loss

    def shuffle_loss(self, input_ids_shuffled, attention_mask_shuffled, image_input):
        batch_size = len(input_ids_shuffled[0])
        num_text = len(input_ids_shuffled)
        # print("batch_size: ",batch_size)
        # print("num_text: ",num_text)

        image_embeds = self.image_encoder(image_input).projected_global_embedding
        image_embeds = nn.functional.normalize(image_embeds, dim=1)

        # Compute the logits
        logits = []

        for txt in range(num_text):
            input_ids = input_ids_shuffled[txt]
            attention_mask = attention_mask_shuffled[txt]

            input_ids_tensor = pad_sequence(input_ids, batch_first=True)
            attention_mask_tensor = pad_sequence(attention_mask, batch_first=True)

            text_embeds = self.text_encoder.get_projected_text_embeddings(
                input_ids=input_ids_tensor, attention_mask=attention_mask_tensor
            )
            text_embeds = nn.functional.normalize(text_embeds, dim=1)

            # [Batch_size, Dim] * [Batch_size, Dim] -> [Batch_size, Dim]
            logits_i = self.logit_scale_shuffle.exp() * image_embeds * text_embeds
            logits.append(torch.sum(logits_i, dim=1)[:, None])

        logits = torch.concatenate(logits, 1)
        # Compute the cross-entropy loss
        targets = torch.zeros(batch_size, dtype=torch.long).to(
            image_embeds.device
        )  # the original text is always at index 0
        loss = self.criterion(logits, targets)

        return loss

    def forward(self, input_ids, attention_mask, image_input):
        # transpose
        # input_ids = torch.transpose(input_ids, 0, 1)
        # attention_mask = torch.transpose(attention_mask, 0, 1)
        # input_ids = np.array(input_ids).T.tolist()
        # attention_mask = np.array(attention_mask).T.tolist()
        device = torch.device("cuda")
        input_ids = [i.to(device) for i in input_ids]
        attention_mask = [a.to(device) for a in attention_mask]

        shuffle_loss = self.shuffle_loss(input_ids, attention_mask, image_input)
        # print("shuffle_loss: ", shuffle_loss)
        contrastive_loss = self.contrastive_loss(
            pad_sequence(input_ids[0], batch_first=True),
            pad_sequence(attention_mask[0], batch_first=True),
            image_input,
        )
        # print("contrastive_loss: ", contrastive_loss)

        loss = contrastive_loss + self.T * shuffle_loss
        # print("total_loss: ", loss)

        return loss


# Create the contrastive model

# print(model.text_encoder)
# print(model.image_encoder)
# model = torch.nn.DataParallel(model, device_ids=device_ids)

# Sample inputs
# text_input = ["There is no pneumothorax or pleural effusion",
# "The extent of the pleural effusion is constant."] # Example text input (replace with your own)
# image_input = torch.randn(2, 3, 128, 128).cuda(non_blocking=True)  # Example image batch with 2 images


# output = model(text_input, image_input)
# print("Output shape:", output.shape)
# print(output)

"""# Define train/test"""


# train for one epoch
def train(net, data_loader, train_optimizer, scaler, epoch, args):
    device = torch.device("cuda:0")  # Change to the desired GPU index
    net = net.to(device)
    net.train()
    adjust_learning_rate(train_optimizer, epoch, args)

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for input_ids, attention_mask, image_input in train_bar:
        # input_ids, attention_mask, image_input = input_ids.to(device), attention_mask.to(device), image_input.to(device)
        image_input = image_input.to(device)
        train_optimizer.zero_grad()

        # print(input_ids.shape, attention_mask.shape, image_input.shape)
        with autocast(device_type="cuda", dtype=torch.float16):
            loss = net(input_ids, attention_mask, image_input)

        scaler.scale(loss).backward()
        scaler.step(train_optimizer)
        scaler.update()
        # torch.cuda.empty_cache()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        train_bar.set_description(
            "Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}".format(
                epoch,
                args.epochs,
                train_optimizer.param_groups[0]["lr"],
                total_loss / total_num,
            )
        )

    return total_loss / total_num


# lr scheduler for training
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


"""# Start training"""


# define optimizer
def main():
    device = torch.device("cuda:0")
    T = args.loss_t
    model = IShuffledContrastiveModel(T, shuffle_temp=args.shuffle_temp).to(device)
    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9
    )
    scaler = torch.cuda.amp.GradScaler()

    # load model if resume
    epoch_start = 1
    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch_start = checkpoint["epoch"] + 1
        print("Loaded from: {}".format(args.resume))

    # logging
    # results = {'train_loss': [], 'test_acc@1': []}
    results = {"train_loss": []}
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    # dump args
    with open(args.results_dir + "/args.json", "w") as fid:
        json.dump(args.__dict__, fid, indent=2)

    # training loop
    for epoch in range(epoch_start, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, scaler, epoch, args)
        results["train_loss"].append(train_loss)
        # test_acc_1 = test(model, memory_loader, test_loader, epoch, args)
        # results['test_acc@1'].append(test_acc_1)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
        data_frame.to_csv(args.results_dir + "/log.csv", index_label="epoch")
        # save model
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            args.results_dir + "/model_last.pth",
        )
        if args.wandb:
            wandb.log({"loss": train_loss})
        # gc.collect()
        # torch.cuda.empty_cache()
        # model.save_pretrained("./save_pretrained/"+str(epoch))
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
