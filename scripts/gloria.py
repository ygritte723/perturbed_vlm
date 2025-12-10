import argparse
import json
# import wandb
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from tqdm import tqdm

from health_multimodal.image.data.io import load_image
from health_multimodal.image.data.transforms import (
    create_chest_xray_transform_for_inference,
)
from health_multimodal.image.model import ImageModel
from health_multimodal.text.model import CXRBertModel, CXRBertTokenizer

parser = argparse.ArgumentParser(description="Train Gloria+BIOVIL")

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
    "--batch-size", default=64, type=int, metavar="N", help="mini-batch size"
)
parser.add_argument("--wd", default=5e-4, type=float, metavar="W", help="weight decay")

parser.add_argument("--loss-l", default=0.1, type=float, help="local loss temperature")
parser.add_argument(
    "--local-temp1", default=4.0, type=float, help="Local loss1 inside temperature"
)
parser.add_argument(
    "--local-temp2", default=5.0, type=float, help="Local loss2 inside temperature"
)
parser.add_argument(
    "--local-temp3", default=10.0, type=float, help="Local loss3 inside temperature"
)

# parser.add_argument('--bn-splits', default=8, type=int, help='simulate multi-gpu behavior of BatchNorm in one gpu; 1 is SyncBatchNorm in multi-gpu')

# parser.add_argument('--symmetric', action='store_true', help='use a symmetric loss function that backprops to both crops')

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
parser.add_argument("--wandb", default=False, type=bool, help="whether to use wandb")

args = parser.parse_args()  # running in command line

# args = parser.parse_args('')  # running in ipynb

# set command line arguments here when running in ipynb
args.epochs = 150
args.cos = True
args.schedule = []  # cos in use
# args.symmetric = False
if args.results_dir == "":
    args.results_dir = "./caches/cache-" + datetime.now().strftime(
        "%Y-%m-%d-%H-%M-%S-moco"
    )

print(args)

# 🐝 1️⃣ Start a new run to track this script
# if args.wandb:
#   wandb.login()
#   wandb.init(
# Set the project where this run will be logged
#       project="1123",
# We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
#       name='experiment_' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-moco"),
# Track hyperparameters and run metadata
#      config=args)


"""# Dataloader"""

images_captions_df = pd.read_csv(
    "/ocean/projects/asc170022p/lisun/xinliu/images/csv/indiana_captions.csv"
)

new_df = images_captions_df.copy()
new_df = new_df.drop(index=range(6000, 6469))

val_df = images_captions_df.copy()
val_df = val_df.drop(index=range(0, 6000))


# print(val_df.head())


def collate_fn(batch):
    input_ids_batch = [item[0] for item in batch]
    attention_mask_batch = [item[1] for item in batch]
    token_type_ids_batch = [item[2] for item in batch]
    cap_lens_batch = [item[3] for item in batch]
    image_input_batch = [item[-1] for item in batch]

    # Pad sequences to the length of the longest sequence in the batch
    # input_ids_padded = pad_sequence(input_ids_batch, batch_first=True)
    # attention_mask_padded = pad_sequence(attention_mask_batch, batch_first=True)
    # token_type_ids_padded = pad_sequence(token_type_ids_batch, batch_first=True)
    # cap_lens_padded = pad_sequence(cap_lens_batch, batch_first=True)
    input_ids_padded = torch.stack(input_ids_batch).squeeze()
    attention_mask_padded = torch.stack(attention_mask_batch).squeeze()
    token_type_ids_padded = torch.stack(token_type_ids_batch).squeeze()
    cap_lens_padded = cap_lens_batch

    # Convert lists to tensors
    input_ids_tensor = input_ids_padded
    attention_mask_tensor = attention_mask_padded
    token_type_ids_tensor = token_type_ids_padded
    cap_lens_tensor = cap_lens_padded

    image_input_tensor = torch.stack(image_input_batch)
    # print(input_ids_tensor.shape, attention_mask_tensor.shape, image_input_tensor.shape)

    sorted_cap_lens, sorted_cap_indices = torch.sort(
        torch.tensor(cap_lens_tensor), 0, True
    )
    # print(image_input_tensor[0])
    return (
        input_ids_tensor[sorted_cap_indices],
        attention_mask_tensor[sorted_cap_indices],
        token_type_ids_tensor[sorted_cap_indices],
        sorted_cap_lens,
        image_input_tensor[sorted_cap_indices],
    )


class OpenIDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.transform = transform
        self.root_dir = root_dir
        self.tokenizer = CXRBertTokenizer.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized", revision="v1.1"
        )
        # self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Text encoding (Bag of Words)
        caption = [self.df.caption.iloc[idx]]
        # print(caption,len(caption))

        tokenizer_output = self.tokenizer(
            caption,
            # add_special_tokens=True,
            # padding='longest',
            return_tensors="pt",
            # return_length=True,
            truncation=True,
            padding="max_length",
            max_length=50,
        )
        input_ids = tokenizer_output.input_ids.view(-1).squeeze()
        attention_mask = tokenizer_output.attention_mask.view(-1).squeeze()
        token_type_ids = tokenizer_output.token_type_ids.view(-1).squeeze()
        cap_lens = len([t for t in tokenizer_output.input_ids[0] if t != 0])

        image = self.df.image.iloc[idx]
        img_path = os.path.join(self.root_dir, image)
        # print(idx)
        img_path = Path(img_path)
        img = load_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        # pixel_values = self.feature_extractor(img, return_tensors="pt").pixel_values
        # print(input_ids.shape, attention_mask.shape, img.shape)
        return input_ids, attention_mask, token_type_ids, cap_lens, img


# train_dataset = OpenIDataset(
# new_df,
# root_dir= "/ocean/projects/asc170022p/lisun/xinliu/images/images_normalized"
# tokenizer=CXRBertTokenizer.from_pretrained(BIOMED_VLP_CXR_BERT_SPECIALIZED, revision=CXR_BERT_COMMIT_TAG)
# )
# print(train_dataset[0][0].shape, train_dataset[0][1].shape, train_dataset[0][2])
# gc.collect()
# torch.cuda.empty_cache()
# del train_dataset


transforms = create_chest_xray_transform_for_inference(
    # resize=512,
    # center_crop_size=448
    resize=256,
    center_crop_size=224,
)

train_dataset = OpenIDataset(
    new_df,
    root_dir="/ocean/projects/asc170022p/lisun/xinliu/images/images_normalized",
    # tokenizer=CXRBertTokenizer.from_pretrained(BIOMED_VLP_CXR_BERT_SPECIALIZED, revision=CXR_BERT_COMMIT_TAG),
    transform=transforms,
)
memory_dataset = OpenIDataset(
    new_df,
    root_dir="/ocean/projects/asc170022p/lisun/xinliu/images/images_normalized",
    # tokenizer=CXRBertTokenizer.from_pretrained(BIOMED_VLP_CXR_BERT_SPECIALIZED, revision=CXR_BERT_COMMIT_TAG),
    transform=transforms,
)
test_dataset = OpenIDataset(
    val_df,
    root_dir="/ocean/projects/asc170022p/lisun/xinliu/images/images_normalized",
    # tokenizer=CXRBertTokenizer.from_pretrained(BIOMED_VLP_CXR_BERT_SPECIALIZED, revision=CXR_BERT_COMMIT_TAG),
    transform=transforms,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=False,
    drop_last=True,
    collate_fn=collate_fn,
)
memory_loader = DataLoader(
    memory_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=False,
    collate_fn=collate_fn,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=False,
    collate_fn=collate_fn,
)
# print(len(train_dataset[0]))
# print(train_dataset[0][0].shape)
# print(train_dataset[0][1].shape)
"""# Contrastive model"""


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def attention_fn(query, context, temp1):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query)
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size * sourceL, queryL)
    attn = nn.Softmax(dim=-1)(attn)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)

    attn = attn * temp1
    attn = nn.Softmax(dim=-1)(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)


class Identity(nn.Module):
    """Identity layer to replace last fully connected layer"""

    def forward(self, x):
        return x


class Gloria(nn.Module):
    def __init__(
            self, L=0.5, last_n_layer=4, local_temp1=4.0, local_temp2=5.0, local_temp3=10.0
    ):
        super(Gloria, self).__init__()

        self.L = L
        self.local_temp1 = local_temp1
        self.local_temp2 = local_temp2
        self.local_temp3 = local_temp3
        self.last_n_layer = last_n_layer  # Text Encoder CXRBert
        # self.text_encoder = CXRBertModel.from_pretrained(BIOMED_VLP_CXR_BERT_SPECIALIZED, revision=CXR_BERT_COMMIT_TAG)
        self.text_encoder = CXRBertModel.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized", revision="v1.1"
        )
        # Image Encoder ResNet50
        self.image_encoder = ImageModel(
            img_encoder_type="resnet50", joint_feature_size=128
        )
        self.local_up = nn.Upsample(
            size=(299, 299), mode="bilinear", align_corners=True
        )
        self.local_image_encoder = resnet50()
        self.local_image_encoder.fc = Identity()
        self.local_embedder = nn.Conv2d(
            1024,
            768,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        # self.local_text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", output_hidden_states=True)
        # self.local_text_encoder = self.text_encoder
        # self.idxtoword = {v: k for k, v in AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").get_vocab().items()}
        self.idxtoword = {
            v: k
            for k, v in CXRBertTokenizer.from_pretrained(
                "microsoft/BiomedVLP-CXR-BERT-specialized", revision="v1.1"
            )
            .get_vocab()
            .items()
        }
        self.criterion = nn.CrossEntropyLoss()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def aggregate_tokens(self, embeddings, caption_ids):
        batch_size, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)
        agg_embs_batch = []
        sentences = []

        # loop over batch
        for embs, caption_id in zip(embeddings, caption_ids):
            agg_embs = []
            token_bank = []
            words = []
            word_bank = []

            # loop over sentence
            for word_emb, word_id in zip(embs, caption_id):
                word = self.idxtoword[word_id.item()]

                if word == "[SEP]":
                    new_emb = torch.stack(token_bank)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))

                    agg_embs.append(word_emb)
                    words.append(word)
                    break

                if not word.startswith("##"):
                    if len(word_bank) == 0:
                        token_bank.append(word_emb)
                        word_bank.append(word)
                    else:
                        new_emb = torch.stack(token_bank)
                        new_emb = new_emb.sum(axis=0)
                        agg_embs.append(new_emb)
                        words.append("".join(word_bank))

                        token_bank = [word_emb]
                        word_bank = [word]
                else:
                    if word.startswith("##"):
                        token_bank.append(word_emb)
                        word_bank.append(word[2:])

            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, dim)
            paddings = paddings.to(agg_embs.device)
            words = words + ["[PAD]"] * padding_size

            agg_embs_batch.append(torch.cat([agg_embs, paddings]))
            sentences.append(words)

        agg_embs_batch = torch.stack(agg_embs_batch)
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
        return agg_embs_batch, sentences

    def resnet_forward(self, x):
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode="bilinear", align_corners=True)(x)

        x = self.local_image_encoder.conv1(x)  # (batch_size, 64, 150, 150)
        x = self.local_image_encoder.bn1(x)
        x = self.local_image_encoder.relu(x)
        x = self.local_image_encoder.maxpool(x)

        x = self.local_image_encoder.layer1(x)  # (batch_size, 64, 75, 75)
        x = self.local_image_encoder.layer2(x)  # (batch_size, 128, 38, 38)
        x = self.local_image_encoder.layer3(x)  # (batch_size, 256, 19, 19)
        local_features = x

        return local_features

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
        # print('global loss: ', str(loss))
        return loss

    def local_loss(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            cap_lens,
            image_input,
            temp1=4.0,
            temp2=5.0,
            temp3=10.0,
    ):
        # print('image_input:', image_input.shape, image_input.min(), image_input.max())
        batch_size = image_input.shape[0]

        # local_image_embeds = self.local_image_encoder(image_input, return_intermediate_layers=True)[3]
        local_image_embeds = self.resnet_forward(image_input)
        if torch.any(torch.isnan(local_image_embeds)) or torch.any(
                torch.isinf(local_image_embeds)
        ):
            print("NaN or Inf in image_input after encoder")
        local_image_embeds = self.local_embedder(local_image_embeds)
        if torch.any(torch.isnan(local_image_embeds)) or torch.any(
                torch.isinf(local_image_embeds)
        ):
            print("NaN or Inf in local_image_embeds after embedder")

        # text_outputs = self.local_text_encoder(input_ids=input_ids,
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            # return_dict=True,
            output_hidden_states=True,
        )
        all_embeddings = text_outputs[3]
        # print(len(all_embeddings))
        # (batch_size, sequence_length, hidden_size)
        embeddings = torch.stack(
            all_embeddings[-self.last_n_layer:]
        )  # layers, batch, sent_len, embedding size

        embeddings = embeddings.permute(1, 0, 2, 3)

        embeddings, sents = self.aggregate_tokens(embeddings, token_type_ids)

        word_embeddings = embeddings.sum(axis=1)

        batch_dim, num_words, feat_dim = word_embeddings.shape
        word_embeddings = word_embeddings.view(batch_dim * num_words, feat_dim)
        word_embeddings = word_embeddings.view(batch_dim, num_words, 768)
        word_embeddings = word_embeddings.permute(0, 2, 1)
        word_embeddings = word_embeddings / (
                torch.norm(word_embeddings, 2, dim=1, keepdim=True).expand_as(
                    word_embeddings
                )
                + 1e-6
        )
        words_emb = word_embeddings

        similarities = []
        # cap_lens = cap_lens.data.tolist()
        for i in range(words_emb.shape[0]):
            # Get the i-th text description
            words_num = cap_lens[i]  # 25
            # TODO: remove [SEP]
            # word = words_emb[i, :, 1:words_num+1].unsqueeze(0).contiguous()    # [1, 768, 25]
            word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()  # [1, 768, 25]
            word = word.repeat(batch_size, 1, 1)  # [48, 768, 25]
            # print(word.shape)
            context = local_image_embeds  # [48, 768, 19, 19]
            # print(context.shape)
            # print("context: ", str(context.shape))
            # print("word: ", str(word.shape))
            # print("Attn: ", str(attn))
            # print("word: ", str(word))
            weiContext, attn = attention_fn(
                word, context, temp1
            )  # [48, 768, 25], [48, 25, 19, 19]
            # print("weiContext: ", str(weiContext.shape))
            # print("Attn: ", str(attn.shape))

            word = word.transpose(1, 2).contiguous()  # [48, 25, 768]
            weiContext = weiContext.transpose(1, 2).contiguous()  # [48, 25, 768]

            word = word.view(batch_size * words_num, -1)  # [1200, 768]
            weiContext = weiContext.view(batch_size * words_num, -1)  # [1200, 768]

            row_sim = cosine_similarity(word, weiContext)
            row_sim = row_sim.view(batch_size, words_num)  # [48, 25]

            row_sim.mul_(temp2).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)  # [48, 1]
            row_sim = torch.log(row_sim)
            similarities.append(row_sim)

        similarities = torch.cat(similarities, 1)  #
        similarities = similarities * temp3
        similarities1 = similarities.transpose(0, 1)  # [48, 48]
        # print('Similarity: ', str(similarities))
        # print('Similarity1: ', str(similarities1))
        labels = (
            torch.arange(batch_size).long().to(similarities.device, non_blocking=True)
        )

        loss0 = self.criterion(similarities, labels)  # labels: arange(batch_size)
        loss1 = self.criterion(similarities1, labels)

        loss = loss0 + loss1 / 2
        # print('local loss: ', str(loss))

        return loss

    def forward(self, input_ids, attention_mask, token_type_ids, cap_lens, image_input):
        loss_g = self.contrastive_loss(input_ids, attention_mask, image_input)
        loss_l = self.local_loss(
            input_ids, attention_mask, token_type_ids, cap_lens, image_input
        )
        loss = loss_g + self.L * loss_l

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
    for input_ids, attention_mask, token_type_ids, cap_lens, image_input in train_bar:
        # print(f"Batch size: {input_ids.size(0)}")
        # print(f"Input IDs range: [{input_ids.min()}, {input_ids.max()}]")
        # print(f"Attention mask range: [{attention_mask.min()}, {attention_mask.max()}]")
        # print(f"Token type IDs range: [{token_type_ids.min()}, {token_type_ids.max()}]")
        # print(f"Caption lengths: {cap_lens}")
        # print(f"Image input range: [{image_input.min()}, {image_input.max()}]")
        if torch.any(torch.isnan(input_ids)) or torch.any(torch.isinf(input_ids)):
            print("NaN or Inf in input_ids")
        if torch.any(torch.isnan(image_input)) or torch.any(torch.isinf(image_input)):
            print("NaN or Inf in image_input")
        input_ids, attention_mask, token_type_ids, cap_lens, image_input = (
            input_ids.to(device),
            attention_mask.to(device),
            token_type_ids.to(device),
            cap_lens.to(device),
            image_input.to(device),
        )
        # print(input_ids.shape, attention_mask.shape, image_input.shape)
        train_optimizer.zero_grad()
        loss = net(input_ids, attention_mask, token_type_ids, cap_lens, image_input)

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
    model = Gloria(
        L=args.loss_l,
        local_temp1=args.local_temp1,
        local_temp2=args.local_temp2,
        local_temp3=args.local_temp3,
    ).to(device)
    model.train()
    print(model)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9
    )
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
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
    #   if args.wandb:
    #       wandb.log({"loss": train_loss})
    # gc.collect()
    # torch.cuda.empty_cache()
    # model.save_pretrained("./save_pretrained/"+str(epoch))


# if args.wandb:
#    wandb.finish()
if __name__ == "__main__":
    main()
