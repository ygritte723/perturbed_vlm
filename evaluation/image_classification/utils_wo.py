import os
import config

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.nn.utils.rnn import pad_sequence

from health_multimodal.image.model import ImageModel
from health_multimodal.text.model import CXRBertModel, CXRBertConfig, CXRBertTokenizer


class IShuffledContrastiveModel(nn.Module):
    def __init__(self, T=0.5, shuffle_temp=0.07):
        super(IShuffledContrastiveModel, self).__init__()

        self.T = T

        print("T: ", T)
        print("shuffle_temp: ", shuffle_temp)

        # Text Encoder CXRBert
        config = CXRBertConfig(
            hidden_size=768,
            projection_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            output_attentions=True,
            return_dict=True,
        )
        self.text_encoder = CXRBertModel(config)
        # Image Encoder ResNet50
        self.image_encoder = ImageModel(
            img_encoder_type="resnet50", joint_feature_size=128
        )

        self.criterion = nn.CrossEntropyLoss()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_shuffle = nn.Parameter(
            torch.ones([]) * np.log(1 / shuffle_temp)
        )

    def contrastive_loss(self, input_ids, attention_mask, image_input):
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
        loss = (
            self.criterion(logits_text_per_image, target)
            + self.criterion(logits_image_per_text, target)
        ) / 2

        return loss

    def shuffle_loss(self, input_ids_shuffled, attention_mask_shuffled, image_input):
        batch_size = len(input_ids_shuffled[0])
        num_text = len(input_ids_shuffled)

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

        logits = torch.cat(logits, 1)
        # Compute the cross-entropy loss
        targets = torch.zeros(batch_size, dtype=torch.long).to(
            image_embeds.device
        )  
        loss = self.criterion(logits, targets)

        return loss

    def forward(self, input_ids, attention_mask, image_input):
        device = torch.device("cuda")
        input_ids = [i.to(device) for i in input_ids]
        attention_mask = [a.to(device) for a in attention_mask]

        shuffle_loss = self.shuffle_loss(input_ids, attention_mask, image_input)
        contrastive_loss = self.contrastive_loss(
            pad_sequence(input_ids[0], batch_first=True),
            pad_sequence(attention_mask[0], batch_first=True),
            image_input,
        )

        loss = contrastive_loss + self.T * shuffle_loss

        return loss


class ContrastiveModel(nn.Module):
    def __init__(self, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        super(ContrastiveModel, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        # Text Encoder CXRBert
        config = CXRBertConfig(
            hidden_size=768,
            projection_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            output_attentions=True,
            return_dict=True,
        )
        self.text_encoder = CXRBertModel(config)
        # Image Encoder ResNet50
        self.image_encoder = ImageModel(
            img_encoder_type="resnet50", joint_feature_size=128
        )

        self.criterion = nn.CrossEntropyLoss()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def contrastive_loss(self, input_ids, attention_mask, image_input):
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
        loss = (
            self.criterion(logits_text_per_image, target)
            + self.criterion(logits_image_per_text, target)
        ) / 2

        return loss

    def forward(self, input_ids, attention_mask, image_input):
        loss = self.contrastive_loss(input_ids, attention_mask, image_input)

        return loss


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