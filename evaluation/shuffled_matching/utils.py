import random
import re
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torchvision import transforms as T

from health_multimodal.image.model import ImageModel
from health_multimodal.text.model import CXRBertModel

# A lot of the approaches here are inspired from the wonderful paper from O'Connor and Andreas 2021.
# https://github.com/lingo-mit/context-ablations
#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

"""# Identical Shuffled Contrastive model"""


class IShuffledContrastiveModel(nn.Module):
    def __init__(self, T=0.5):
        super(IShuffledContrastiveModel, self).__init__()

        self.T = T
        # Text Encoder CXRBert
        self.text_encoder = CXRBertModel.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized", revision="v1.1"
        )
        # Image Encoder ResNet50
        self.image_encoder = ImageModel(
            img_encoder_type="resnet50", joint_feature_size=128
        )

        self.criterion = nn.CrossEntropyLoss()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.margin = 2.0

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

        # Compute the similarity matrix
        logits = torch.zeros((batch_size, num_text)).to(image_embeds.device)
        for txt in range(num_text):
            input_ids = input_ids_shuffled[txt]
            attention_mask = attention_mask_shuffled[txt]
            input_ids_tensor = pad_sequence(input_ids, batch_first=True)
            attention_mask_tensor = pad_sequence(attention_mask, batch_first=True)

            text_embeds = self.text_encoder.get_projected_text_embeddings(
                input_ids=input_ids_tensor, attention_mask=attention_mask_tensor
            )
            text_embeds = nn.functional.normalize(text_embeds, dim=1)

            logits_text_per_image = (
                self.logit_scale.exp() * image_embeds @ text_embeds.t()
            )
            logits_image_per_text = logits_text_per_image.t()

            logits[:, txt] = (
                torch.sum(logits_text_per_image, dim=1)
                + torch.sum(logits_image_per_text, dim=1)
            ) / 2

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
        device = torch.device("cuda:0")
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

        loss = (1 - self.T) * contrastive_loss + self.T * shuffle_loss
        # print("total_loss: ", loss)

        return loss


"""# Shuffled Contrastive model"""


# noinspection PyUnusedLocal
class ShuffledContrastiveModel(nn.Module):
    # noinspection PyUnusedLocal
    def __init__(self, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        super(ShuffledContrastiveModel, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        # Text Encoder CXRBert
        self.text_encoder = CXRBertModel.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized", revision="v1.1"
        )
        # Image Encoder ResNet50
        self.image_encoder = ImageModel(
            img_encoder_type="resnet50", joint_feature_size=128
        )

        self.criterion = nn.CrossEntropyLoss()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.margin = 2.0

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

    # noinspection PyUnusedLocal
    def shuffle_loss(self, input_ids_shuffled, attention_mask_shuffled, image_input):
        # https://jamesmccaffrey.wordpress.com/2022/03/04/contrastive-loss-function-in-pytorch/

        batch_size = len(input_ids_shuffled[0])
        num_text = len(input_ids_shuffled)
        # print("batch_size: ",batch_size)
        # print("num_text: ",num_text)

        image_embeds = self.image_encoder(image_input).projected_global_embedding
        image_embeds = nn.functional.normalize(image_embeds, dim=1)
        # Text encoding
        # input_ids = [i[0] for i in input_ids_shuffled]
        # attention_mask=[a[0] for a in attention_mask_shuffled]
        input_ids = input_ids_shuffled[0]
        attention_mask = attention_mask_shuffled[0]

        # Debugging: Print shapes before padding
        # for seq in input_ids:
        # print("Shape of input_ids before padding:", seq.shape)
        # for seq in attention_mask:
        # print("Shape of attention_mask before padding:", seq.shape)
        # Debugging: Print content of option_set before padding
        # for option_set in zip(*input_ids):
        # print("Content of option_set for input_ids:", option_set)
        # pad_sequence([seq for seq in option_set], batch_first=True)
        # for option_set in zip(*attention_mask):
        # print("Content of option_set for attention_mask:", option_set)
        # pad_sequence([seq for seq in option_set], batch_first=True)

        # input_ids = [pad_sequence([seq for seq in option_set], batch_first=True) for option_set in zip(*input_ids)]
        # attention_mask = [pad_sequence([seq for seq in option_set], batch_first=True) for option_set in zip(*attention_mask)]
        input_ids = pad_sequence(input_ids, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)

        # Debugging: Print shapes after padding
        # for seq in input_ids:
        # print("Shape of input_ids after padding:", seq.shape)
        # for seq in attention_mask:
        # print("Shape of attention_mask after padding:", seq.shape)

        input_ids_tensor = input_ids
        attention_mask_tensor = attention_mask

        text_embeds = self.text_encoder.get_projected_text_embeddings(
            input_ids=input_ids_tensor, attention_mask=attention_mask_tensor
        )
        text_embeds = nn.functional.normalize(text_embeds, dim=1)
        euc_dist = nn.functional.pairwise_distance(text_embeds, image_embeds)
        ori_sim = torch.mean(torch.pow(euc_dist, 2))

        loss = 0
        shu_sim = 0

        for txt in range(1, num_text):
            # Text encoding
            # input_ids = [i[txt] for i in input_ids_shuffled]
            # attention_mask=[a[txt] for a in attention_mask_shuffled]

            # input_ids = [pad_sequence([seq for seq in option_set], batch_first=True) for option_set in zip(*input_ids)]
            # attention_mask = [pad_sequence([seq for seq in option_set], batch_first=True) for option_set in zip(*attention_mask)]

            # input_ids_tensor = torch.stack(input_ids)
            # attention_mask_tensor = torch.stack(attention_mask)
            input_ids = input_ids_shuffled[txt]
            attention_mask = attention_mask_shuffled[txt]
            input_ids_tensor = pad_sequence(input_ids, batch_first=True)
            attention_mask_tensor = pad_sequence(attention_mask, batch_first=True)

            text_embeds = self.text_encoder.get_projected_text_embeddings(
                input_ids=input_ids_tensor, attention_mask=attention_mask_tensor
            )
            text_embeds = nn.functional.normalize(text_embeds, dim=1)
            euc_dist = nn.functional.pairwise_distance(text_embeds, image_embeds)
            delta = self.margin - euc_dist  # sort of reverse distance
            delta = torch.clamp(delta, min=0.0, max=None)
            sim = torch.mean(torch.pow(delta, 2))
            shu_sim += sim
        loss = ori_sim * shu_sim

        return loss

    def forward(self, input_ids, attention_mask, image_input):
        # transpose
        # input_ids = torch.transpose(input_ids, 0, 1)
        # attention_mask = torch.transpose(attention_mask, 0, 1)
        # input_ids = np.array(input_ids).T.tolist()
        # attention_mask = np.array(attention_mask).T.tolist()
        device = torch.device("cuda:0")
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

        loss = contrastive_loss + shuffle_loss

        return loss


# noinspection PyUnusedLocal
class ContrastiveModel(nn.Module):
    # noinspection PyUnusedLocal
    def __init__(self, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        super(ContrastiveModel, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        # Text Encoder CXRBert
        self.text_encoder = CXRBertModel.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized", revision="v1.1"
        )
        # Image Encoder ResNet50
        self.image_encoder = ImageModel(
            img_encoder_type="resnet50", joint_feature_size=128
        )

        self.criterion = nn.CrossEntropyLoss()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

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

    def forward(self, input_ids, attention_mask, image_input):
        loss = self.contrastive_loss(input_ids, attention_mask, image_input)

        return loss


def get_text_perturb_fn(text_perturb_fn):
    if text_perturb_fn == "shuffle_nouns_and_adj":
        return shuffle_nouns_and_adj
    elif text_perturb_fn == "shuffle_allbut_nouns_and_adj":
        return shuffle_allbut_nouns_and_adj
    elif text_perturb_fn == "shuffle_within_trigrams":
        return shuffle_within_trigrams
    elif text_perturb_fn == "shuffle_all_words":
        return shuffle_all_words
    elif text_perturb_fn == "shuffle_trigrams":
        return shuffle_trigrams

    # newly added
    elif text_perturb_fn == "swap_adjacent_words":
        return swap_adjacent_words
    elif text_perturb_fn == "shuffle_all_words":
        return shuffle_all_words
    elif text_perturb_fn == "reverse_sentence":
        return reverse_sentence
    elif text_perturb_fn == "shuffle_nouns_verbs_adj":
        return shuffle_nouns_verbs_adj
    elif text_perturb_fn == "replace_adjectives_with_antonyms":
        return replace_adjectives_with_antonyms

    elif text_perturb_fn is None:
        return None
    else:
        print(
            "Unknown text perturbation function: {}, returning None".format(
                text_perturb_fn
            )
        )
        return None


def get_image_perturb_fn(image_perturb_fn):
    if image_perturb_fn == "shuffle_rows_4":
        return partial(shuffle_rows, n_rows=4)
    elif image_perturb_fn == "shuffle_patches_9":
        return partial(shuffle_patches, n_ratio=3)
    elif image_perturb_fn == "shuffle_cols_4":
        return partial(shuffle_columns, n_cols=4)
    elif image_perturb_fn is None:
        return None
    else:
        print(
            "Unknown image perturbation function: {}, returning None".format(
                image_perturb_fn
            )
        )
        return None


class TextShuffler:
    def __init__(self):
        import spacy

        self.nlp = spacy.load("en_core_web_sm")

    def shuffle_nouns_and_adj(self, ex):
        doc = self.nlp(ex)
        tokens = [token.text for token in doc]
        text = np.array(tokens)
        noun_idx = [
            i
            for i, token in enumerate(doc)
            if token.tag_ in ["NN", "NNS", "NNP", "NNPS"]
        ]
        ## Finding adjectives
        adjective_idx = [
            i for i, token in enumerate(doc) if token.tag_ in ["JJ", "JJR", "JJS"]
        ]
        ## Shuffle the nouns of the text
        text[noun_idx] = np.random.permutation(text[noun_idx])
        ## Shuffle the adjectives of the text
        text[adjective_idx] = np.random.permutation(text[adjective_idx])

        return " ".join(text)

    def shuffle_all_words(self, ex):
        return " ".join(np.random.permutation(ex.split(" ")))

    def shuffle_allbut_nouns_and_adj(self, ex):
        doc = self.nlp(ex)
        tokens = [token.text for token in doc]
        text = np.array(tokens)
        noun_adj_idx = [
            i
            for i, token in enumerate(doc)
            if token.tag_ in ["NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"]
        ]
        ## Finding adjectives

        else_idx = np.ones(text.shape[0])
        else_idx[noun_adj_idx] = 0

        else_idx = else_idx.astype(bool)
        ## Shuffle everything that are nouns or adjectives
        text[else_idx] = np.random.permutation(text[else_idx])
        return " ".join(text)

    def get_trigrams(self, sentence):
        # Taken from https://github.com/lingo-mit/context-ablations/blob/478fb18a9f9680321f0d37dc999ea444e9287cc0/code/transformers/src/transformers/data/data_augmentation.py
        trigrams = []
        trigram = []
        for i in range(len(sentence)):
            trigram.append(sentence[i])
            if i % 3 == 2:
                trigrams.append(trigram[:])
                trigram = []
        if trigram:
            trigrams.append(trigram)
        return trigrams

    def trigram_shuffle(self, sentence):
        trigrams = self.get_trigrams(sentence)
        for trigram in trigrams:
            random.shuffle(trigram)
        return " ".join([" ".join(trigram) for trigram in trigrams])

    def shuffle_within_trigrams(self, ex):
        import nltk

        tokens = nltk.word_tokenize(ex)
        shuffled_ex = self.trigram_shuffle(tokens)
        return shuffled_ex

    def shuffle_trigrams(self, ex):
        import nltk

        tokens = nltk.word_tokenize(ex)
        trigrams = self.get_trigrams(tokens)
        random.shuffle(trigrams)
        shuffled_ex = " ".join([" ".join(trigram) for trigram in trigrams])
        return shuffled_ex

    # Newly added funcs
    def reverse_sentence(self, ex):
        """
        Reverse the order of words in the sentence.
        """
        return " ".join(reversed(ex.split()))

    def shuffle_nouns_verbs_adj(self, ex):
        """
        Shuffle nouns, verbs, and adjectives within the text.
        """
        doc = self.nlp(ex)
        tokens = [token.text for token in doc]
        text = np.array(tokens)
        idx = [
            i
            for i, token in enumerate(doc)
            if token.tag_
            in [
                "NN",
                "NNS",
                "NNP",
                "NNPS",
                "JJ",
                "JJR",
                "JJS",
                "VB",
                "VBD",
                "VBG",
                "VBN",
                "VBP",
                "VBZ",
            ]
        ]
        text[idx] = np.random.permutation(text[idx])
        return " ".join(text)

    def replace_adjectives_with_antonyms(self, ex):
        """
        Replace adjectives in the sentence with their antonyms.
        """
        from nltk.corpus import wordnet

        doc = self.nlp(ex)
        new_sentence = []
        for token in doc:
            if token.pos_ == "ADJ":
                antonyms = []
                for syn in wordnet.synsets(token.text):
                    if syn.pos() == wordnet.ADJ:
                        for lemma in syn.lemmas():
                            if lemma.antonyms():
                                antonyms.append(lemma.antonyms()[0].name())
                # Choose an antonym if available, otherwise use original word
                new_word = random.choice(antonyms) if antonyms else token.text
                new_sentence.append(new_word)
            else:
                new_sentence.append(token.text)
        return " ".join(new_sentence)

    def swap_adjacent_words(self, ex):
        """
        Swap adjacent words in the sentence.
        """
        words = ex.split()
        swapped_sentence = []
        i = 0
        while i < len(words):
            if i + 1 < len(words):
                swapped_sentence.append(words[i + 1])
                swapped_sentence.append(words[i])
                i += 2
            else:
                swapped_sentence.append(words[i])
                i += 1
        return " ".join(swapped_sentence)


def _handle_image_4shuffle(x):
    return_image = False
    if not isinstance(x, torch.Tensor):
        # print(f"x is not a tensor: {type(x)}. Trying to handle but fix this or I'll annoy you with this log")
        t = torch.tensor(np.array(x)).unsqueeze(dim=0).float()
        t = t.permute(0, 3, 1, 2)
        return_image = True
        return t, return_image
    if len(x.shape) != 4:
        # print("You did not send a tensor of shape NxCxWxH. Unsqueezing not but fix this or I'll annoy you with this log")
        return x.unsqueeze(dim=0), return_image
    else:
        # Good boi
        return x, return_image


def shuffle_rows(x, n_rows=7):
    """
    Shuffle the rows of the image tensor where each row has a size of 14 pixels.
    Tensor is of shape N x C x W x H
    """
    x, return_image = _handle_image_4shuffle(x)
    patch_size = x.shape[-2] // n_rows
    u = nnf.unfold(
        x, kernel_size=(patch_size, x.shape[-1]), stride=patch_size, padding=0
    )
    # permute the patches of each image in the batch
    pu = torch.cat([b_[:, torch.randperm(b_.shape[-1])][None, ...] for b_ in u], dim=0)
    # fold the permuted patches back together
    f = nnf.fold(
        pu,
        x.shape[-2:],
        kernel_size=(patch_size, x.shape[-1]),
        stride=patch_size,
        padding=0,
    )

    image = f.squeeze()  # C W H
    if return_image:
        return T.ToPILImage()(image.type(torch.uint8))
    else:
        return image


def shuffle_columns(x, n_cols=7):
    """
    Shuffle the columns of the image tensor where we'll have n_cols columns.
    Tensor is of shape N x C x W x H
    """
    x, return_image = _handle_image_4shuffle(x)
    patch_size = x.shape[-1] // n_cols
    u = nnf.unfold(
        x, kernel_size=(x.shape[-2], patch_size), stride=patch_size, padding=0
    )
    # permute the patches of each image in the batch
    pu = torch.cat([b_[:, torch.randperm(b_.shape[-1])][None, ...] for b_ in u], dim=0)
    # fold the permuted patches back together
    f = nnf.fold(
        pu,
        x.shape[-2:],
        kernel_size=(x.shape[-2], patch_size),
        stride=patch_size,
        padding=0,
    )
    image = f.squeeze()  # C W H
    if return_image:
        return T.ToPILImage()(image.type(torch.uint8))
    else:
        return image


def shuffle_patches(x, n_ratio=4):
    """
    Shuffle the rows of the image tensor where each row has a size of 14 pixels.
    Tensor is of shape N x C x W x H
    """
    x, return_image = _handle_image_4shuffle(x)
    patch_size_x = x.shape[-2] // n_ratio
    patch_size_y = x.shape[-1] // n_ratio
    u = nnf.unfold(
        x,
        kernel_size=(patch_size_x, patch_size_y),
        stride=(patch_size_x, patch_size_y),
        padding=0,
    )
    # permute the patches of each image in the batch
    pu = torch.cat([b_[:, torch.randperm(b_.shape[-1])][None, ...] for b_ in u], dim=0)
    # fold the permuted patches back together
    f = nnf.fold(
        pu,
        x.shape[-2:],
        kernel_size=(patch_size_x, patch_size_y),
        stride=(patch_size_x, patch_size_y),
        padding=0,
    )
    image = f.squeeze()  # C W H
    if return_image:
        return T.ToPILImage()(image.type(torch.uint8))
    else:
        return image


def pre_caption(caption, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        " ",
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        " ",
        caption,
    )
    caption = caption.rstrip("\n")
    caption = caption.strip(" ")

    # truncate caption
    caption_words = caption.split(" ")
    if len(caption_words) > max_words:
        caption = " ".join(caption_words[:max_words])

    return caption
