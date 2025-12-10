import random
import re
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torchvision import transforms as T
from torchvision.models import resnet50
from torch.nn.utils.rnn import pad_sequence

from health_multimodal.image.model import ImageModel
from health_multimodal.text.model import CXRBertModel, CXRBertConfig, CXRBertTokenizer


class IShuffledContrastiveModel(nn.Module):
    """
    Identical Shuffled Contrastive model.
    """
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

        target = torch.arange(batch_size).long().to(text_embeds.device, non_blocking=True)
        
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
        targets = torch.zeros(batch_size, dtype=torch.long).to(image_embeds.device)
        loss = self.criterion(logits, targets)

        return loss

    def forward(self, input_ids, attention_mask, image_input):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input_ids = [i.to(device) for i in input_ids]
        attention_mask = [a.to(device) for a in attention_mask]

        shuffle_loss = self.shuffle_loss(input_ids, attention_mask, image_input)
        contrastive_loss = self.contrastive_loss(
            pad_sequence(input_ids[0], batch_first=True),
            pad_sequence(attention_mask[0], batch_first=True),
            image_input,
        )

        loss = (1 - self.T) * contrastive_loss + self.T * shuffle_loss
        return loss


class ShuffledContrastiveModel(nn.Module):
    """
    Shuffled Contrastive Model.
    """
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

        target = torch.arange(batch_size).long().to(text_embeds.device, non_blocking=True)
        loss = (
            self.criterion(logits_text_per_image, target)
            + self.criterion(logits_image_per_text, target)
        ) / 2
        return loss

    def shuffle_loss(self, input_ids_shuffled, attention_mask_shuffled, image_input):
        num_text = len(input_ids_shuffled)

        image_embeds = self.image_encoder(image_input).projected_global_embedding
        image_embeds = nn.functional.normalize(image_embeds, dim=1)
        
        input_ids = input_ids_shuffled[0]
        attention_mask = attention_mask_shuffled[0]

        input_ids = pad_sequence(input_ids, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)

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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input_ids = [i.to(device) for i in input_ids]
        attention_mask = [a.to(device) for a in attention_mask]

        shuffle_loss = self.shuffle_loss(input_ids, attention_mask, image_input)
        contrastive_loss = self.contrastive_loss(
            pad_sequence(input_ids[0], batch_first=True),
            pad_sequence(attention_mask[0], batch_first=True),
            image_input,
        )

        loss = contrastive_loss + shuffle_loss
        return loss


class ContrastiveModel(nn.Module):
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

        target = torch.arange(batch_size).long().to(text_embeds.device, non_blocking=True)
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


def _handle_image_4shuffle(x):
    return_image = False
    if not isinstance(x, torch.Tensor):
        t = torch.tensor(np.array(x)).unsqueeze(dim=0).float()
        t = t.permute(0, 3, 1, 2)
        return_image = True
        return t, return_image
    if len(x.shape) != 4:
        return x.unsqueeze(dim=0), return_image
    else:
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
        r"([.!"()*#:;~])",
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
        batch_size = image_input.shape[0]

        local_image_embeds = self.resnet_forward(image_input)
        local_image_embeds = self.local_embedder(local_image_embeds)

        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        all_embeddings = text_outputs[3]
        
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
        for i in range(words_emb.shape[0]):
            words_num = cap_lens[i] 
            word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()  # [1, 768, 25]
            word = word.repeat(batch_size, 1, 1)  # [48, 768, 25]
            
            context = local_image_embeds 
            
            weiContext, attn = attention_fn(
                word, context, temp1
            ) 

            word = word.transpose(1, 2).contiguous() 
            weiContext = weiContext.transpose(1, 2).contiguous() 

            word = word.view(batch_size * words_num, -1) 
            weiContext = weiContext.view(batch_size * words_num, -1) 

            row_sim = cosine_similarity(word, weiContext)
            row_sim = row_sim.view(batch_size, words_num) 

            row_sim.mul_(temp2).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True) 
            row_sim = torch.log(row_sim)
            similarities.append(row_sim)

        similarities = torch.cat(similarities, 1) 
        similarities = similarities * temp3
        similarities1 = similarities.transpose(0, 1) 
        
        labels = (
            torch.arange(batch_size).long().to(similarities.device, non_blocking=True)
        )

        loss0 = self.criterion(similarities, labels) 
        loss1 = self.criterion(similarities1, labels)

        loss = loss0 + loss1 / 2

        return loss

    def forward(self, input_ids, attention_mask, token_type_ids, cap_lens, image_input):
        loss_g = self.contrastive_loss(input_ids, attention_mask, image_input)
        loss_l = self.local_loss(
            input_ids, attention_mask, token_type_ids, cap_lens, image_input
        )
        loss = loss_g + self.L * loss_l

        return loss