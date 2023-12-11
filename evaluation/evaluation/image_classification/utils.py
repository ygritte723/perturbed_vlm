import torch
import random
import numpy as np
from functools import partial
import torch.nn.functional as nnf
import torch.nn as nn
from torchvision import transforms as T
from pathlib import Path

import pydicom as dicom
from PIL import Image
import re

from typing import Callable, Sequence, Optional, Tuple, Any, Union
from torchvision.models import resnet50

from health_multimodal.image.model import ImageModel
from health_multimodal.text.model import CXRBertModel, CXRBertTokenizer
from torch.utils.data import Dataset
import csv

# A lot of the approaches here are inspired from the wonderful paper from O'Connor and Andreas 2021.
# https://github.com/lingo-mit/context-ablations
#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

class CheXpertDataSet(Dataset):
    def __init__(self, data_PATH, transform = None, policy = "ones"):
        """
        data_PATH: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        Upolicy: name the policy with regard to the uncertain labels.
        """
        image_names = []
        labels = []
        self.dict = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
                     {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}, ]
        self.strange_list = ['CheXpert-v1.0-small/train/patient00773/study2/view1_frontal.jpg', \
                            'CheXpert-v1.0-small/train/patient00770/study1/view1_frontal.jpg', \
                            'CheXpert-v1.0-small/train/patient34662/study18/view1_frontal.jpg']

        with open(data_PATH, "r") as f:
            header = f.readline().strip('\n').split(',')
            self._label_header = [
                header[7],
                header[10],
                header[11],
                header[13],
                header[15]]
            #print(self._label_header)
            csvReader = csv.reader(f)
            #next(csvReader, None) # skip the header
            for line in csvReader:
                label_list = []
                image_name = line[0]

                label = line[5:]
                
                #for i in range(2):
                 #   if label[i]:
                 #       #print(label[i])
                 #       a = float(label[i])
                 #       if a == 1:
                 #           label[i] = 1
                 #       else:
                 #           label[i] = 0
                 #   else:
                 #       label[i] = 0
                for index, value in enumerate(label):
                    if index == 5 or index == 8:
                        label_list.append(self.dict[1].get(value))
                    elif index == 2 or index == 6 or index == 10:
                        label_list.append(self.dict[0].get(value))
                label_list = list(map(int, label_list))

                if image_name in self.strange_list :
                    pass
                else:
                    image_names.append(image_name)
                    labels.append(label_list)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        image_name = self.image_names[index]
        image_name = "/jet/home/lisun/work/xinliu/images/" + image_name
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label), image_name

    def __len__(self):
        return len(self.image_names)

class ExpandChannels:
    """
    Transforms an image with one channel to an image with three channels by copying
    pixel intensities of the image along the 1st dimension.
    """

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        :param data: Tensor of shape [1, H, W].
        :return: Tensor with channel copied three times, shape [3, H, W].
        """
        if data.shape[0] != 1:
            raise ValueError(f"Expected input of shape [1, H, W], found {data.shape}")
        return torch.repeat_interleave(data, 3, dim=0)

"""# Identical Shuffled Contrastive model"""

class IShuffledContrastiveModel(nn.Module):
    def __init__(self, T=0.5, shuffle_temp=0.07):
        super(IShuffledContrastiveModel, self).__init__()

        self.T = T
        print("T: ", T)
        print("shuffle_temp: ", shuffle_temp)
        
        # Text Encoder CXRBert
        self.text_encoder = CXRBertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", revision="v1.1")
        # Image Encoder ResNet50
        self.image_encoder = ImageModel(img_encoder_type="resnet50", joint_feature_size=128)

     
        self.criterion = nn.CrossEntropyLoss()
        
        self.logit_scale = (nn.Parameter(torch.ones([]) * np.log(1 / 0.07)))
        self.logit_scale_shuffle = (nn.Parameter(torch.ones([]) * np.log(1 / shuffle_temp)))
 
       
    def contrastive_loss(self, input_ids, attention_mask, image_input):
        #print('image_input:', image_input.shape, image_input.min(), image_input.max())
        batch_size = input_ids.shape[0]
        
        # Text encoding
        text_embeds = self.text_encoder.get_projected_text_embeddings(
                            input_ids=input_ids,
                            attention_mask=attention_mask)

        # Image encoding
        image_embeds = self.image_encoder(image_input).projected_global_embedding
        
        text_embeds = nn.functional.normalize(text_embeds, dim=1)
        image_embeds = nn.functional.normalize(image_embeds, dim=1)
        
        logits_text_per_image = self.logit_scale.exp() * image_embeds @ text_embeds.t()
        logits_image_per_text = logits_text_per_image.t()
        
        target = torch.arange(batch_size).long().to(text_embeds.device, non_blocking = True)
        #print(text_embeds.device, image_embeds.device, self.device, logits_text_per_image.device)
        loss = (self.criterion(logits_text_per_image, target) + self.criterion(logits_image_per_text, target)) / 2

        return loss
        
    def shuffle_loss(self, input_ids_shuffled, attention_mask_shuffled, image_input):
        

        batch_size = len(input_ids_shuffled[0])
        num_text = len(input_ids_shuffled)
        #print("batch_size: ",batch_size)
        #print("num_text: ",num_text)

        
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
                                input_ids=input_ids_tensor,
                                attention_mask=attention_mask_tensor)
            text_embeds = nn.functional.normalize(text_embeds, dim=1)
            
            # [Batch_size, Dim] * [Batch_size, Dim] -> [Batch_size, Dim]
            logits_i = self.logit_scale_shuffle.exp() * image_embeds * text_embeds            
            logits.append(torch.sum(logits_i, dim=1)[:,None])

        logits = torch.concatenate(logits, 1)
        # Compute the cross-entropy loss
        targets = torch.zeros(batch_size, dtype=torch.long).to(image_embeds.device)  # the original text is always at index 0
        loss = self.criterion(logits, targets)
  
        return loss       

    def forward(self, input_ids, attention_mask, image_input):
        # transpose 
        #input_ids = torch.transpose(input_ids, 0, 1)
        #attention_mask = torch.transpose(attention_mask, 0, 1)
        #input_ids = np.array(input_ids).T.tolist()
        #attention_mask = np.array(attention_mask).T.tolist()
        device = torch.device('cuda')
        input_ids = [i.to(device) for i in input_ids]
        attention_mask = [a.to(device) for a in attention_mask]
        
        
        shuffle_loss = self.shuffle_loss(input_ids, attention_mask, image_input)
        #print("shuffle_loss: ", shuffle_loss)
        contrastive_loss = self.contrastive_loss(pad_sequence(input_ids[0], batch_first=True), pad_sequence(attention_mask[0], batch_first=True), image_input)
        #print("contrastive_loss: ", contrastive_loss)

        loss = contrastive_loss + self.T*shuffle_loss
        #print("total_loss: ", loss)

        return loss

class ContrastiveModel(nn.Module):
    def __init__(self, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        super(ContrastiveModel, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        # Text Encoder CXRBert
        self.text_encoder = CXRBertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", revision="v1.1")
        # Image Encoder ResNet50
        self.image_encoder = ImageModel(img_encoder_type="resnet50", joint_feature_size=128)

        self.criterion = nn.CrossEntropyLoss()
        
        self.logit_scale = (nn.Parameter(torch.ones([]) * np.log(1 / 0.07)))

    def contrastive_loss(self, input_ids, attention_mask, image_input):
        #print('image_input:', image_input.shape, image_input.min(), image_input.max())
        batch_size = input_ids.shape[0]
        
        # Text encoding
        text_embeds = self.text_encoder.get_projected_text_embeddings(
                            input_ids=input_ids,
                            attention_mask=attention_mask)

        # Image encoding
        image_embeds = self.image_encoder(image_input).projected_global_embedding
        
        text_embeds = nn.functional.normalize(text_embeds, dim=1)
        image_embeds = nn.functional.normalize(image_embeds, dim=1)
        
        logits_text_per_image = self.logit_scale.exp() * image_embeds @ text_embeds.t()
        logits_image_per_text = logits_text_per_image.t()
        
        target = torch.arange(batch_size).long().to(text_embeds.device, non_blocking = True)
        #print(text_embeds.device, image_embeds.device, self.device, logits_text_per_image.device)
        loss = (self.criterion(logits_text_per_image, target) + self.criterion(logits_image_per_text, target)) / 2

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


class Gloria(nn.Module):
    def __init__(self, L=0.5, last_n_layer = 4, local_temp1=4.0, \
                local_temp2=5.0, local_temp3=10.0):
        super(Gloria, self).__init__()

        self.L = L
        self.local_temp1 = local_temp1
        self.local_temp2 = local_temp2
        self.local_temp3 = local_temp3
        self.last_n_layer = last_n_layer        # Text Encoder CXRBert
        #self.text_encoder = CXRBertModel.from_pretrained(BIOMED_VLP_CXR_BERT_SPECIALIZED, revision=CXR_BERT_COMMIT_TAG)
        self.text_encoder = CXRBertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", revision="v1.1")
        # Image Encoder ResNet50
        self.image_encoder = ImageModel(img_encoder_type="resnet50", joint_feature_size=128)
        self.local_up = nn.Upsample(size=(299, 299), mode="bilinear", align_corners=True)
        self.local_image_encoder = resnet50()
        self.local_image_encoder.fc = Identity()
        self.local_embedder = nn.Conv2d(
            1024,
            768,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False, )

        #self.local_text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", output_hidden_states=True)
        #self.local_text_encoder = self.text_encoder
        #self.idxtoword = {v: k for k, v in AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").get_vocab().items()}
        self.idxtoword = {v: k for k, v in CXRBertTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", revision="v1.1").get_vocab().items()}
        self.criterion = nn.CrossEntropyLoss()
        
        self.logit_scale = (nn.Parameter(torch.ones([]) * np.log(1 / 0.07)))
    
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
        #print('image_input:', image_input.shape, image_input.min(), image_input.max())
        batch_size = input_ids.shape[0]
        
        # Text encoding
        text_embeds = self.text_encoder.get_projected_text_embeddings(
                            input_ids=input_ids,
                            attention_mask=attention_mask)

        # Image encoding
        image_embeds = self.image_encoder(image_input).projected_global_embedding
        
        text_embeds = nn.functional.normalize(text_embeds, dim=1)
        image_embeds = nn.functional.normalize(image_embeds, dim=1)
        
        logits_text_per_image = self.logit_scale.exp() * image_embeds @ text_embeds.t()
        logits_image_per_text = logits_text_per_image.t()
        
        target = torch.arange(batch_size).long().to(text_embeds.device, non_blocking = True)
        #print(text_embeds.device, image_embeds.device, self.device, logits_text_per_image.device)
        loss = (self.criterion(logits_text_per_image, target) + self.criterion(logits_image_per_text, target)) / 2
        #print('global loss: ', str(loss))
        return loss
    
    def local_loss(self, input_ids, attention_mask, token_type_ids, cap_lens, image_input, temp1=4.0, temp2=5.0, temp3=10.0):
        #print('image_input:', image_input.shape, image_input.min(), image_input.max())
        batch_size = image_input.shape[0]
        
        #local_image_embeds = self.local_image_encoder(image_input, return_intermediate_layers=True)[3]
        local_image_embeds = self.resnet_forward(image_input)
        if torch.any(torch.isnan(local_image_embeds)) or torch.any(torch.isinf(local_image_embeds)):
            print("NaN or Inf in image_input after encoder")
        local_image_embeds = self.local_embedder(local_image_embeds)
        if torch.any(torch.isnan(local_image_embeds)) or torch.any(torch.isinf(local_image_embeds)):
            print("NaN or Inf in local_image_embeds after embedder")

        #text_outputs = self.local_text_encoder(input_ids=input_ids, 
        text_outputs = self.text_encoder(input_ids=input_ids,
                                        attention_mask=attention_mask, 
                                        token_type_ids=token_type_ids, 
                                        #return_dict=True,
                                        output_hidden_states=True,
                                        )
        all_embeddings = text_outputs[3]
        #print(len(all_embeddings))
        # (batch_size, sequence_length, hidden_size)
        embeddings = torch.stack(all_embeddings[-self.last_n_layer :])  # layers, batch, sent_len, embedding size

        embeddings = embeddings.permute(1, 0, 2, 3)

        embeddings, sents = self.aggregate_tokens(embeddings, token_type_ids)

        word_embeddings = embeddings.sum(axis=1)

        batch_dim, num_words, feat_dim = word_embeddings.shape
        word_embeddings = word_embeddings.view(batch_dim * num_words, feat_dim)
        word_embeddings = word_embeddings.view(batch_dim, num_words, 768)
        word_embeddings = word_embeddings.permute(0, 2, 1)
        word_embeddings = word_embeddings / (torch.norm(word_embeddings, 2, dim=1, keepdim=True).expand_as(word_embeddings)+1e-6)
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
            #print(word.shape)
            context = local_image_embeds  # [48, 768, 19, 19]
            #print(context.shape)
            #print("context: ", str(context.shape))
            #print("word: ", str(word.shape))
            #print("Attn: ", str(attn))
            #print("word: ", str(word))
            weiContext, attn = attention_fn(word, context, temp1)  # [48, 768, 25], [48, 25, 19, 19]
            #print("weiContext: ", str(weiContext.shape))
            #print("Attn: ", str(attn.shape))
            

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
        #print('Similarity: ', str(similarities))
        #print('Similarity1: ', str(similarities1))
        labels = torch.arange(batch_size).long().to(similarities.device, non_blocking = True)

        loss0 = self.criterion(similarities, labels)  # labels: arange(batch_size)
        loss1 = self.criterion(similarities1, labels)

        loss = loss0 + loss1 / 2
        #print('local loss: ', str(loss))

        return loss
        

        
        
    def forward(self, input_ids, attention_mask, token_type_ids, cap_lens, image_input):

        loss_g = self.contrastive_loss(input_ids, attention_mask, image_input)
        loss_l = self.local_loss(input_ids, attention_mask, token_type_ids, cap_lens, image_input)
        loss = loss_g + self.L*loss_l

        return loss
       

class GShuffle(nn.Module):
    def __init__(self, T=0.5, shuffle_temp=0.07, \
                L=0.5, last_n_layer = 4, local_temp1=4.0, \
                local_temp2=5.0, local_temp3=10.0):
        super(GShuffle, self).__init__()

        self.T = T
        self.L = L
        self.local_temp1 = local_temp1
        self.local_temp2 = local_temp2
        self.local_temp3 = local_temp3
        self.last_n_layer = last_n_layer
        
        #print("T: ", T)
        #print("L: ", L)
        #print("shuffle_temp: ", shuffle_temp)

        # Text Encoder CXRBert
        self.text_encoder = CXRBertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", revision="v1.1")
        # Image Encoder ResNet50
        self.image_encoder = ImageModel(img_encoder_type="resnet50", joint_feature_size=128)

        self.criterion = nn.CrossEntropyLoss()
        
        self.logit_scale = (nn.Parameter(torch.ones([]) * np.log(1 / 0.07)))
        self.logit_scale_shuffle = (nn.Parameter(torch.ones([]) * np.log(1 / shuffle_temp)))

        self.local_up = nn.Upsample(size=(299, 299), mode="bilinear", align_corners=True)
        self.local_image_encoder = resnet50()
        self.local_image_encoder.fc = Identity()
        self.local_embedder = nn.Conv2d(
            1024,
            768,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False, )
        self.idxtoword = {v: k for k, v in CXRBertTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", revision="v1.1").get_vocab().items()}
        self.criterion = nn.CrossEntropyLoss() 
       
    def contrastive_loss(self, input_ids, attention_mask, image_input):
        #print('image_input:', image_input.shape, image_input.min(), image_input.max())
        batch_size = input_ids.shape[0]
        
        # Text encoding
        text_embeds = self.text_encoder.get_projected_text_embeddings(
                            input_ids=input_ids,
                            attention_mask=attention_mask)

        # Image encoding
        image_embeds = self.image_encoder(image_input).projected_global_embedding
        
        text_embeds = nn.functional.normalize(text_embeds, dim=1)
        image_embeds = nn.functional.normalize(image_embeds, dim=1)
        
        logits_text_per_image = self.logit_scale.exp() * image_embeds @ text_embeds.t()
        logits_image_per_text = logits_text_per_image.t()
        
        target = torch.arange(batch_size).long().to(text_embeds.device, non_blocking = True)
        #print(text_embeds.device, image_embeds.device, self.device, logits_text_per_image.device)
        loss = (self.criterion(logits_text_per_image, target) + self.criterion(logits_image_per_text, target)) / 2

        return loss
        
    def shuffle_loss(self, input_ids_shuffled, attention_mask_shuffled, image_input):
        

        batch_size = len(input_ids_shuffled[0])
        num_text = len(input_ids_shuffled)
        #print("batch_size: ",batch_size)
        #print("num_text: ",num_text)

        
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
                                input_ids=input_ids_tensor,
                                attention_mask=attention_mask_tensor)
            text_embeds = nn.functional.normalize(text_embeds, dim=1)
            
            # [Batch_size, Dim] * [Batch_size, Dim] -> [Batch_size, Dim]
            logits_i = self.logit_scale_shuffle.exp() * image_embeds * text_embeds            
            logits.append(torch.sum(logits_i, dim=1)[:,None])

        logits = torch.cat(logits, 1)
        # Compute the cross-entropy loss
        targets = torch.zeros(batch_size, dtype=torch.long).to(image_embeds.device)  # the original text is always at index 0
        loss = self.criterion(logits, targets)
  
        return loss       
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
        
    def local_loss(self, input_ids, attention_mask, token_type_ids, cap_lens, image_input, temp1=4.0, temp2=5.0, temp3=10.0):
        #print('image_input:', image_input.shape, image_input.min(), image_input.max())
        batch_size = image_input.shape[0]
        
        #local_image_embeds = self.local_image_encoder(image_input, return_intermediate_layers=True)[3]
        local_image_embeds = self.resnet_forward(image_input)
        if torch.any(torch.isnan(local_image_embeds)) or torch.any(torch.isinf(local_image_embeds)):
            print("NaN or Inf in image_input after encoder")
        local_image_embeds = self.local_embedder(local_image_embeds)
        if torch.any(torch.isnan(local_image_embeds)) or torch.any(torch.isinf(local_image_embeds)):
            print("NaN or Inf in local_image_embeds after embedder")

        #text_outputs = self.local_text_encoder(input_ids=input_ids, 
        text_outputs = self.text_encoder(input_ids=input_ids,
                                        attention_mask=attention_mask, 
                                        token_type_ids=token_type_ids, 
                                        #return_dict=True,
                                        output_hidden_states=True,
                                        )
        all_embeddings = text_outputs[3]
        #print(len(all_embeddings))
        # (batch_size, sequence_length, hidden_size)
        embeddings = torch.stack(all_embeddings[-self.last_n_layer :])  # layers, batch, sent_len, embedding size

        embeddings = embeddings.permute(1, 0, 2, 3)

        embeddings, sents = self.aggregate_tokens(embeddings, token_type_ids)

        word_embeddings = embeddings.sum(axis=1)

        batch_dim, num_words, feat_dim = word_embeddings.shape
        word_embeddings = word_embeddings.view(batch_dim * num_words, feat_dim)
        word_embeddings = word_embeddings.view(batch_dim, num_words, 768)
        word_embeddings = word_embeddings.permute(0, 2, 1)
        word_embeddings = word_embeddings / (torch.norm(word_embeddings, 2, dim=1, keepdim=True).expand_as(word_embeddings)+1e-6)
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
            #print(word.shape)
            context = local_image_embeds  # [48, 768, 19, 19]
            #print(context.shape)
            #print("context: ", str(context.shape))
            #print("word: ", str(word.shape))
            #print("Attn: ", str(attn))
            #print("word: ", str(word))
            weiContext, attn = attention_fn(word, context, temp1)  # [48, 768, 25], [48, 25, 19, 19]
            #print("weiContext: ", str(weiContext.shape))
            #print("Attn: ", str(attn.shape))
            

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
        #print('Similarity: ', str(similarities))
        #print('Similarity1: ', str(similarities1))
        labels = torch.arange(batch_size).long().to(similarities.device, non_blocking = True)

        loss0 = self.criterion(similarities, labels)  # labels: arange(batch_size)
        loss1 = self.criterion(similarities1, labels)

        loss = loss0 + loss1 / 2
        #print('local loss: ', str(loss))

        return loss
        

    
    def forward(self, input_ids, attention_mask, token_type_ids, cap_lens, image_input):
        # transpose 
        #input_ids = torch.transpose(input_ids, 0, 1)
        #attention_mask = torch.transpose(attention_mask, 0, 1)
        #input_ids = np.array(input_ids).T.tolist()
        #attention_mask = np.array(attention_mask).T.tolist()
        device = torch.device('cuda')
        input_ids = [i.to(device) for i in input_ids]
        attention_mask = [a.to(device) for a in attention_mask]
        
        
        shuffle_loss = self.shuffle_loss(input_ids, attention_mask, image_input)
        #print("shuffle_loss: ", shuffle_loss)
        contrastive_loss = self.contrastive_loss(pad_sequence(input_ids[0], batch_first=True), \
                                                 pad_sequence(attention_mask[0], batch_first=True), image_input)
        #print("contrastive_loss: ", contrastive_loss)
        loss_l = self.local_loss(pad_sequence(input_ids[0], batch_first=True), \
                                 pad_sequence(attention_mask[0], batch_first=True), \
                                 pad_sequence(token_type_ids, batch_first=True), \
                                 cap_lens, image_input)

        loss = contrastive_loss + self.T*shuffle_loss + self.L*loss_l
        #print("total_loss: ", loss)

        return loss
