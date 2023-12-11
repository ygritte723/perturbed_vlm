---
license: mit
base_model: microsoft/BiomedVLP-CXR-BERT-specialized
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: batch12_t
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# batch12_t

This model is a fine-tuned version of [microsoft/BiomedVLP-CXR-BERT-specialized](https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.7559
- Accuracy: 0.6396

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 2
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 4
- gradient_accumulation_steps: 2
- total_train_batch_size: 16
- total_eval_batch_size: 32
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 5.0

### Training results



### Framework versions

- Transformers 4.31.0
- Pytorch 2.0.1+cu117
- Datasets 2.14.4
- Tokenizers 0.13.3
