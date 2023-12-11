set -x

export MODEL_NAME=microsoft/BiomedVLP-CXR-BERT-specialized
export TASK_NAME=radnli
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1
export NOTICE=v5
export NOTICE1=pretrained
export NOTICE2=gshuffle
export NOTICE3=bt50s001

torchrun \
--nproc_per_node 4 \
--master_port 29502 \
  run_glue.py \
  --model_name_or_path $MODEL_NAME \
  --train_file /jet/home/lisun/work/tokenizer/shiba-main/data/${TASK_NAME}/radnli_dev_v1.jsonl \
  --validation_file /jet/home/lisun/work/tokenizer/shiba-main/data/${TASK_NAME}/radnli_test_v1.jsonl \
  --do_train \
  --do_eval \
  --max_seq_length 512 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --report_to "none" \
  --overwrite_output_dir \
  --save_total_limit 1 \
  --fp16 \
  --output_dir ./results/shuffled_model/${MODEL_NAME}/${TASK_NAME}/${NOTICE}/${NOTICE1}/${NOTICE2}/${NOTICE3}/ \
  --tokenizer_name microsoft/BiomedVLP-CXR-BERT-specialized \
  --model_revision v1.1 \
  --trust_remote_code \
  --resume /jet/home/lisun/work/xinliu/hi-ml/hi-ml-multimodal/src/new_caches_v7/T0.1_L0.1_shuffle-temp0.01/cache-2023-11-27-06-06-56-moco/model_last.pth
  #--resume /jet/home/lisun/work/xinliu/hi-ml/hi-ml-multimodal/src/new_caches_v4/bt12/cache-2023-10-12-23-41-39-moco/model_last.pth \

  #--resume_from_checkpoint ./results/bert-base-cased/${TASK_NAME}/ \
  #--random_init_model \