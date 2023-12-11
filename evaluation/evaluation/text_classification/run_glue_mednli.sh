set -x

export MODEL_NAME=microsoft/BiomedVLP-CXR-BERT-specialized
export TASK_NAME=mednli
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1
export NOTICE=batch64-wo

torchrun \
--nproc_per_node 4 \
--master_port 29502 \
  run_glue.py \
  --model_name_or_path $MODEL_NAME \
  --train_file /jet/home/lisun/work/tokenizer/shiba-main/data/${TASK_NAME}/mli_train_v1.jsonl \
  --validation_file /jet/home/lisun/work/tokenizer/shiba-main/data/${TASK_NAME}/mli_test_v1.jsonl \
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
  --output_dir ./results/${MODEL_NAME}/${TASK_NAME}/${NOTICE}/ \
  --tokenizer_name microsoft/BiomedVLP-CXR-BERT-specialized \
  --model_revision v1.1 \
  --trust_remote_code \
  --resume /jet/home/lisun/work/xinliu/hi-ml/hi-ml-multimodal/src/caches-wopretrain/bt64/cache-2023-11-27-03-22-08-moco/model_last.pth \
  # for wopretrain
  #--resume_from_checkpoint ./results/bert-base-cased/${TASK_NAME}/ \
  #--random_init_model \
