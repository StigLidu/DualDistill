python sft/train.py \
    --model_path models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --data_path dataset/train/dual_distill_data.jsonl \
    --epochs 4 \
    --code_mode \
    --batch_size 1 \
    --save_interval 2 \
    --data_seed 42 \
    --save_path agentic_R1_Qwen-7B