model_path=$1
save_path=$2

python sft/train.py \
    --model_path $model_path \
    --data_path dataset/train/self_distillation/iteration_1_correct_replay_buffer_deduplicated.jsonl \
    dataset/train/self_distillation/iteration_1_incorrect_replay_buffer_revised_deduplicated_0.9.jsonl \
    --save_path $save_path \
    --epochs 4 \
    --code_mode \
    --batch_size 1 \
    --save_interval 2 \
    --data_seed 42 \
    --max_length 8192 \
    --lr 1e-5 \
    --resume \
    --resume_path models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B_e0_latest_20250510_222416_3_agentic_r1_sd_20250707_234942_3
    # make sure lr * total_steps (max_data_count * epochs) a constant
    # actually, we should control lr * total_tokens a constant, while it is hard to control total_tokens
    # Be careful, LLM will suffer from overfitting if the total_tokens is too large