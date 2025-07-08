model_path=$1 # http://localhost:8125/v1
model_name=$2 # DeepSeek-R1-Distill-Qwen-7B
data_path=$3 # dataset/test/DeepMath-Large.jsonl
code_mode=$4 # true
max_tokens=$5 # 4096
retrieve_path=$6 # retrieve_path

if [ "$code_mode" = "true" ]; then
    extra_args1="--code_mode"
else
    extra_args1=""
fi

# if max_tokens is not set, set it to 4096
if [ -z "$max_tokens" ]; then
    max_tokens=4096
fi

if [ "$retrieve_path" != "" ]; then
    extra_args2="--generation_save_path $retrieve_path"
else
    extra_args2=""
fi

python sft/evaluate.py \
    --use_server_inference \
    --num_samples 5 \
    --model_path $model_path \
    --model_name $model_name \
    --data_path $data_path \
    --max_tokens $max_tokens \
    $extra_args1 \
    $extra_args2