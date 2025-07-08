model_path=$1 # models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
display_name=$2 # DeepSeek-R1-Distill-Qwen-7B
port=$3

# default port is 8123
if [ -z "$port" ]; then
    port=8123
fi

vllm serve $model_path \
    -pp 2 \
    -tp 2 \
    --gpu-memory-utilization 0.8 \
    --host 0.0.0.0 --port $port \
    --served-model-name $display_name \
    --enable-prefix-caching