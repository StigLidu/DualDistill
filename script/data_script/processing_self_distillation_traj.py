import json
import os
import sys

# add the root directory to the python path
sys.path.append(os.path.abspath("."))

from math_utils.format_checking import check_format
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
root_dir = "dataset/train/self_distillation"
data_path = os.path.join(root_dir, "iteration_1_correct_replay_buffer.jsonl")
wrong_data_path = os.path.join(root_dir, "iteration_1_incorrect_replay_buffer.jsonl")
correct_record_path = os.path.join(root_dir, "iteration_1_accuracy.jsonl")
text_reasoning_path = "dataset/train/whole_training_set_with_solution.jsonl"

num_samples = 16
correctness_bar = 0.9

with open(text_reasoning_path, "r") as f:
    text_reasoning_data = [json.loads(line) for line in f]

data = []
count = 0
with open(data_path, "r") as f:
    raw_data = [json.loads(line) for line in f]
    count = 0
    temp_add_data = None
    for i in range(len(raw_data)):
        # If not all the data are correct, we add an example into expert iteration
        # Do not add a text reasoning path because this part has been added in SFT stage
        # One idx may have multiple data, we only add one data piece into expert iteration to ensure diversity
        if i == 0 or raw_data[i]["idx"] != raw_data[i-1]["idx"]:
            if count < num_samples and temp_add_data is not None:
                # > 0, < correctness_bar
                data.append(temp_add_data)
                temp_add_data = None
            count = 0
        count += 1
        if check_format(raw_data[i]["synthetic_data"]):
            attempt = raw_data[i]["synthetic_data"].split("<answer>")[0].strip()
            if temp_add_data is None:
                for j in range(len(text_reasoning_data)):
                    if text_reasoning_data[j]["problem"] == raw_data[i]["problem"]:
                        temp_add_data = attempt + "\n<think>\nWait, we can also use text-reasoning as an alternative way to verify the solution.\n\n" + text_reasoning_data[j]["solution"]
                        raw_data[i]["synthetic_data"] = temp_add_data
                        temp_add_data = raw_data[i]
                        count += 1
                        break

print("self-distillation part 1: ", len(data))

wrong_data = []

# Wrong data but format correct
# Replace the last block with correct text reasoning
# At most one wrong data per idx
# If exist a correct sample, we do not add a corrected wrong sample into expert iteration

with open(wrong_data_path, "r") as f:
    raw_data = [json.loads(line) for line in f]
    with open(correct_record_path, "r") as ff:
        correct_record = [json.loads(line) for line in ff]
    max_attempt_len = 0
    for i in range(len(raw_data)):
        if i == 0 or raw_data[i]["idx"] != raw_data[i - 1]["idx"]:
            count = 0
        if correct_record[raw_data[i]["idx"] - 1]["accuracy"] >= correctness_bar:
            continue
        attempt = "</think>".join(raw_data[i]["synthetic_data"].split("</think>")[:-1]) + "</think>\n"
        if "Wait, the code is not correct, let's try text reasoning" in attempt:
            continue
        if check_format(attempt) and count < 1:
            max_attempt_len = max(max_attempt_len, len(tokenizer.encode(attempt)))
            # find corresponding correct text-reasoning
            for j in range(len(text_reasoning_data)):
                if text_reasoning_data[j]["problem"] == raw_data[i]["problem"]:
                    new_attempt = attempt + "\n<think>\nWait, the code is not correct, let's try text reasoning.\n\n" + text_reasoning_data[j]["solution"]
                    if check_format(new_attempt):
                        new_data = raw_data[i]
                        new_data["synthetic_data"] = new_attempt
                        wrong_data.append(new_data)
                        count += 1
                        break

print("self-distillation part 2: ", len(wrong_data))
data_root_dir = "dataset/train/self_distillation"

with open(os.path.join(data_root_dir, f"iteration_1_correct_replay_buffer_deduplicated.jsonl"), "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")

with open(os.path.join(data_root_dir, f"iteration_1_incorrect_replay_buffer_revised_deduplicated_{correctness_bar}.jsonl"), "w") as f:
    for item in wrong_data:
        f.write(json.dumps(item) + "\n")