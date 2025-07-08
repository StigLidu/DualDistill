import json
import os
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from math_utils.utils import read_json_or_jsonl, find_question, find_answer
from sft.evaluate import server_inference
from math_utils.utils import compute_score
from tqdm import tqdm

def self_distillation_sampler(server_url: str, model_name: str, model_path: str, data_path: str, num_samples: int, save_path: str, data_size: int, iteration: int=1):
    """
    Self-distillation sampler
    The model generates #num_samples samples for each question in the data_path
    """
    # check paths
    os.makedirs(save_path, exist_ok=True)
    # load the dataset
    data = read_json_or_jsonl(data_path)
    data = data[:data_size]
    # generate #num_samples samples for each question in the data_path
    correct_replay_buffer_path = os.path.join(save_path, f"iteration_{iteration}_correct_replay_buffer.jsonl")
    incorrect_replay_buffer_path = os.path.join(save_path, f"iteration_{iteration}_incorrect_replay_buffer.jsonl")
    if os.path.exists(correct_replay_buffer_path.replace(".jsonl", "_last_sample_id.txt")):
        with open(correct_replay_buffer_path.replace(".jsonl", "_last_sample_id.txt"), "r") as f:
            last_sample_id = int(f.read())
    else:
        last_sample_id = 0
    if os.path.exists(correct_replay_buffer_path.replace("correct_replay_buffer.jsonl", "accuracy.jsonl")):
        with open(correct_replay_buffer_path.replace("correct_replay_buffer.jsonl", "accuracy.jsonl"), "r") as f:
            accuracy_data = [json.loads(line) for line in f]
            sum_accuracy = sum([data["accuracy"] for data in accuracy_data])
    else:
        accuracy_data = []
        sum_accuracy = 0
    with tqdm(data[last_sample_id:], total=len(data) - last_sample_id, desc="Generating samples") as pbar:
        for i, item in enumerate(pbar, 1):
            # generate #num_samples samples
            correct = 0
            with ThreadPoolExecutor(max_workers=num_samples) as executor:
                futures = [executor.submit(server_inference, \
                                        model_base_url=server_url, \
                                        model_name=model_name, \
                                        tokenizer_path=model_path, \
                                        input=find_question(item), \
                                        code_mode=True, \
                                        max_tokens=4096, \
                                        is_ipython=False) for i in range(num_samples)]
                outputs = [future.result() for future in futures]

            correct = 0
            correct_data_to_save = []
            incorrect_data_to_save = []
            for output in outputs:
                # check if the sample is correct
                if compute_score(output.split("<｜Assistant｜>")[-1], find_answer(item)) == 1:
                    # add the sample to the data
                    correct += 1
                    correct_data_to_save.append(output)
                else:
                    incorrect_data_to_save.append(output)

            # save the samples
            for data in correct_data_to_save:
                with open(correct_replay_buffer_path, "a") as f:
                    f.write(json.dumps({"idx": i + last_sample_id, "problem": find_question(item), "synthetic_data": data.split("<｜Assistant｜>")[-1]}) + "\n")
            for data in incorrect_data_to_save:
                with open(incorrect_replay_buffer_path, "a") as f:
                    f.write(json.dumps({"idx": i + last_sample_id, "problem": find_question(item), "synthetic_data": data.split("<｜Assistant｜>")[-1]}) + "\n")

            sum_accuracy += correct / num_samples
            accuracy_data.append({"idx": i + last_sample_id, "accuracy": correct / num_samples, "correct": correct, "total": num_samples})
            pbar.set_postfix({"accuracy": f"{sum_accuracy / (i + last_sample_id):.2%}", "correct": correct, "total": num_samples})
            with open(correct_replay_buffer_path.replace(".jsonl", "_last_sample_id.txt"), "w") as f:
                f.write(str(i + last_sample_id))
            with open(correct_replay_buffer_path.replace("correct_replay_buffer.jsonl", "accuracy.jsonl"), "w") as f:
                for data in accuracy_data:
                    f.write(json.dumps(data) + "\n")

    print(f"Iteration {iteration}, average accuracy: {sum_accuracy / len(data)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", type=str, default="http://localhost:8123/v1", help="The server url")
    parser.add_argument("--model_name", type=str, default="DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--model_path", type=str, default="models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--data_path", type=str, default="dataset/train/whole_training_set.jsonl")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--save_path", type=str, help="The path to save the self-distillation trajectories", default="dataset/train/self_distillation")
    parser.add_argument("--data_size", type=int, default=-1, help="The number of data to sample, -1 means all data")
    args = parser.parse_args()
    print(args)
    self_distillation_sampler(server_url=args.server_url, \
                     model_name=args.model_name, \
                     model_path=args.model_path, \
                     data_path=args.data_path, \
                     num_samples=args.num_samples, \
                     save_path=args.save_path, \
                     data_size=args.data_size)