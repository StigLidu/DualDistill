import json
from datasets import load_dataset
import random

problem_path = "dataset/train/whole_training_set.jsonl"
ds = load_dataset("zwhe99/DeepMath-103K")['train']

# find a solution for each problem
with open(problem_path, "r") as f:
    all_problems = sorted([json.loads(line) for line in f], key=lambda x: x["problem"])

print("Number of problems: ", len(all_problems))
ds = list(ds)
ds.sort(key=lambda x: x["question"])

last_idx = 0
data = []
for i in range(len(all_problems)):
    while ds[last_idx]["question"] != all_problems[i]["problem"]:
        last_idx += 1
    text_reasoning = ds[last_idx][f"r1_solution_{random.randint(1, 3)}"]
    text_reasoning = text_reasoning.split("</think>")[0] + "</think>\n" + "<answer>" + text_reasoning.split("</think>")[1] + "\n</answer>"
    data.append({
        "problem": all_problems[i]["problem"],
        "solution": text_reasoning,
        "answer": all_problems[i]["answer"]
    })

with open("dataset/train/whole_training_set_with_solution.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")