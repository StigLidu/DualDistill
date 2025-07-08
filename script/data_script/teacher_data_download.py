from datasets import load_dataset
import json
# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("VanishD/DualDistill")

# Save the dataset to a local directory in jsonl format
save_path = f"dataset/train/dual_distill_data.jsonl"
with open(save_path, "w") as f:
    for item in ds["train"]:
        f.write(json.dumps(item) + "\n")