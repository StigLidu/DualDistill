from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
import numpy as np
import sys
import os
import json

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from math_utils.utils import SYSTEM_PROMPT_TPL, CODE_INSTRUCTION
from math_utils.format_checking import check_format

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 0:
        start = haystack.find(needle, start + len(needle))
        n -= 1
    return start

class TrainData(Dataset):
    def __init__(self, data, tokenizer, code_instruction, max_data_count=None, data_seed=42, debug=False, max_length=16384):
        self.tokenizer = tokenizer
        self.items = []
        self.max_length = max_length
        self.total_loss_calculation_token_count = 0
        #TODO: seems like the max length of the model is 16384 because it throws an warning when the length is larger than 16384
        self.debug = debug
        system_prompt_tpl = SYSTEM_PROMPT_TPL.format(code_instruction=CODE_INSTRUCTION)

        for sample in tqdm(data, desc="Processing data"):
            question = sample["problem"]
            answer   = sample["synthetic_data"]

            if not check_format(answer):
                continue

            # only keep the <answer> … </answer> part, avoid extra content
            if "</answer>" in answer:
                answer = answer.split("</answer>")[0] + "</answer>"

            system_prompt = system_prompt_tpl
            messages = (
                system_prompt + "\n\n<｜User｜>" + question +
                "\n\n<｜Assistant｜>" + answer
            )

            input_ids = tokenizer.encode(messages)
            input_ids.append(tokenizer.eos_token_id)
            if len(input_ids) > self.max_length:
                # ignore the input_ids
                continue

            q_ids = tokenizer.encode(
                system_prompt + "\n\n<｜User｜>" + question + "\n\n<｜Assistant｜>"
            )

            labels = [-100] * len(q_ids) + input_ids[len(q_ids):]

            #TODO: For expert iteration, the number of <code> and </code> may mismatch, so we need to check the number of <code> and </code>
            errors = ["SyntaxError", "Traceback (most recent call last)", "Error: Code execution timed out."]
            code_block_count = sample["synthetic_data"].count("</code>")
            code_block_flag = [False] * code_block_count
            for c_id in range(code_block_count):
                # find the c_id-th <executor> and </executor>
                executor_start = find_nth(sample["synthetic_data"], "<executor>", c_id)
                executor_end = sample["synthetic_data"].find("</executor>", executor_start)
                if executor_start == -1 or executor_end == -1:
                    code_block_flag[c_id] = False
                    continue
                code_block_flag[c_id] = True
                for e in errors:
                    if e in sample["synthetic_data"][executor_start:executor_end]:
                        code_block_flag[c_id] = False
                        break

            # do not calculate the loss for the code blocks with errors
            decoded_str = ""
            for i in range(len(q_ids), len(input_ids)):
                decoded_str += tokenizer.decode(input_ids[i])
                if decoded_str.count("<code>") > decoded_str.count("</code>") and code_block_flag[decoded_str.count("<code>") - 1] == False:
                    labels[i] = -100

            # do not calculate the loss for the executor feedback
            decoded_str = ""
            for i in range(len(q_ids), len(input_ids)):
                decoded_str += tokenizer.decode(input_ids[i])
                if decoded_str.count("<executor>") > decoded_str.count("</executor>"):
                    labels[i] = -100

            # do not calculate the loss before the turn-over words
            turn_over_words = ["Wait, use text reasoning is too tedious, let's try code reasoning.", \
                                "<think>\nWait, the code is not correct, let's try text reasoning.", \
                                "<think>\nWait, the code may be incorrect, let's try text reasoning."
                               ]
            turn_over_flag = False
            last_occurrence = np.inf
            for word in turn_over_words:
                if word in answer:
                    # find the first occurrence of the word
                    last_o = answer.find(word)
                    if last_o < last_occurrence:
                        last_occurrence = last_o
                        turn_over_flag = True
                        turn_over_word = word
                        turn_over_num_token = len(tokenizer.encode(word))

            if turn_over_flag:
                decoded_str = ""
                for i in range(len(q_ids), len(input_ids)):
                    decoded_str += tokenizer.decode(input_ids[i])
                    labels[max(0, i - turn_over_num_token - 1)] = -100 # TODO: maybe contains 1 offset error, but it's ok.
                    if turn_over_word in decoded_str:
                        break

            # cache the tensors; change to half / int16 if memory is limited
            self.items.append((
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(labels,    dtype=torch.long),
            ))

            # loss calculation token count
            loss_calculation_token_count = 0
            for i in range(len(q_ids), len(input_ids)):
                if labels[i] != -100:
                    loss_calculation_token_count += 1
            self.total_loss_calculation_token_count += loss_calculation_token_count

            if self.debug:
                # print the str that calculate the loss
                for i in range(len(q_ids), len(input_ids)):
                    if labels[i] != -100:
                        print(tokenizer.decode(input_ids[i]), end="")
                print()

        if max_data_count is None:
            max_data_count = len(self.items)
        np.random.seed(data_seed)
        np.random.shuffle(self.items)
        self.items = self.items[:max_data_count]
        print(f"Shuffled data with seed {data_seed} and got {len(self.items)} samples")
        print(f"Total loss calculation token count: {self.total_loss_calculation_token_count}")

    def __getitem__(self, idx):
        return self.items[idx]

    def __len__(self):
        return len(self.items)

    @staticmethod
    def collate_fn(batch):
        input_ids, labels = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        labels    = pad_sequence(labels,    batch_first=True, padding_value=-100)
        return {"input_ids": input_ids, "labels": labels}

if __name__ == "__main__":
    data = [
        {"problem": "What is the sum of the first 100 natural numbers?", "synthetic_data": "The sum of the first 100 natural numbers is 5050."},
        {"problem": "What is the sum of the first 100 natural numbers?", "synthetic_data": "The sum of the first 100 natural numbers is 5050. <executor> The sum of the first 100 natural numbers is 5050. </executor>"},
        {"problem": "What is the sum of the first 100 natural numbers?", "synthetic_data": "The sum of the first 100 natural numbers is 5050. <code> test 1 </code> <executor> Traceback (most recent call last) </executor>"},
        {"problem": "What is the sum of the first 100 natural numbers?", "synthetic_data": "The sum of the first 100 natural numbers is 5050. <code> test 2 </code> <executor> SyntaxError: </executor> <think> I think the code is correct? </think>"},
        {"problem": "What is the sum of the first 100 natural numbers?", "synthetic_data": "The sum of the first 100 natural numbers is 5050. <code> test 3 </code> <executor> Error: Code execution timed out. </executor> <think> I think the code is correct. </think>"},
        {"problem": "What is the sum of the first 100 natural numbers?", "synthetic_data": "The sum of the first 100 natural numbers is 5050. <code> test 4 </code> <executor> successful </executor>"},
        {"problem": "What is the sum of the first 100 natural numbers?", "synthetic_data": "The sum of the first 100 natural numbers is 5050. <think> hahahaha </think><think>\nWait, the code is not correct, let's try text reasoning.\n</think> <think> I think the code is correct. </think>"},
        {"problem": "What is the sum of the first 100 natural numbers?", "synthetic_data": "The sum of the first 100 natural numbers is 5050. <think> hahahaha Wait, use text reasoning is too tedious, let's try code reasoning. </think> <think> hahaha here is code! </think>"},
    ]
    tokenizer = AutoTokenizer.from_pretrained("models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    train_data = TrainData(data, tokenizer, CODE_INSTRUCTION, debug=True)