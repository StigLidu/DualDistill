import os
import json
import sys
import gc, torch
from torch.utils.data import Dataset
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from typing import List, Union
import numpy as np
import random

# Optional: Weights & Biases will only be used if a project name is passed.
try:
    import wandb
except ImportError:
    wandb = None

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from sft.dataloader import TrainData
from math_utils.utils import read_json_or_jsonl
CODE_INSTRUCTION = """Meanwhile, you can use Python code to help you reasoning. The code should be enclosed within <code> </code> tags. For example, <code> code here </code>.
A executor will run the code and provide feedback immediately after the code. The executor feedback should be enclosed within <executor> </executor> tags.
You can use the executor feedback to improve your reasoning.
"""

def train(model_path: str, 
          data_path: Union[List[str], List[dict]], 
          epochs: int, 
          save_path: str, 
          wandb_run=None, 
          resume=False, 
          resume_path=None, 
          save_interval=1, 
          batch_size=1,
          code_mode=False, 
          max_data_count=None, 
          data_seed=42, 
          max_length=16384, 
          lr=1e-5, 
          gradient_accumulation_steps=4):
    """Train the model with the given parameters.

    Args:
        model_path (str): Path to the model checkpoint or identifier.
        data_path (List[str] or List[dict]): Path to the JSONL training data or list of training data.
        epochs (int): Number of training epochs.
        save_path (str): Where to save the fine-tuned model.
        wandb_run (wandb.Run or None): An optional wandb run instance.
    """
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    model_path = model_path[:-1] if model_path.endswith('/') else model_path
    if resume:
        load_path = resume_path
        start_epoch = int(resume_path.split('_')[-1]) + 1
    else:
        load_path = model_path
        start_epoch = 0
    tokenizer = AutoTokenizer.from_pretrained(load_path)
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    data = []
    for p in data_path:
        if isinstance(p, dict):
            data.append(p)
        else:
            data.extend(read_json_or_jsonl(p))

    if code_mode:
        dataset = TrainData(data, tokenizer, CODE_INSTRUCTION, max_data_count=max_data_count, data_seed=data_seed, max_length=max_length)
    else:
        dataset = TrainData(data, tokenizer, "", max_data_count=max_data_count, data_seed=data_seed, max_length=max_length)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=dataset.collate_fn, shuffle=True,
        batch_size=batch_size, num_workers=1
    )
    optimizer = AdamW(model.parameters(), lr=lr)
    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

    global_step = start_epoch * len(data_loader)
    for epoch in range(start_epoch, epochs):
        accelerator.print(f'Training epoch {epoch}')
        accelerator.wait_for_everyone()
        model.train()

        tk0 = tqdm(data_loader, total=len(data_loader), disable=not accelerator.is_main_process)
        loss_report = []

        loss_report = []
        grad_acc = accelerator.gradient_accumulation_steps

        for step, batch in enumerate(tk0):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                # --- each micro-step gather the loss for statistics ---
                loss_val = accelerator.gather(loss.detach()).mean().item()
                loss_report.append(loss_val)

                if accelerator.sync_gradients: 
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    # --- WandB loss ---
                    if wandb_run is not None:
                        window = loss_report[-grad_acc:]
                        wandb_run.log(
                            {"train_loss": sum(window) / len(window),
                            "epoch": epoch},
                            step=global_step
                        )
                    global_step += 1

            # --- average loss ---
            tk0.set_postfix(loss=sum(loss_report[-100:]) / len(loss_report[-100:]))

        if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                f'{save_path}_{epoch}',
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )
            tokenizer.save_pretrained(f'{save_path}_{epoch}')
    # Clean up
    del model, optimizer, data_loader, dataset, tokenizer

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
    parser.add_argument("--data_path", type=str, nargs='+', default=["data/dataset/train/dual_distill_data.jsonl"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--code_mode", action='store_true')
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_data_count", type=int, default=None)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=16384)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of batches to accumulate before each optimizer.step()"
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    # Added W&B arguments
    parser.add_argument("--use_wandb", action='store_true', default=False)
    parser.add_argument("--wandb_project", type=str, default="dualdistill", 
                        help="If set, will enable wandb logging to the given project.")
    parser.add_argument("--wandb_run_name", type=str, default=None, 
                        help="An optional run name for wandb.")

    args = parser.parse_args()
    print(args)

    # fix all seeds
    torch.manual_seed(args.data_seed)
    torch.cuda.manual_seed(args.data_seed)
    torch.cuda.manual_seed_all(args.data_seed)
    np.random.seed(args.data_seed)
    random.seed(args.data_seed)

    assert args.epochs % args.save_interval == 0, "epochs must be divisible by save_interval"
    if args.model_path is None and not args.eval_only:
        raise ValueError("model_path is required for training")
    time_tag = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if args.resume:
        if args.resume_path is None:
            raise ValueError("resume_path is required for resume")
        time_tag = "_".join(args.resume_path.split("_")[-3:-1])
        print(f"Resuming from {args.resume_path} with time tag {time_tag}")

    if args.save_path is None:
        if len(args.data_path) == 1:
            args.save_path = args.model_path.strip("/") + "_" + args.data_path[0].strip("/").split("/")[-1].split(".")[0] + "_fine-tuned" + "_" + time_tag
        else:
            args.save_path = args.model_path.strip("/") + "_" + args.data_path[0].strip("/").split("/")[-2].split(".")[0] + "_mixed_data_" + "fine-tuned" + "_" + time_tag
    else:
        args.save_path = args.model_path.strip("/") + "_" + args.save_path.strip("/") + "_" + time_tag

    if args.wandb_run_name is None:
        args.wandb_run_name = (args.save_path.strip("/").split("/")[-1] +
                               "_" + str(args.code_mode) +
                               "_" + str(args.epochs) +
                               "_" + time_tag)

    # Initialize wandb if user has set a project
    wandb_run = None
    if args.use_wandb and wandb is not None:
        config = vars(args)
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=config)

    train(
        model_path=args.model_path,
        data_path=args.data_path,
        epochs=args.epochs,
        save_path=args.save_path,
        wandb_run=wandb_run,
        resume=args.resume,
        resume_path=args.resume_path,
        save_interval=args.save_interval,
        batch_size=args.batch_size,
        code_mode=args.code_mode,
        max_data_count=args.max_data_count,
        data_seed=args.data_seed,
        max_length=args.max_length,
        lr=args.lr,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    if wandb_run is not None:
        wandb_run.finish()

if __name__ == '__main__':
    main()