import os
import time
import torch
import json
from vllm import LLM, SamplingParams
import sys
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from math_utils.utils import compute_score

from math_utils.utils import read_json_or_jsonl, code_block, find_question, find_answer
from concurrent.futures import ThreadPoolExecutor
def chunks_to_text(chunks):
    count = 0
    text = ""
    for chunk in chunks:
        text += chunk.choices[0].text
        count += 1
    return text, count

def get_token_length(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=False))

def server_inference(model_base_url: str, model_name: str, tokenizer_path: str, input: str, code_mode=False, max_tokens=4096, is_ipython=False):
    """Perform inference with vLLM.

    Args:
        model_base_url (str): The base URL of the model.
        model_name (str): The name of the model.
        tokenizer_path (str): The path to the tokenizer.
        input (str): the question
        code_mode (bool): Whether to run the special code-block logic.
        max_tokens (int): Maximum tokens to generate.
        is_ipython (bool): Whether to run the special ipython logic.
    Returns:
        str: The generated text.
    """

    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY" # TODO: remove this or set it to the correct value
    openai_api_base = model_base_url
    client = OpenAI(
        base_url=openai_api_base,
        api_key=openai_api_key,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    CODE_INSTRUCTION = """Meanwhile, you can use Python code to help you reasoning. The code should be enclosed within <code> </code> tags. For example, <code> code here </code>.
A executor will run the code and provide feedback immediately after the code. The executor feedback should be enclosed within <executor> </executor> tags.
You can use the executor feedback to improve your reasoning.
"""

    input_format = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. 
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
The final answer should be enclosed within \\boxed tags, i.e., \\boxed{{answer here}}.
{code_instruction}

<｜User｜>{problem}

<｜Assistant｜><think>
"""
    input_prompt = input_format.format(
        problem=input,
        code_instruction=CODE_INSTRUCTION if code_mode else ""
    )
    raw_input_prompt = input_prompt

    output = client.completions.create(
        model=model_name,
        prompt=input_prompt,
        max_tokens=max_tokens,
        temperature=0.6,
        extra_body={"stop": ["</code>"], "include_stop_str_in_output": True, "skip_special_tokens": False},
        stream=True,
    )
    llm_output, token_length = chunks_to_text(output)
    total_inference_length = token_length
    previous_code = ""
    # Keep going when we see '</code>' in the last chunk
    code_use_count = 0
    while "</code>" in llm_output and total_inference_length < max_tokens and "</answer>" not in llm_output and code_use_count < 10:
        code_use_count += 1
        try:
            raw_code = llm_output.split("<code>")[-1].split("</code>")[0]
            code_output, error, previous_code = code_block(raw_code, is_ipython=is_ipython, previous_code=previous_code)
        except Exception as e:
            error = str(e)
            code_output = "Error"
        if error:
            executor_feedback = f"\n<executor>\n{error}\n</executor>\n"
        else:
            executor_feedback = f"\n<executor>\n{code_output}\n</executor>\n"

        llm_output = (
            llm_output.split("</code>")[0]
            + "</code>"
            + executor_feedback
        )
        total_inference_length += len(tokenizer.encode(executor_feedback, add_special_tokens=False))
        if total_inference_length >= max_tokens:
            break

        input_prompt = input_prompt + llm_output
        output = client.completions.create(
            model=model_name,
            prompt=input_prompt,
            max_tokens=max_tokens - total_inference_length,
            temperature=0.6,
            extra_body={"stop": ["</code>"], "include_stop_str_in_output": True, "skip_special_tokens": False},
            stream=True,
        )
        llm_output, token_length = chunks_to_text(output)
        total_inference_length += token_length

    if code_use_count == 10 and total_inference_length < max_tokens:
        # excced the max number of code use
        input_prompt = input_prompt + llm_output + "\n<executor>\nThe code use count has exceeded the limit. Please stop using code.\n</executor>\n<answer>" 
        output = client.completions.create(
            model=model_name,
            prompt=input_prompt,
            max_tokens=max_tokens - total_inference_length,
            temperature=0.6,
            extra_body={"stop": ["</code>"], "include_stop_str_in_output": True, "skip_special_tokens": False},
            stream=True,
        )
        llm_output, token_length = chunks_to_text(output)
        total_inference_length += token_length

    whole_output = (input_prompt + llm_output)[len(raw_input_prompt):]

    # truncate the output to max_tokens
    whole_output_tokens = tokenizer.encode(whole_output, add_special_tokens=False)
    whole_output_tokens = whole_output_tokens[:max_tokens]
    whole_output = tokenizer.decode(whole_output_tokens, skip_special_tokens=True)
    return raw_input_prompt + whole_output

def inference(model: LLM, input: str, code_mode=False, max_tokens=4096, is_ipython=False, model_name = None, tokenizer_path = None):
    """Perform inference with vLLM.

    Args:
        model (LLM): The vLLM instance.
        input (str): the question
        code_mode (bool): Whether to run the special code-block logic.
        max_tokens (int): Maximum tokens to generate.
        is_ipython (bool): Whether to run the special ipython logic.
    Returns:
        str: The generated text.
    """

    CODE_INSTRUCTION = """Meanwhile, you can use Python code to help you reasoning. The code should be enclosed within <code> </code> tags. For example, <code> code here </code>.
A executor will run the code and provide feedback immediately after the code. The executor feedback should be enclosed within <executor> </executor> tags.
You can use the executor feedback to improve your reasoning.
"""

    input_format = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. 
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
The final answer should be enclosed within \\boxed tags, i.e., \\boxed{{answer here}}.
{code_instruction}

<｜User｜>{problem}

<｜Assistant｜><think>
"""
    input_prompt = input_format.format(
        problem=input,
        code_instruction=CODE_INSTRUCTION if code_mode else ""
    )
    input_prefix_tokens = model.get_tokenizer().encode(input_prompt)
    prefix_length = len(input_prefix_tokens)

    if code_mode:
        total_inference_length = 0
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.6,
            stop=["</code>"],
            include_stop_str_in_output=True,
            skip_special_tokens=False
        )
        output = model.generate(
            prompts=[{"prompt_token_ids": input_prefix_tokens}],
            sampling_params=sampling_params,
            use_tqdm=False
        )
        llm_output = output[0].outputs[0].text
        total_inference_length += len(output[0].outputs[0].token_ids)

        previous_code = ""
        # Keep going when we see '</code>' in the last chunk
        code_use_count = 0
        while "</code>" in llm_output and total_inference_length <= max_tokens and "</answer>" not in llm_output and code_use_count < 10:
            code_use_count += 1
            try:
                raw_code = llm_output.split("<code>")[-1].split("</code>")[0]
                code_output, error, previous_code = code_block(raw_code, is_ipython=is_ipython, previous_code=previous_code)
            except Exception as e:
                error = str(e)
                code_output = "Error"

            if error:
                executor_feedback = f"\n<executor>\n{error}\n</executor>\n"
            else:
                executor_feedback = f"\n<executor>\n{code_output}\n</executor>\n"

            llm_output = (
                llm_output.split("</code>")[0]
                + "</code>"
                + executor_feedback
            )
            total_inference_length += len(
                model.get_tokenizer().encode(executor_feedback, add_special_tokens=False)
            )
            if total_inference_length > max_tokens:
                break
            input_prefix_tokens.extend(
                model.get_tokenizer().encode(llm_output, add_special_tokens=False)
            )
            output = model.generate(
                prompts=[{"prompt_token_ids": input_prefix_tokens}],
                sampling_params=sampling_params,
                use_tqdm=False
            )
            llm_output = output[0].outputs[0].text
            total_inference_length += len(output[0].outputs[0].token_ids)

        if total_inference_length > max_tokens:
            # TODO: check if the output is truncated
            # If we reached the token limit, force close the reasoning
            input_prefix_tokens.extend(
                model.get_tokenizer().encode(llm_output, add_special_tokens=False)
            )
            input_prefix_tokens = input_prefix_tokens[: max_tokens + prefix_length]
            llm_output = ""
    else:
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.6)
        output = model.generate(
            prompts=[{"prompt_token_ids": input_prefix_tokens}],
            sampling_params=sampling_params,
            use_tqdm=False
        )
        llm_output = output[0].outputs[0].text

    whole_output = (
        model.get_tokenizer().decode(input_prefix_tokens, skip_special_tokens=True)
        + llm_output
    )
    return whole_output

def llm_eval(model_path: str,
         data_path: str = None,
         prompt: str = None,
         code_mode: bool = False,
         max_tokens: int = 4096,
         generation_save_path: str = "result",
         is_ipython: bool = False,
         use_server_inference: bool = False,
         model_name = "DeepSeek-R1-Distill-Qwen-7B_finetuned",
         tokenizer_path = "models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
         num_samples: int = 1):
    """Evaluate a model on a dataset or a single prompt.

    Args:
        model_path (str): Path to the model or model name.
        data_path (str, optional): JSON data file path for batch evaluation. Defaults to None.
        prompt (str, optional): A single prompt for inference. Defaults to None.
        code_mode (bool): Whether to parse code blocks in the response.
        max_tokens (int): Maximum tokens to generate.
        generation_save_path (str): Folder to store output text or results.
        wandb_run (wandb.Run or None): Optional wandb run to log metrics.
    """
    # call nvidia-smi to check the memory usage
    if not use_server_inference:
        display_name = model_path.strip("/").split("/")[-1]
    else:
        display_name = model_name

    if "/" not in generation_save_path:
        generation_save_path = os.path.join(
            generation_save_path,
            time.strftime("%Y%m%d_%H%M%S", time.localtime())
            + ("_" + display_name)
            + ("_" + data_path.strip("/").split("/")[-1] if data_path else "")
            + ("_" + str(code_mode))
            + ("_" + str(num_samples))
            + ("_" + str(max_tokens))
        )
        if not os.path.exists(generation_save_path):
            os.makedirs(generation_save_path)
        if use_server_inference:
            resume_id = []
        else:
            resume_id = 0
    else: 
        # resume generation
        file_list = os.listdir(generation_save_path)
        file_list = [f for f in file_list if f.endswith(".txt")]
        # find the biggest number
        ids = [int(f.split(".")[0]) for f in file_list]
        if len(ids) > 0:
            if use_server_inference:
                resume_id = ids
            else:
                resume_id = (max(ids) + 1) // num_samples
        else:
            if use_server_inference:
                resume_id = []
            else:
                resume_id = 0

    if not use_server_inference:
        assert num_samples == 1, "num_samples must be 1 when use_server_inference is False"
        model = LLM(model=model_path, tokenizer=model_path, swap_space=8, tensor_parallel_size=1, gpu_memory_utilization=0.8)
        # it is set to 1 to avoid the fork or spawn error

        # TODO: the format is for Qwen model, we need to change it to the format for other models
        assert data_path is not None or prompt is not None, "Must provide either data_path or prompt."

        if prompt is not None:
            result = inference(
                model, 
                prompt, 
                code_mode=code_mode, 
                max_tokens=max_tokens,
                is_ipython=is_ipython
            )
            print(result)

            if data_path is None:
                del model
                torch.cuda.empty_cache()

        if data_path is not None:
            data = read_json_or_jsonl(data_path)
            if os.path.exists(os.path.join(generation_save_path, "result.json")):
                with open(os.path.join(generation_save_path, "result.json"), "r") as f:
                    result = json.load(f)
                    total = result["total"]
                    correct = result["correct"]
                assert total == resume_id, f"The resume_id is not consistent with the result.json file, {total} != {resume_id}"
            else:
                total = 0
                correct = 0
            pbar = tqdm(data[resume_id:], desc="Evaluating", total=len(data) - resume_id)
            for item in pbar:
                output = inference(
                    model,
                    find_question(item),
                    code_mode=code_mode,
                    max_tokens=max_tokens,
                    is_ipython=is_ipython
                )
                answer = item['answer'] if 'answer' in item else item['reward_model']['ground_truth'] if 'reward_model' in item else item["final_answer"]
                try:
                    score = compute_score(output.split("<｜Assistant｜>")[-1], answer)
                    if score == 1:
                        correct += 1
                except Exception as e:
                    correct += 0
                total += 1
                accuracy = correct / total
                pbar.set_postfix(correct=correct, accuracy=f"{accuracy:.2%}")

                text_validation = (
                    "\n########################################################\n"
                    f"correct answer: {answer}\n"
                    f"score: {score}\n"
                    "########################################################"
                )
                with open(os.path.join(generation_save_path, f"{total - 1}.txt"), "w", encoding="utf-8") as f:
                    f.write(output + text_validation)
                with open(os.path.join(generation_save_path, "result.json"), "w") as f:
                    f.write(json.dumps({"total": total, "correct": correct, "accuracy": accuracy}))

            del model
            torch.cuda.empty_cache()
            return correct / total
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        def run_one(item_idx: int, sample_idx: int, item):
            """
            One call of `server_inference` ⇒ returns (item_idx, sample_idx, output, score)
            """
            answer  = find_answer(item)
            output  = server_inference(model_base_url=model_path,
                                    model_name=model_name,
                                    tokenizer_path=tokenizer_path,
                                    input=find_question(item),
                                    code_mode=code_mode,
                                    max_tokens=max_tokens,
                                    is_ipython=is_ipython)
            return item_idx, sample_idx, output, answer

        data = read_json_or_jsonl(data_path)
        n_calls = [i for i in range(len(data) * num_samples) if i not in resume_id]

        tot_lock   = threading.Lock()
        total      = 0
        correct    = 0

        pbar = tqdm(total=len(n_calls), desc="Evaluating")

        with ThreadPoolExecutor(max_workers=num_samples) as pool:
            futures = []
            for i in n_calls:
                futures.append(pool.submit(run_one, i // num_samples, i % num_samples, data[i // num_samples]))

            for fut in as_completed(futures):
                item_idx, sample_idx, output, answer = fut.result()
                score = compute_score(output.split("<｜Assistant｜>")[-1], answer)

                # --- global bookkeeping (thread-safe) ---
                with tot_lock:
                    total   += 1
                    correct += (score == 1)

                # --- write per-sample result ---
                fname = os.path.join(generation_save_path,
                                    f"{item_idx*num_samples + sample_idx}.txt")
                text_validation = (
                    "\n########################################################\n"
                    f"correct answer: {answer}\n"
                    f"score: {score}\n"
                    "########################################################"
                )
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(output + text_validation)

                # --- write running summary (overwrite) ---
                with open(os.path.join(generation_save_path, "result.json"), "w") as f:
                    json.dump({"total": total,
                            "correct": correct,
                            "accuracy": correct / total},
                            f, ensure_ascii=False, indent=2)

                pbar.set_postfix(correct=correct, accuracy=f"{correct/total:.2%}")
                pbar.update()

        pbar.close()
        return correct / total

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
    parser.add_argument("--model_name", type=str, default="DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--data_path", type=str, default="synthetic_data_7b_all_2.jsonl")
    parser.add_argument("--code_mode", action='store_true')
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--generation_save_path", type=str, default="result")
    parser.add_argument("--is_ipython", action='store_true')
    parser.add_argument("--use_server_inference", action='store_true')
    parser.add_argument("--num_samples", type=int, default=1)
    args = parser.parse_args()
    print(args)
    llm_eval(
        model_path=args.model_path,
        model_name=args.model_name,
        data_path=args.data_path,
        code_mode=args.code_mode,
        max_tokens=args.max_tokens,
        generation_save_path=args.generation_save_path,
        is_ipython=args.is_ipython,
        use_server_inference=args.use_server_inference,
        num_samples=args.num_samples
    )