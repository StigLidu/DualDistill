import tempfile
import subprocess
import json
import io
import contextlib
import ast
import os
import multiprocessing
from math_verify import parse, verify

def find_question(data: dict):
    """
    Find the question in the data.
    """
    if "problem" in data:
        return data["problem"]
    elif "question" in data:
        return data["question"]
    elif "prompt" in data:
        return data["prompt"][-1]["content"]
    else:
        raise ValueError("No question found in the data, all keys: " + str(data.keys()))

def find_answer(data: dict):
    """
    Find the answer in the data.
    """
    if "answer" in data:
        return data["answer"]
    elif "reward_model" in data:
        return data["reward_model"]["ground_truth"]
    elif "ground_truth" in data:
        return data["ground_truth"]
    elif "final_answer" in data:
        return data["final_answer"]
    else:
        raise ValueError("No answer found in the data, all keys: " + str(data.keys()))

def code_block(raw_code: str, is_ipython=False, previous_code: str = ""):
    raw_code = raw_code.replace("```python", "").replace("```", "")
    # if the last line of the code is not "print(a)" but "a", change it to "print(a)"
    raw_code = raw_code.strip(" \n")
    # change from ipython code to python code

    if is_ipython:
        raw_code_last_line = raw_code.split("\n")[-1]
        if "print(" not in raw_code_last_line and not raw_code_last_line.strip().startswith("#"):
            # if the last line is not a print statement, add a print statement to the last line
            raw_code_before_last_line = "\n".join(raw_code.split("\n")[:-1])
            raw_code = raw_code_before_last_line + "\nprint(" + raw_code_last_line + ")"
    
    raw_code = previous_code + "\nprint('<<NEW CODE START>>')\n" + raw_code

    # incremental code execution
    os.makedirs("workspace", exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir="workspace", suffix=".py", delete=False, encoding='utf-8') as tmp_file:
        tmp_file.write(raw_code)
        tmp_filename = tmp_file.name
    try:
        result = subprocess.run(
            ["python", tmp_filename],
            capture_output=True,
            text=True,
            timeout=3
        )
        os.remove(tmp_filename)
        output = result.stdout.split("<<NEW CODE START>>\n")[-1]
        if result.stderr:
            return output, result.stderr, previous_code
        else:
            return output, result.stderr, previous_code + "\n" + raw_code
    except subprocess.TimeoutExpired:
        os.remove(tmp_filename)
        return "", "Error: Code execution timed out.", previous_code

def code_block_with_io(raw_code: str):
    code = raw_code.replace("```python", "").replace("```", "")
    
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    env = {}

    try:
        tree = ast.parse(code, mode='exec')
        
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            for node in tree.body:
                if isinstance(node, ast.Expr):
                    expr_code = compile(ast.Expression(node.value), filename="<ast>", mode="eval")
                    result = eval(expr_code, env)
                    if result is not None:
                        print(result)
                else:
                    stmt_code = compile(ast.Module([node], []), filename="<ast>", mode="exec")
                    exec(stmt_code, env)
    except Exception as e:
        stderr_capture.write(str(e))
    
    return stdout_capture.getvalue(), stderr_capture.getvalue()

def parse_boxed(text, reverse=False):
    """
    Returns a list of all the contents inside \\boxed{...} in `text`,
    handling nested braces to a reasonable extent.
    """
    results = []
    search_start = 0
    marker = r'\boxed{'
    
    while True:
        # Look for the next occurrence of \boxed{
        start_index = text.find(marker, search_start)
        if start_index == -1:
            # No more \boxed{ found
            break
        
        # The position right after '\boxed{'
        brace_start = start_index + len(marker)
        
        # Use a stack to find the matching '}'
        brace_count = 1
        pos = brace_start
        
        while pos < len(text) and brace_count > 0:
            if text[pos] == '{':
                brace_count += 1
            elif text[pos] == '}':
                brace_count -= 1
            pos += 1
        
        # If brace_count == 0, 'pos-1' is where the matching '}' was found
        if brace_count == 0:
            content = text[brace_start : pos - 1]
            results.append(content)
            # Continue searching after this boxed content
            search_start = pos
        else:
            # We reached the end of the text without finding a matching brace
            break
    if len(results) == 0:
        return "No Answer"
    if not reverse:
        return results[0]
    else:
        return results[-1]

def read_json_or_jsonl(file_path):
    with open(file_path, 'r') as f:
        if file_path.endswith('.json'):
            return json.load(f)
        elif file_path.endswith('.jsonl'):
            return [json.loads(line) for line in f]

SYSTEM_PROMPT_TPL = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it.\n"
    "The assistant first thinks about the reasoning process in the mind and then "
    "provides the user with the answer. \n"
    "The reasoning process and answer are enclosed within <think> </think> and "
    "<answer> </answer> tags, respectively, i.e., <think> reasoning process here "
    "</think> <answer> answer here </answer>.\n\n"
    "The final answer should be enclosed within \\boxed tags, i.e., "
    "\\boxed{{answer here}}.\n\n"
    "{code_instruction}\n\n"
    "Do not write text outside the tags."
)

CODE_INSTRUCTION = """Meanwhile, you can use Python code to help you reasoning. The code should be enclosed within <code> </code> tags. For example, <code> code here </code>.
A executor will run the code and provide feedback immediately after the code. The executor feedback should be enclosed within <executor> </executor> tags.
You can use the executor feedback to improve your reasoning.
"""

def add_comma_into_number(number_str):
    try:
        number = float(number_str)
        if number.is_integer():
            return "{:,}".format(int(number))
        else:
            return "{:,}".format(number)
    except (ValueError, TypeError):
        return number_str

def compute_score(solution_str, ground_truth, able_to_use_original_solution=False) -> float:
    ground_truth = str(ground_truth)

    # for non-finetuned model (i.e., except Agentic-R1), use additional pass by the original solution
    original_solution_str = solution_str

    # For the agentic trajectory, remove the redundant part
    if "</code>" in solution_str:
        interim_strs = ["</think>", "</code>", "</answer>", "</executor>"]
        found = [(solution_str.rfind(s), i) for i, s in enumerate(interim_strs) if solution_str.rfind(s) != -1]
        if found:
            last_interim_str, last_interim_str_idx = max(found)
            solution_str = solution_str[:last_interim_str + len(interim_strs[last_interim_str_idx])]

    if "<answer>" in solution_str and "</answer>" in solution_str:
        # for the case that the model output a final answer within <answer>...</answer>
        solution_str_parsed = solution_str.split("<answer>")[1].split("</answer>")[0]
        if ground_truth in solution_str_parsed or add_comma_into_number(ground_truth) in solution_str_parsed:
            return 1.

    elif "<executor>" in solution_str and "</code>" in solution_str:
        # otherwise, try to parse the last output from the executor
        solution_str_parsed = solution_str.split("<executor>")[1].split("</executor>")[0]
        if ground_truth in solution_str_parsed or add_comma_into_number(ground_truth) in solution_str_parsed:
            return 1.

    elif "</think>" in solution_str and "<executor>" not in solution_str:
        # for the case that the model output is <think>...</think> and answer after think, i.e., the text-reasoning model
        solution_str_parsed = solution_str.split("</think>")[-1]
        if ground_truth in solution_str_parsed or add_comma_into_number(ground_truth) in solution_str_parsed:
            return 1.

    # fuzzy match the ground truth and the solution
    retval = 0.
    if '\\boxed' not in ground_truth: ground_truth = '\\boxed{' + str(ground_truth) + '}'
    correct = verify(parse(ground_truth), parse(solution_str)) or (able_to_use_original_solution and verify(parse(ground_truth), parse(original_solution_str)))
    if correct:
        retval = 1.

    return retval

# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string

if __name__ == "__main__":
    code_output, error, previous_code = code_block("a=5\nprint(a*2)", is_ipython=False, previous_code="")
    print(code_output, error)
    code_output, error, previous_code = code_block("b=a*2\nprint(b + 1)", is_ipython=False, previous_code=previous_code)
    print(code_output, error)
    code_output, error, previous_code = code_block("import time\ntime.sleep(5)\nprint('ok')", is_ipython=False, previous_code=previous_code)
    print(code_output, error)
    code_output, error, previous_code = code_block("print(b * 2)", is_ipython=False, previous_code=previous_code)
    print(code_output, error)
    code_output, error, previous_code = code_block("print(c * 2)", is_ipython=False, previous_code=previous_code)
    print(code_output, error)
    code_output, error, previous_code = code_block("print(b * 2)", is_ipython=False, previous_code=previous_code)
    print(code_output, error)