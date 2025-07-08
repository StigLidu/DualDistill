import json
import os
import sys

def check_format(data : str, interim_list : list[str] = ["think", "code", "executor"], end_token : str = "answer", allow_no_end_token : bool = True):
    data = data.strip()
    while not data.startswith("<" + end_token + ">"):
        next_block_exists = False
        for interim in interim_list:
            if data.startswith("<" + interim + ">"):
                next_block_exists = True
                if "</" + interim + ">" not in data:
                    return False
                try:
                    block_content = ("<" + interim + ">").join(data.split("</" + interim + ">")[0].strip().split("<" + interim + ">")[1:]).strip()
                    for other in interim_list:
                        if "<" + other + ">" in block_content:
                            return False
                    data = ("</" + interim + ">").join(data.split("</" + interim + ">")[1:]).strip()
                except:
                    return False
        if not next_block_exists:
            # if allow_no_end_token is True, check if there is any interim block after the last interim block
            if allow_no_end_token:
                for interim in interim_list:
                    if "<" + interim + ">" in data:
                        return False
                return True
            else:
                return False
    if "</" + end_token + ">" not in data:
        return False
    for interim in interim_list:
        if "<" + interim + ">" in data:
            return False
    return True

if __name__ == "__main__":
    data = "<code><code>print('Hello, world!')</code>."
    print(check_format(data))