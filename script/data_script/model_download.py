# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from requests.exceptions import HTTPError
import sys
from pathlib import Path
from typing import Optional


def hf_download(
    repo_id: Optional[str] = None,
    hf_token: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> None:
    from huggingface_hub import snapshot_download

    local_dir = local_dir or "checkpoints"

    os.makedirs(f"{local_dir}/{repo_id}", exist_ok=True)
    try:
        snapshot_download(
            repo_id,
            local_dir=f"{local_dir}/{repo_id}",
            local_dir_use_symlinks=False,
            token=hf_token,
        )
    except HTTPError as e:
        if e.response.status_code == 401:
            print(
                "You need to pass a valid `--hf_token=...` to download private checkpoints."
            )
        else:
            raise e


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download data from HuggingFace Hub.")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="checkpoints/meta-llama/llama-2-7b-chat-hf",
        help="Repository ID to download from.",
    )
    parser.add_argument(
        "--local_dir", type=str, default=None, help="Local directory to download to."
    )
    parser.add_argument(
        "--hf_token", type=str, default=None, help="HuggingFace API token."
    )

    args = parser.parse_args()
    hf_download(args.repo_id, args.hf_token, args.local_dir)