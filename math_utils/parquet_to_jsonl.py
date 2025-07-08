# turn parquet file to json file

import pandas as pd
import sys

file_path = sys.argv[1]

df = pd.read_parquet(file_path)

df.to_json(file_path.replace(".parquet", ".jsonl"), orient='records', lines=True)