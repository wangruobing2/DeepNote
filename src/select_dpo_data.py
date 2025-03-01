import argparse
import json
import re
import random

parser = argparse.ArgumentParser()
parser.add_argument(
    "--init_num", type=int, default=1900, help="Number of init_note samples"
)
parser.add_argument(
    "--refine_num", type=int, default=1900, help="Number of refine_note samples"
)
parser.add_argument(
    "--query_num", type=int, default=1900, help="Number of gen_query samples"
)
parser.add_argument("--rag_num", type=int, default=300, help="Number of RAG samples")
parser.add_argument(
    "--data_path", type=str, required=True, help="Path to input data file"
)
parser.add_argument(
    "--output_path",
    type=str,
    default="../data/dpo/processed/train.jsonl",
    help="Path to output file",
)
args = parser.parse_args()

with open(args.data_path, "r", encoding="utf-8") as file:
    data = [json.loads(i) for i in file]
random.shuffle(data)


def extract_and_match(input):
    match = re.search(r"-(\d+)", input)
    return match.group(1)


# Collect samples for each category
init_data, refine_data, query_data, rag_data = [], [], [], []
for d in data:
    if not (d["chosen"] and d["rejected"]):
        continue

    fields = {"prompt": d["prompt"], "chosen": d["chosen"], "rejected": d["rejected"]}

    if "init" in d["id"] and len(init_data) < args.init_num:
        init_data.append(fields)
    elif "refine" in d["id"] and len(refine_data) < args.refine_num:
        refine_data.append(fields)
    elif "query" in d["id"] and len(query_data) < args.query_num:
        query_data.append(fields)
    elif "rag" in d["id"] and len(rag_data) < args.rag_num:
        rag_data.append(fields)

# Combine and save processed data
train_data = init_data + refine_data + query_data + rag_data
random.shuffle(train_data)

with open(args.output_path, "w") as fout:
    for res in train_data:
        fout.write(json.dumps(res, ensure_ascii=False) + "\n")
print(
    f"Init: {len(init_data)}, Refine: {len(refine_data)}, Query: {len(query_data)}, RAG: {len(rag_data)}"
)
print(f"Total training samples: {len(train_data)}")
