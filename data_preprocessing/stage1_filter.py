import os
from math_util import *

rm_stat = {}


def tell_choice(d, extracted_model_output):
    choices = ["A", "B", "C"]
    if all([f"{choice}." in d["problem"] for choice in choices]):
        return True

    if all([f"{choice}:" in d["problem"] for choice in choices]):
        return True

    if all([f"({choice})" in d["problem"] for choice in choices]):
        return True

    return False


def add_rm_stat(subset, type, d):
    if type not in rm_stat[subset]:
        rm_stat[subset][type] = {"cnt": 0, "sampled_data": []}

    rm_stat[subset][type]["cnt"] += 1
    if len(rm_stat[subset][type]["sampled_data"]) < 10:
        rm_stat[subset][type]["sampled_data"].append(d)


def process_fn(d, idx, check_type=False):
    rm = False
    q_type = "qa"

    subset = d["source"]
    problem = d["problem"].lower()
    is_matched, extracted_model_output = match_answer(d["solution"])

    if check_type:
        if "figure" in problem:
            rm = True
            add_rm_stat(subset, "figure", d)
        elif "prove" in problem or "show that" in problem:
            rm = True
            add_rm_stat(subset, "prove", d)
        elif tell_choice(d, extracted_model_output):
            rm = False
            q_type = "choice"
            add_rm_stat(subset, "choice", d)
        elif "\\underline{" in problem:
            rm = True
            add_rm_stat(subset, "fill-in-blank", d)
        elif not is_matched:
            rm = True
            add_rm_stat(subset, "ans_not_matched", d)
    else:
        if not is_matched:
            rm = True
            add_rm_stat(subset, "ans_not_matched", d)
    nd = {
        "id": str(idx),
        "source": d["source"],
        "prompt": d["problem"],
        "solution": d["solution"],
        "reference": extracted_model_output,
        "remove": rm,
        "type": q_type,
    }
    return nd


import re


def clean_text(data):
    match = re.search(r"\|end_header_id\|>\n\n(.*?)<\|eot_id\|>", data, re.DOTALL)
    if not match:
        return data
    return match.group(1)


import json

from datasets import load_from_disk
from IPython import embed
from tqdm import tqdm

date = "cleaned"
data_path = os.path.expanduser(
    "~/workspace/github/lm-evaluation-harness/decontaminate_latent/decontaminated2.jsonl"
)  # "path/to/NuminaMath-CoT"  # TODO
subsets = ["synthetic_math"]
for subset in subsets:
    rm_stat[subset] = {}

# dataset = load_from_disk(data_path)["train"]
dataset = []
with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        data["source"] = "synthetic_math"
        # change src to problem, change tgt to solution
        data["problem"] = clean_text(data["src"])
        data["solution"] = data["tgt"]
        del data["src"]
        del data["tgt"]
        dataset.append(data)
print(dataset[0])
# dataset = dataset.shuffle(seed=42)
# dataset = dataset.select(range(100))

new_dataset = []
for idx in tqdm(range(len(dataset))):
    data = dataset[idx]
    if data["source"] in subsets:
        new_dataset.append(process_fn(data, idx))

os.makedirs(date, exist_ok=True)

with open(f"{date}/stage1_stat.json", "w", encoding="utf-8") as f:
    json.dump(rm_stat, f, ensure_ascii=False, indent=4)

with open(f"{date}/stage1_filtered.jsonl", "w", encoding="utf-8") as f:
    for data in new_dataset:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
