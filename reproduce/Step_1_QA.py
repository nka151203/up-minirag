import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv

from tqdm import trange

from minirag import QueryParam
from reproduce.config import add_llm_args, EndpointConfig


def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG QA")
    add_llm_args(parser)
    parser.add_argument("--workingdir", type=str, default="./LiHua-World")
    parser.add_argument("--datapath", type=str, default="./dataset/LiHua-World/data/")
    parser.add_argument(
        "--querypath", type=str, default="./dataset/LiHua-World/qa/query_set.csv"
    )
    parser.add_argument("--outputpath", type=str, default="./logs/Default_output.csv")
    return parser.parse_args()


args = get_args()

WORKING_DIR = args.workingdir
QUERY_PATH = args.querypath
OUTPUT_PATH = args.outputpath

print("USING LLM:", args.model)
print("USING BASE URL:", args.base_url)
print("USING WORKING DIR:", WORKING_DIR)

os.makedirs(WORKING_DIR, exist_ok=True)

rag = EndpointConfig.from_args(args).make_rag(WORKING_DIR)

# Now QA
QUESTION_LIST = []
GA_LIST = []
with open(QUERY_PATH, mode="r", encoding="utf-8") as question_file:
    reader = csv.DictReader(question_file)
    for row in reader:
        QUESTION_LIST.append(row["Question"])
        GA_LIST.append(row["Gold Answer"])


def run_experiment(output_path):
    headers = ["Question", "Gold Answer", "minirag"]

    q_already = []
    if os.path.exists(output_path):
        with open(output_path, mode="r", encoding="utf-8") as question_file:
            reader = csv.DictReader(question_file)
            for row in reader:
                q_already.append(row["Question"])

    row_count = len(q_already)
    print("row_count", row_count)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, mode="a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        if row_count == 0:
            writer.writerow(headers)

        for QUESTIONid in trange(row_count, len(QUESTION_LIST)):  #
            QUESTION = QUESTION_LIST[QUESTIONid]
            Gold_Answer = GA_LIST[QUESTIONid]
            print()
            print("QUESTION", QUESTION)
            print("Gold_Answer", Gold_Answer)

            try:
                minirag_answer = (
                    rag.query(QUESTION, param=QueryParam(mode="mini"))
                    .replace("\n", "")
                    .replace("\r", "")
                )
            except Exception as e:
                print("Error in minirag_answer", e)
                minirag_answer = "Error"

            writer.writerow([QUESTION, Gold_Answer, minirag_answer])

    print(f"Experiment data has been recorded in the file: {output_path}")


run_experiment(OUTPUT_PATH)
