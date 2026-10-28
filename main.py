import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse

from minirag import QueryParam
from reproduce.config import add_llm_args, EndpointConfig


def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG quick start")
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
DATA_PATH = args.datapath

print("USING LLM:", args.model)
print("USING BASE URL:", args.base_url)
print("USING WORKING DIR:", WORKING_DIR)

os.makedirs(WORKING_DIR, exist_ok=True)

rag = EndpointConfig.from_args(args).make_rag(WORKING_DIR)


# Now indexing
def find_txt_files(root_path):
    txt_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".txt"):
                txt_files.append(os.path.join(root, file))
    return txt_files


WEEK_LIST = find_txt_files(DATA_PATH)
for WEEK in WEEK_LIST:
    id = WEEK_LIST.index(WEEK)
    print(f"{id}/{len(WEEK_LIST)}")
    with open(WEEK) as f:
        rag.insert(f.read())

# A toy query
query = 'What does LiHua predict will happen in "The Rings of Power"?'
answer = (
    rag.query(query, param=QueryParam(mode="mini")).replace("\n", "").replace("\r", "")
)
print(answer)
