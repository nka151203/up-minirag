# NDAKG-VNH3045 dataset (copied from NDAKG)

This folder lets the clean MiniRAG pipeline run on the NDAKG historical
benchmark without using any code from `NDAKG/`.

## Provenance

| Part | Source | Copy |
|---|---|---|
| Knowledge corpus | `NDAKG/adnkg/dataset/.fined/wiki-text/` | `data/` (5 `.txt`, see note) |
| Multiple-choice questions | `NDAKG/adnkg/dataset/.fined/query_set_VNHXX3045.csv` | `qa/query_set.csv` (176 rows) |

Files are copied verbatim; no cleaning or relabelling was applied.

## Corpus layout

```
data/
├── Overview/Vietnam_History_Summary_Wiki.txt   (34 chunks)
└── VN_1930_1945/*.txt                          (46 chunks)
```

- Vietnamese plain text.
- Current checkout: **5 files, 50 chunks** with MiniRAG defaults
  (`chunk_token_size=1200`, `chunk_overlap_token_size=100`).
- 6 files were randomly removed to reduce API calls for a trial run
  (`Viet_Minh`, `Phu_Rieng_Do`, `1945_Famine`, `Ba_To_Uprising`,
  `Bac_Son_Uprising`, `Yen_Bai_Uprising`). The full 11-file corpus
  (80 chunks) is still available in
  `NDAKG/adnkg/dataset/.fined/wiki-text/` and can be copied back.
- Accuracy on questions whose evidence lies in a removed file will be lower.

## Question layout

`qa/query_set.csv` columns:

| Column | Meaning |
|---|---|
| `Question` | Question text including four options `A.`–`D.` |
| `Gold Answer` | Text of the correct option |
| `Evidence` | Usually empty in this release |
| `Type` | `Trắc nghiệm` (multiple choice) |
| `Period` | Source period, e.g. `VN_1930_1945` |

`reproduce/Step_1_QA.py` reads `Question` and `Gold Answer`, so this CSV is
directly compatible.

## Reproduce

From the repository root (`MiniRAG/`), with `.env` configured:

```bash
PYTHON=/path/to/venv/bin/python ./reproduce/run_ndakg_vnh3045.sh all
```

The output directory follows
`./outputs/NDAKG-VNH3045-{LLM_MODEL}-{EMBEDDING_BACKEND}`, with `/`, `:` and
spaces replaced by `_`. With the current `.env`
(`LLM_MODEL=Qwen/Qwen3.5-9B`, `EMBEDDING_BACKEND=hf`) that is:

```text
./outputs/NDAKG-VNH3045-Qwen_Qwen3.5-9B-hf/
├── graph_chunk_entity_relation.graphml
├── qa_output.csv
└── qa_eval.csv
```

Equivalent manual commands:

```bash
python3 reproduce/Step_0_index.py \
  --workingdir ./outputs/NDAKG-VNH3045-Qwen_Qwen3.5-9B-hf \
  --datapath ./dataset/NDAKG-VNH3045/data

python3 reproduce/Step_1_QA.py \
  --workingdir ./outputs/NDAKG-VNH3045-Qwen_Qwen3.5-9B-hf \
  --querypath ./dataset/NDAKG-VNH3045/qa/query_set.csv \
  --outputpath ./outputs/NDAKG-VNH3045-Qwen_Qwen3.5-9B-hf/qa_output.csv

python3 reproduce/Step_2_eval.py \
  --input  ./outputs/NDAKG-VNH3045-Qwen_Qwen3.5-9B-hf/qa_output.csv \
  --output ./outputs/NDAKG-VNH3045-Qwen_Qwen3.5-9B-hf/qa_eval.csv
```

`.env` defaults to `EMBEDDING_BACKEND=hf` / `EMBEDDING_DIM=384`; the embedding
must be identical between indexing and QA. The LLM (`LLM_MODEL`) and base URL
(`OPENAI_API_BASE`) also come from `.env`.

`Step_0_A_graph_statistic.py` can be pointed at the same working directory for
node/edge statistics.
