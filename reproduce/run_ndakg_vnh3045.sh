#!/usr/bin/env bash
# Reproduce MiniRAG on the NDAKG-VNH3045 dataset.
#
# Usage:
#   PYTHON=/path/to/venv/bin/python ./reproduce/run_ndakg_vnh3045.sh [index|qa|eval|all]
#
# Requires a .env at the repository root with:
#   OPENAI_API_BASE, OPENAI_API_KEY, LLM_MODEL
# Embedding defaults to local all-MiniLM-L6-v2 (384d); override with
# EMB_BACKEND / EMB_MODEL / EMB_DIM (e.g. EMB_BACKEND=openai EMB_DIM=3072).
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"

# Read a value from .env without executing the file.
env_value() {
  [ -f .env ] || return 0
  sed -n "s/^[[:space:]]*$1[[:space:]]*=//p" .env | tail -n 1 \
    | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
          -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//" \
    | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

# Make a value safe for use in a directory name.
sanitize_tag() {
  printf '%s' "$1" \
    | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
    | tr -s '/: ' '_' \
    | sed -e 's/^_*//' -e 's/_*$//'
}

# LLM comes from .env; embedding defaults come from .env too, but can be
# overridden with EMB_BACKEND / EMB_MODEL / EMB_DIM.
EMB_BACKEND="${EMB_BACKEND:-$(env_value EMBEDDING_BACKEND)}"; EMB_BACKEND="${EMB_BACKEND:-hf}"
EMB_MODEL="${EMB_MODEL:-$(env_value EMBEDDING_MODEL)}"; EMB_MODEL="${EMB_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
EMB_DIM="${EMB_DIM:-$(env_value EMBEDDING_DIM)}"; EMB_DIM="${EMB_DIM:-384}"

LLM_TAG="$(sanitize_tag "${LLM_MODEL:-$(env_value LLM_MODEL)}")"
EMB_TAG="$(sanitize_tag "$EMB_BACKEND")"

# OpenAI SDK appends /chat/completions itself; accept a full endpoint URL too.
LLM_BASE="${OPENAI_API_BASE:-$(env_value OPENAI_API_BASE)}"
LLM_BASE="${LLM_BASE%/}"
LLM_BASE="${LLM_BASE%/chat/completions}"
LLM_BASE="${LLM_BASE%/}"
BASE_ARGS=()
[ -n "$LLM_BASE" ] && BASE_ARGS=(--base-url "$LLM_BASE")

WORKDIR="${WORKDIR:-./outputs/NDAKG-VNH3045-${LLM_TAG}-${EMB_TAG}}"
DATA="${DATA:-./dataset/NDAKG-VNH3045/data}"
QUERIES="${QUERIES:-./dataset/NDAKG-VNH3045/qa/query_set.csv}"
QA_OUT="${QA_OUT:-$WORKDIR/qa_output.csv}"
EVAL_OUT="${EVAL_OUT:-$WORKDIR/qa_eval.csv}"

EMB_ARGS=(--embedding-backend "$EMB_BACKEND" --embedding-model "$EMB_MODEL" --embedding-dim "$EMB_DIM")

export PYTHONUTF8=1
export MAX_PARALLEL_INSERT="${MAX_PARALLEL_INSERT:-2}"

stage="${1:-all}"
mkdir -p "$WORKDIR"
echo "==> Working dir: $WORKDIR"
echo "==> Embedding: $EMB_BACKEND / $EMB_MODEL (dim=$EMB_DIM)"

if [[ "$stage" == "index" || "$stage" == "all" ]]; then
  echo "==> Indexing $DATA into $WORKDIR"
  "$PYTHON" reproduce/Step_0_index.py \
    --workingdir "$WORKDIR" \
    --datapath "$DATA" \
    "${BASE_ARGS[@]}" \
    "${EMB_ARGS[@]}"
fi

if [[ "$stage" == "qa" || "$stage" == "all" ]]; then
  echo "==> Answering $QUERIES"
  "$PYTHON" reproduce/Step_1_QA.py \
    --workingdir "$WORKDIR" \
    --querypath "$QUERIES" \
    --outputpath "$QA_OUT" \
    "${BASE_ARGS[@]}" \
    "${EMB_ARGS[@]}"
fi

if [[ "$stage" == "eval" || "$stage" == "all" ]]; then
  echo "==> Judging $QA_OUT"
  "$PYTHON" reproduce/Step_2_eval.py \
    --input "$QA_OUT" \
    --output "$EVAL_OUT" \
    "${BASE_ARGS[@]}"
fi
