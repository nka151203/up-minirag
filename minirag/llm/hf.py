"""
Hugging Face Embedding Helper for MiniRAG
=========================================

MiniRAG chỉ dùng phần embedding local từ Hugging Face. Toàn bộ phần sinh văn bản
(LLM completion) đã được chuyển sang endpoint OpenAI-compatible trong
``minirag/llm/openai.py``.
"""

import os

import pipmaster as pm  # Pipmaster for dynamic library install

# install specific modules
if not pm.is_installed("transformers"):
    pm.install("transformers")
if not pm.is_installed("torch"):
    pm.install("torch")

import numpy as np
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"


async def hf_embed(texts: list[str], tokenizer, embed_model) -> np.ndarray:
    """Mean-pool the last hidden state of a Hugging Face encoder."""
    device = next(embed_model.parameters()).device
    input_ids = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    ).input_ids.to(device)
    with torch.no_grad():
        outputs = embed_model(input_ids)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    if embeddings.dtype == torch.bfloat16:
        return embeddings.detach().to(torch.float32).cpu().numpy()
    else:
        return embeddings.detach().cpu().numpy()
