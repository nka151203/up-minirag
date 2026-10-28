from minirag.llm.openai import (
    gpt_4o_complete,
    gpt_4o_mini_complete,
    openai_complete,
    openai_complete_if_cache,
    openai_embed,
)


async def hf_embed(*args, **kwargs):
    """Lazy import so torch/transformers are only loaded when local embedding is used."""
    from minirag.llm.hf import hf_embed as _hf_embed

    return await _hf_embed(*args, **kwargs)
