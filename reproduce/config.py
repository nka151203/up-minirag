"""
Cấu hình dùng chung cho MiniRAG với endpoint OpenAI-compatible.

Chỉ cần điền 3 thông tin cho phần LLM:
  OPENAI_API_BASE   - URL endpoint, ví dụ https://api.openai.com/v1
  OPENAI_API_KEY    - API key
  LLM_MODEL         - tên model, ví dụ gpt-4o-mini

Embedding mặc định vẫn chạy local bằng ``sentence-transformers/all-MiniLM-L6-v2``
(giống paper). Có thể chuyển embedding sang chính endpoint đó bằng
``--embedding-backend openai``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Nạp .env ở thư mục gốc repo, không phụ thuộc vào cwd hiện tại.
load_dotenv(
    dotenv_path=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
)

from minirag import MiniRAG
from minirag.llm import hf_embed, openai_complete, openai_embed
from minirag.utils import EmbeddingFunc

HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_EMBEDDING_DIM = 384

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_DIM = 1536


def add_llm_args(parser):
    """Thêm các tham số endpoint OpenAI-compatible vào argparse parser."""
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_API_BASE"),
        help="Base URL của endpoint OpenAI-compatible (default: $OPENAI_API_BASE)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="API key của endpoint (default: $OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL"),
        help="Tên model dùng để sinh văn bản (default: $LLM_MODEL)",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["hf", "openai"],
        default=os.getenv("EMBEDDING_BACKEND", "hf"),
        help="hf = all-MiniLM-L6-v2 chạy local (mặc định); openai = dùng endpoint",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL"),
        help="Tên model embedding (mặc định phụ thuộc --embedding-backend)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=int(os.environ["EMBEDDING_DIM"]) if os.getenv("EMBEDDING_DIM") else None,
        help="Số chiều embedding; phải khớp với model (hf: 384, text-embedding-3-small: 1536)",
    )
    return parser


@dataclass
class EndpointConfig:
    base_url: str
    api_key: str
    llm_model: str
    embedding_backend: str = "hf"
    embedding_model: Optional[str] = None
    embedding_dim: Optional[int] = None

    @classmethod
    def from_args(cls, args) -> "EndpointConfig":
        missing = []
        if not args.base_url:
            missing.append("--base-url hoặc OPENAI_API_BASE")
        if not args.api_key:
            missing.append("--api-key hoặc OPENAI_API_KEY")
        if not args.model:
            missing.append("--model hoặc LLM_MODEL")
        if missing:
            raise SystemExit("Thiếu cấu hình: " + ", ".join(missing))

        if args.embedding_backend == "hf":
            embedding_model = args.embedding_model or HF_EMBEDDING_MODEL
            embedding_dim = args.embedding_dim or HF_EMBEDDING_DIM
        else:
            embedding_model = args.embedding_model or OPENAI_EMBEDDING_MODEL
            embedding_dim = args.embedding_dim or OPENAI_EMBEDDING_DIM

        return cls(
            base_url=args.base_url.rstrip("/"),
            api_key=args.api_key,
            llm_model=args.model,
            embedding_backend=args.embedding_backend,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
        )

    def make_rag(self, working_dir: str) -> MiniRAG:
        """Tạo MiniRAG dùng endpoint OpenAI-compatible cho toàn bộ phần LLM."""
        # Đặt key qua biến môi trường để không bị ghi vào config/log của MiniRAG.
        os.environ["OPENAI_API_KEY"] = self.api_key
        os.environ["OPENAI_API_BASE"] = self.base_url

        return MiniRAG(
            working_dir=working_dir,
            llm_model_func=openai_complete,
            llm_model_name=self.llm_model,
            llm_model_kwargs={"base_url": self.base_url},
            llm_model_max_token_size=200,
            embedding_func=self._embedding_func(),
        )

    def _embedding_func(self) -> EmbeddingFunc:
        if self.embedding_backend == "hf":
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
            embed_model = AutoModel.from_pretrained(self.embedding_model)
            return EmbeddingFunc(
                embedding_dim=self.embedding_dim,
                max_token_size=1000,
                func=lambda texts: hf_embed(
                    texts, tokenizer=tokenizer, embed_model=embed_model
                ),
            )

        return EmbeddingFunc(
            embedding_dim=self.embedding_dim,
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts,
                model=self.embedding_model,
                base_url=self.base_url,
            ),
        )
