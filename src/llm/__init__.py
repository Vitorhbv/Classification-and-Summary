"""Utilities de LLM para sumarização e classificação (PT-BR).

Contém helpers leves, fallbacks e wrappers para pipelines do HuggingFace
(usando transformers.pipeline quando disponível).
"""

from .classification import DEFAULT_CATEGORIES, classify_zero_shot_pt, get_zero_shot
from .summarization import get_summarizer, summarize_pt

__all__ = [
    "DEFAULT_CATEGORIES",
    "classify_zero_shot_pt",
    "get_summarizer",
    "get_zero_shot",
    "summarize_pt",
]
