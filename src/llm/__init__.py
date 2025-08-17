"""Utilities de LLM para sumarização e classificação (PT-BR).

Contém helpers leves, fallbacks e wrappers para pipelines do HuggingFace
(usando transformers.pipeline quando disponível).
"""

from .summarization import summarize_pt, get_summarizer
from .classification import classify_zero_shot_pt, get_zero_shot, DEFAULT_CATEGORIES

__all__ = [
    "summarize_pt",
    "get_summarizer",
    "classify_zero_shot_pt",
    "get_zero_shot",
    "DEFAULT_CATEGORIES",
]