from loguru import logger

try:
    from transformers import pipeline
except Exception as e:
    pipeline = None
    logger.warning(f"Transformers indisponível: {e}")