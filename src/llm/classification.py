"""Funções e pipelines de classificação."""

from .base import logger, pipeline

ZERO_SHOT_MODEL = "joeddav/xlm-roberta-large-xnli"
DEFAULT_CATEGORIES = [
    "Feedback",
    "Reclamação",
    "Suporte técnico",
    "Dúvida",
    "Solicitação de serviço",
]

_ZS = None


def _fallback_classify(text: str, labels: list[str]) -> dict[str, float]:
    """
    Classificador heurístico de fallback (rules-based) para cenários sem modelo.

    Usa palavras-chave para atribuir pontuações simples às categorias
    fornecidas e retorna a melhor categoria encontrada.

    Args:
        text: Texto a ser classificado.
        labels: Lista de rótulos candidatos.

    Returns:
        Dicionário com a chave 'label' (melhor rótulo) e 'scores' mapeando
        cada rótulo para sua pontuação (float).
    """
    text_l = (text or "").lower()
    labels = labels or DEFAULT_CATEGORIES
    rules = {
        "Reclamação": ["erro", "não funciona", "demora", "reclama"],
        "Suporte técnico": [
            "bug",
            "instalar",
            "acesso",
            "senha",
            "configurar",
            "técnico",
        ],
        "Dúvida": ["como", "onde", "posso", "duvida", "dúvida"],
        "Solicitação de serviço": [
            "pedido",
            "solicito",
            "provisionar",
            "ativar",
            "criar",
        ],
        "Feedback": ["sugestão", "gostei", "melhorar", "ideia"],
    }
    scores = {lbl: 0.01 for lbl in labels}
    for lbl, kws in rules.items():
        if lbl not in scores:
            continue
        for kw in kws:
            if kw in text_l:
                scores[lbl] += 0.2
    best = max(scores, key=scores.get) if scores else ""
    return {"label": best, "scores": scores}


def get_zero_shot():
    """
    Inicializa (ou retorna em cache) o pipeline de zero-shot-classification.

    Quando transformers não estiver disponível, retorna o marcador de fallback.

    Returns:
        Pipeline do transformers ou a string "FALLBACK" em caso de indisponibilidade.
    """
    global _ZS
    if _ZS is not None:
        return _ZS
    if pipeline is None:
        logger.warning("Sem transformers; usando fallback de classificação.")
        _ZS = "FALLBACK"
        return _ZS
    try:
        _ZS = pipeline(
            task="zero-shot-classification",
            model=ZERO_SHOT_MODEL,
            device=-1,
        )
        _ = _ZS("Teste", candidate_labels=DEFAULT_CATEGORIES)
        logger.info(f"Zero-shot carregado: {ZERO_SHOT_MODEL} (device={-1})")
        return _ZS
    except Exception as e:
        logger.error(f"Falha carregando zero-shot '{ZERO_SHOT_MODEL}': {e}")
        _ZS = "FALLBACK"
        return _ZS


def classify_zero_shot_pt(text: str, labels: list[str] | None = None) -> dict[str, float]:
    """
    Classifica o texto em rótulos fornecidos usando zero-shot ou fallback.

    Tenta usar o pipeline de zero-shot; se indisponível, usa um classificador
    heurístico simples.

    Args:
        text: Texto a ser classificado.
        labels: Lista de rótulos candidatos. Se None, usa DEFAULT_CATEGORIES.

    Returns:
        Dicionário contendo ao menos as chaves 'label' (o rótulo escolhido)
        e 'scores' (mapeamento rótulo->score).
    """
    text = (text or "").strip()
    labels = [lbl for lbl in (labels or DEFAULT_CATEGORIES) if lbl]
    if not text or not labels:
        return {"label": "", "scores": {}}

    z = get_zero_shot()
    if z == "FALLBACK":
        return _fallback_classify(text, labels)

    try:
        out = z(
            text,
            candidate_labels=labels,
            hypothesis_template="This text is about {}.",  # geralmente mais estável
        )
        best = {"label": out["labels"][0], "score": float(out["scores"][0])}
        best["scores"] = {
            lbl: float(sc) for lbl, sc in zip(out["labels"], out["scores"], strict=False)
        }
        return best
    except Exception as e:
        logger.error(f"Erro no zero-shot: {e}")
        return _fallback_classify(text, labels)
