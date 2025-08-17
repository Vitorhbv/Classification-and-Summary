from typing import List, Dict
from loguru import logger
import re

# ---------- Constantes e Modelos ----------
SUMMARIZATION_MODEL = "HuggingFaceTB/SmolLM3-3B"
ZERO_SHOT_MODEL = "joeddav/xlm-roberta-large-xnli"
SHORT_WORDS_THRESHOLD = 6 # textos curtos usam resumo rule-based
device= -1 # CPU; mude para 0 se tiver GPU

# ---------- Tentativa de carregar transformers ----------
try:
    from transformers import pipeline
except Exception as e:
    pipeline = None
    logger.warning(f"Transformers indisponível: {e}")

DEFAULT_CATEGORIES = [
    "Feedback",
    "Reclamação",
    "Suporte técnico",
    "Dúvida",
    "Solicitação de serviço",
]

_SUMMARY = None
_ZS = None

# ---------- Helpers ----------
def _postprocess_summary(raw: str, max_sentences: int) -> str:
    """
    Limpa prefixos (ex.: 'Resumo:'), remove duplicatas e limita a N frases.
    """
    if not raw:
        return ""
    t = re.sub(r"\s+", " ", raw).strip()

    # remove marcadores de cabeçalho comuns
    t = re.sub(r"(?i)\bresumo\s*:\s*", "", t)

    # quebra em frases
    sents = re.split(r"(?<=[.!?])\s+", t)

    seen, out = set(), []
    for s in sents:
        s = s.strip(' "«»“”')
        if not s:
            continue
        # remove 'Resumo' que porventura restou no início da sentença
        s = re.sub(r"(?i)^\s*resumo\s*[-—:]\s*", "", s).strip()
        # chaves para deduplicação (case-insensitive e sem pontuação)
        key = re.sub(r"\W+", "", s.lower())
        if len(s) < 3 or key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= max_sentences:
            break

    if not out:
        return ""
    return " ".join(out) + (" ..." if len(sents) > max_sentences else "")

def _rb_summary_pt(text: str) -> str:
    """Resumo determinístico para frases muito curtas (3ª pessoa)."""
    t = (text or "").strip().rstrip(".")
    m = re.match(r"^\s*(solicito|gostaria de|quero|preciso)\s+(.*)$", t, flags=re.IGNORECASE)
    if m:
        core = m.group(2)
        t = f"Solicita {core}"
    t = t[0].upper() + t[1:] if t else t
    if t and not t.endswith("."):
        t += "."
    return t

def _fallback_summary(text: str, max_sentences: int = 3) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    sents = re.split(r"(?<=[.!?])\s+", text)
    resumo = " ".join(sents[:max_sentences]).strip()
    if len(sents) > max_sentences:
        resumo += " ..."
    return f"(Resumo automático simples) {resumo}"

def _fallback_classify(text: str, labels: List[str]) -> Dict[str, float]:
    text_l = (text or "").lower()
    labels = labels or DEFAULT_CATEGORIES
    rules = {
        "Reclamação": ["erro", "não funciona", "demora", "reclama"],
        "Suporte técnico": ["bug", "instalar", "acesso", "senha", "configurar", "técnico"],
        "Dúvida": ["como", "onde", "posso", "duvida", "dúvida"],
        "Solicitação de serviço": ["pedido", "solicito", "provisionar", "ativar", "criar"],
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


# ---------- Pipelines ----------
def get_summarizer():
    """
    Qwen2-1.5B-Instruct via text-generation.
    CPU por padrão (device=-1). Se tiver GPU, mude para 0.
    """
    global _SUMMARY
    if _SUMMARY is not None:
        return _SUMMARY

    if pipeline is None:
        logger.warning("Sem transformers; usando fallback de resumo.")
        _SUMMARY = "FALLBACK"
        return _SUMMARY

    try:
        _SUMMARY = pipeline(
            task="text-generation",
            model=SUMMARIZATION_MODEL,
            tokenizer=SUMMARIZATION_MODEL,
            device=device,
            # Se tiver GPU NVIDIA e quiser economizar VRAM:
            # model_kwargs={"torch_dtype": "auto"},  # + bitsandbytes p/ 8-bit (Linux)
        )
        _ = _SUMMARY("Resuma em 1 frase: teste.", max_new_tokens=24, do_sample=False)
        logger.info(f"Summarizer carregado: {SUMMARIZATION_MODEL} (device={device})")
        return _SUMMARY
    except Exception as e:
        logger.error(f"Falha carregando summarizer '{SUMMARIZATION_MODEL}': {e}")
        _SUMMARY = "FALLBACK"
        return _SUMMARY
def get_zero_shot():

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


# ---------- APIs públicas ----------
def summarize_pt(text: str, max_sentences: int = 3) -> str:
    """
    Resumo com flan-T5-large (text-summarization), com regra para textos curtos
    e pós-processamento.
    """
    text = (text or "").strip()
    if not text:
        return ""

    #textos muito curtos: regra determinística (evita saídas vazias/eco)
    n_words = len(re.findall(r"\w+", text, flags=re.UNICODE))
    if n_words <= SHORT_WORDS_THRESHOLD:
        return _rb_summary_pt(text)

    summ = get_summarizer()
    if summ == "FALLBACK":
        return _fallback_summary(text, max_sentences)

    try:
        prompt = f"""
Você é um analista de suporte. Resuma o texto do chamado em português do Brasil,
em apenas uma frase curta e objetiva, na terceira pessoa.
Não escreva cabeçalhos nem explique seus passos.

Texto:
\"\"\"{text}\"\"\"
Saída:
""".strip()

        out = summ(
            prompt,
            max_new_tokens=80,        # 2-3 frases
            do_sample=False,
            num_beams=2,              # beams altos tendem a repetir
            temperature=0.0,
            repetition_penalty=1.25,
            no_repeat_ngram_size=3,
            return_full_text=False,
            pad_token_id=summ.tokenizer.eos_token_id,
            eos_token_id=summ.tokenizer.eos_token_id,
        )[0]["generated_text"].strip()

        return _postprocess_summary(out, max_sentences=1)

    except Exception as e:
        logger.error(f"Erro no summarizer (SmolLM3): {e}")
        return _fallback_summary(text, max_sentences=3)

def classify_zero_shot_pt(text: str, labels: List[str] = None) -> Dict[str, float]:
    text = (text or "").strip()
    labels = [l for l in (labels or DEFAULT_CATEGORIES) if l]
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
        best["scores"] = {lbl: float(sc) for lbl, sc in zip(out["labels"], out["scores"])}
        return best
    except Exception as e:
        logger.error(f"Erro no zero-shot: {e}")
        return _fallback_classify(text, labels)
