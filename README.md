
# Triagem Inteligente (Gradio) — Versão Resiliente
- **Resumo**: `flan-t5-small` (CPU), com fallback local se o download falhar
- **Classificação zero-shot**: `xlm-roberta-large-xnli` (CPU), com fallback por regras se falhar
- **UI**: Gradio (texto único + CSV em lote)

## Rodar local
```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python app_gradio.py
```

Se tiver problema com `torch`, use (CPU):
```
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
```

## Docker
```
docker build -t triagem-gradio:cpu .
docker run -p 7860:7860 triagem-gradio:cpu
```

## Notas
- Sem internet para baixar os modelos? O app **não quebra**: usa heurísticas simples como fallback.
- Informe a coluna de texto no CSV (ex.: `descricao`).
