# Triagem Inteligente - Blip

Aplicação simples para resumir e classificar mensagens de chamados em português (PT‑BR).
A proposta é oferecer uma interface que utiliza LLMs open‑source quando disponíveis
e heurísticas de fallback quando não for possível baixar ou executar os modelos.

Principais tecnologias
- Gradio: UI web leve para inputs/outputs interativos (texto único e upload de CSV).
- Transformers / Hugging Face: pipelines opcionais para sumarização e classificação (zero‑shot).
- Pandas: leitura e escrita de CSV, manipulação de dados em batch.
- Docker: empacotamento da aplicação para execução isolada.
- Loguru: logging.

Modelos utilizados
- HuggingFaceTB/SmolLM3-3B : LLM simples de 3B de parâmetros usado para summarizar
- joeddav/xlm-roberta-large-xnli: Multilingual LLM treinado para zero-shot classification.

Como funciona (fluxo)
1. Usuário fornece input via interface Gradio (web):
   - Texto único: campo livre com a mensagem do chamado (ex.: reclamação, suporte, feedback).
   - CSV em batch: upload de arquivo; o usuário informa a coluna que contém a descrição/mensagem.
2. Backend processa cada texto chamando dois passos:
   - summarize_pt(text) → gera um resumo curto (modelo HF ou fallback heurístico).
   - classify_zero_shot_pt(text, labels) → rótulo + scores (modelo zero‑shot ou fallback).
3. Resultados são exibidos na UI; no caso do CSV, um arquivo processado é salvo e retornado com duas colunas adicionais:
   - resumo: texto gerado pelo LLM (ou fallback)
   - categoria_llm: rótulo predito pelo LLM (ou fallback)

Inputs e categorias
- Categorias padrão: existem categorias pré‑definidas no sistema (por exemplo: Feedback, Reclamação, Suporte técnico, Dúvida, Solicitação de serviço).
- Categorias customizadas: o usuário pode adicionar manualmente outras categorias pela interface (campo de texto separadas por vírgula).
- CSV batch: ao subir um CSV, selecione a coluna que contém a mensagem/descrição — o LLM fará o resumo e a classificação baseado no conteúdo dessa coluna.

Nota: existe uma pasta `datasets/` no repositório contendo arquivos CSV de exemplo (por exemplo `exemplos_chamados.csv` e `exemplos_chamados_longos.csv`) que podem e foram usados como input de teste para o fluxo de CSV em lote.

Limitações — LLMs open‑source pequenos / CPU
- Modelos com poucos parâmetros ou executados apenas em CPU têm limitações claras:
  - Imprecisão: podem perder contexto, omitir detalhes importantes ou produzir resumos genéricos que não refletem nuances do chamado.
  - Classificação fraca em casos sutis: zero‑shot sem fine‑tuning(não tenho gpu) tende a confundir categorias próximas.
  - Latência: inferência em CPU é mais lenta, especialmente em batch grande.
  - Determinismo / estabilidade: geração pode variar; erros ocasionais são esperados.
- Mitigações neste projeto:
  - Fallback heurístico (rules‑based) para não interromper o fluxo quando modelos não estão disponíveis.
  - Permitir customização de categorias para melhorar sinal sem retraining.
  - Recomendação: para alta precisão, usar modelos maiores via API (LLMs robustas de providers como Openai ou Gemini) ou finetuning no domínio(requer mais poder computacional).


Rodar local 

1. Após clonar o repositório, crie e ative ambiente virtual:

WSL / Linux / macOS:
```
python -m venv .venv
source .venv/bin/activate
```

2. Instale dependências:
```
pip install --upgrade pip
pip install -r requirements.txt ou pip install -e .
```
Se houver problemas com torch na CPU:
```
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
```

3. Inicie a aplicação:
```
python app.py
```
Acesse: http://localhost:7860

Executar com Docker
1. Build da imagem:
```
docker build -t blip-triagem:latest .
```
2. Executar container:
```
docker run --rm -it --name blip-triagem -p 7860:7860 blip-triagem:latest
```

Arquitetura (diagrama simplificado)

```
Browser (usuário)
  |
  +-> Gradio UI (app.py)  [http://localhost:7860]
        |
        +-> Handlers (app.py)
        |     - Texto único -> src.utils.csv_tools.process_text_single()
        |         -> src.llm.summarize_pt() + src.llm.classify_zero_shot_pt()
        |     - CSV batch  -> src.utils.csv_tools.process_csv()
        |         -> pandas lê .csv -> itera linhas -> chama summarize + classify
        |
        +-> Lógica de LLM (src/llm/)
        |     - summarization.py (summarize_pt, get_summarizer)
        |     - classification.py (classify_zero_shot_pt, get_zero_shot)
        |     - base.py (wrapper do pipeline; usa transformers se disponível)
        |
        +-> Utils (src/utils/)
        |     - csv_tools.py (leitura inteligente, parse de labels, processamento batch)
        |
        +-> Dados e I/O
              - datasets/ (exemplos CSV para testes)
              - Saída: arquivo CSV gerado em diretório temporário (tickets_processados.csv)
              - Logs: console (loguru)

Execução/empacotamento
- Dockerfile cria imagem e expõe porta 7860
- requirements.txt / pyproject.toml definem dependências
```

Recomendações 
- Para produção, fixar versões das dependências (requirements com ==) e usar variáveis de ambiente em vez de hardcoding.
- Se precisar de maior qualidade, usar modelos maiores/finetuned e GPU para reduzir latência.

Noções de Qualidade & CI (simples)
[![lint](https://github.com/Vitorhbv/Triagem-Inteligente/actions/workflows/lint.yml/badge.svg)](../../actions/workflows/lint.yml)

- **CI (GitHub Actions):** roda **Ruff** a cada *push*/PR (branches `main`/`feat/**`).
- Objetivo: manter o código consistente e evitar *lint issues* antes de merge.

Notas finais
- O projeto foi desenhado como um MVP: barato, rodável em CPU e fácil de demonstrar.
   Para produção, é recomendado:
   
      - Observabilidade (latência, taxa de fallback, distribuição de rótulos).

      - Avaliação com amostras reais e ground truth.
      
      - LLM robusto com mais de 70B de parâmetros, usar RAG se tiver base de conhecimento contendo exemplos de chamados rotulados.

      - Hospedar na nuvem com GPU e autoscaling (endpoint gerenciado).