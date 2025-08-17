import gradio as gr
import pandas as pd
from typing import List, Union
from loguru import logger
from src.llm import summarize_pt, classify_zero_shot_pt, DEFAULT_CATEGORIES
import tempfile
import os
import chardet  # opcional: se não tiver, remova e use encoding padrão

# ---------- Utils ----------
def parse_labels(cats: Union[str, List[str], None]):
    # aceita string "A, B, C" ou lista/tupla
    if isinstance(cats, str):
        return [c.strip() for c in cats.replace(";", ",").split(",") if c.strip()]
    if isinstance(cats, (list, tuple, set)):
        return [str(c).strip() for c in cats if str(c).strip()]
    return []

def _read_csv_smart(file_path: str, sep: str = ";") -> pd.DataFrame:
    """Tenta detectar encoding e lê CSV com separador informado."""
    try:
        with open(file_path, "rb") as f:
            raw = f.read(4096)
        enc = chardet.detect(raw)["encoding"] or "utf-8"
    except Exception:
        enc = "utf-8"
    return pd.read_csv(file_path, sep=sep or ";", encoding=enc)


# ---------- Handlers ----------
def process_text_single(texto: str, categorias):
    labels = parse_labels(categorias) or DEFAULT_CATEGORIES
    # resumo de 1 frase (altere para 3 se quiser)
    resumo = summarize_pt(texto)  # sua summarize_pt já garante 1 frase se você trocou; senão use summarize_pt(texto, max_sentences=1)
    cls = classify_zero_shot_pt(texto, labels)
    return resumo, cls.get("label", ""), cls.get("scores", {})

def process_csv(file, col_texto: str, categorias_texto: str = "", sep: str = ";"):
    """
    Lê o CSV enviado no Gradio, gera 'resumo' (1 frase) e 'categoria_llm',
    salva em diretório temporário e retorna (prévia_dataframe, caminho_arquivo).
    """
    try:
        # Gradio pode entregar str (path) ou objeto/tmpfile; lidamos com ambos
        file_path = file if isinstance(file, str) else getattr(file, "name", None)
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError("Arquivo não encontrado.")

        df = _read_csv_smart(file_path, sep=sep)

        if col_texto not in df.columns:
            raise ValueError(f"Coluna '{col_texto}' não encontrada. Colunas disponíveis: {list(df.columns)}")

        labels = parse_labels(categorias_texto) or DEFAULT_CATEGORIES

        textos = df[col_texto].astype(str).fillna("").tolist()

        # processamento em lote (simples); se quiser, pode fazer cache ou batch
        resumos = [summarize_pt(t) for t in textos]  # 1 frase
        categorias = [classify_zero_shot_pt(t, labels).get("label", "") for t in textos]

        out_df = df.copy()
        out_df["resumo"] = resumos
        out_df["categoria_llm"] = categorias

        # cria diretório temporário exclusivo e salva
        temp_dir = tempfile.mkdtemp(prefix="triagem_")
        out_path = os.path.join(temp_dir, "tickets_processados.csv")
        out_df.to_csv(out_path, index=False)

        return out_df.head(15), out_path

    except Exception as e:
        logger.exception(e)
        raise gr.Error(f"Erro ao processar CSV: {e}")
        return pd.DataFrame(), None


# ---------- UI ----------
with gr.Blocks(title="Triagem Inteligente — MVP (LLM Open-Source)") as demo:
    gr.Markdown("# Triagem Inteligente — Resumo + Classificação (LLM Open-Source)")
    gr.Markdown("Se os modelos não puderem ser baixados, o app **usa fallback local** (regras simples) para não quebrar.")

    with gr.Tab("Texto Único"):
        inp_text = gr.Textbox(label="Descrição do chamado (texto livre)", lines=8, placeholder="Cole aqui a descrição do chamado...")
        inp_cats = gr.Textbox(label="Categorias (separadas por vírgula)", value=", ".join(DEFAULT_CATEGORIES))
        btn = gr.Button("Processar")
        out_resumo = gr.Textbox(label="Resumo")
        out_label = gr.Textbox(label="Categoria prevista")
        out_scores = gr.JSON(label="Scores por categoria")
        btn.click(process_text_single, inputs=[inp_text, inp_cats], outputs=[out_resumo, out_label, out_scores])

    with gr.Tab("CSV em Lote"):
        gr.Markdown("Informe a **coluna de texto** do chamado (ex.: `descricao`).")
        up = gr.File(label="CSV de entrada")
        col = gr.Textbox(label="Coluna de texto do chamado", placeholder="Ex.: descricao")
        cats2 = gr.Textbox(label="Categorias (vírgula)", value=", ".join(DEFAULT_CATEGORIES))
        sep = gr.Textbox(label="Separador CSV", value=";")
        btn2 = gr.Button("Processar CSV")
        out_tbl = gr.Dataframe(label="Prévia (15 linhas)")
        out_file = gr.File(label="Baixar CSV processado")
        btn2.click(process_csv, inputs=[up, col, cats2, sep], outputs=[out_tbl, out_file])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
