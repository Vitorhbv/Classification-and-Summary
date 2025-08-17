"""App Gradio para Triagem Inteligente (PT-BR).

Interface mínima para gerar resumos e classificar textos de chamados
(usando src.llm). Suporta entrada de texto único e processamento em lote
via CSV. Fornece fallbacks locais quando os modelos não estão disponíveis.
"""

import gradio as gr

from src.llm import DEFAULT_CATEGORIES
from src.utils.csv_tools import process_csv, process_text_single

with gr.Blocks(title="Triagem Inteligente — MVP (LLM Open-Source)") as demo:
    gr.Markdown("# Triagem Inteligente — Resumo + Classificação (LLM Open-Source)")
    gr.Markdown(
        "Se os modelos não puderem ser baixados, o app usa fallback local."
    )

    with gr.Tab("Texto Único"):
        inp_text = gr.Textbox(
            label="Descrição do chamado (texto livre)",
            lines=8,
            placeholder="Cole aqui a descrição do chamado...",
        )
        inp_cats = gr.Textbox(
            label="Categorias (separadas por vírgula)",
            value=", ".join(DEFAULT_CATEGORIES),
        )
        btn = gr.Button("Processar")
        out_resumo = gr.Textbox(label="Resumo")
        out_label = gr.Textbox(label="Categoria prevista")
        out_scores = gr.JSON(label="Scores por categoria")
        btn.click(
            process_text_single,
            inputs=[inp_text, inp_cats],
            outputs=[out_resumo, out_label, out_scores],
        )

    with gr.Tab("CSV em Batch"):
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
