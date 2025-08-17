"""Funções utilitárias para manipulação de CSVs e processamento de texto."""

import pandas as pd
from typing import List, Union, Tuple, Dict
from loguru import logger
import tempfile
import os
import chardet
import gradio as gr
from src.llm import summarize_pt, classify_zero_shot_pt, DEFAULT_CATEGORIES

def parse_labels(cats: Union[str, List[str], None]):
    """Converte uma entrada de categorias em uma lista de strings.

    Args:
        cats (Union[str, List[str], None]): String com categorias separadas por
            vírgula ou ponto-e-vírgula, ou iterável de strings (list/tuple/set).

    Returns:
        List[str]: Lista de categorias limpas; retorna lista vazia se `cats` for
        None ou inválido.
    """
    # aceita string "A, B, C" ou lista/tupla
    if isinstance(cats, str):
        return [c.strip() for c in cats.replace(";", ",").split(",") if c.strip()]
    if isinstance(cats, (list, tuple, set)):
        return [str(c).strip() for c in cats if str(c).strip()]
    return []

def _read_csv_smart(file_path: str, sep: str = ";") -> pd.DataFrame:
    """Lê um CSV tentando detectar automaticamente a codificação.

    Args:
        file_path (str): Caminho para o arquivo CSV.
        sep (str): Separador de colunas (padrão ';').

    Returns:
        pandas.DataFrame: DataFrame lido a partir do CSV.
    """
    try:
        with open(file_path, "rb") as f:
            raw = f.read(4096)
        enc = chardet.detect(raw)["encoding"] or "utf-8"
    except Exception:
        enc = "utf-8"
    return pd.read_csv(file_path, sep=sep or ";", encoding=enc)


def process_text_single(texto: str, categorias):
    """Processa um texto único: gera resumo e realiza classificação.

    Args:
        texto (str): Texto do chamado a ser processado.
        categorias (Union[str, List[str], None]): Categorias a serem usadas para
            classificação (string separada por vírgula ou iterável).

    Returns:
        Tuple[str, str, Dict[str, float]]: Tupla com (resumo, rótulo_predito, scores).
    """
    labels = parse_labels(categorias) or DEFAULT_CATEGORIES
    
    resumo = summarize_pt(texto)  
    cls = classify_zero_shot_pt(texto, labels)
    return resumo, cls.get("label", ""), cls.get("scores", {})

def process_csv(file, col_texto: str, categorias_texto: str = "", sep: str = ";"):
    """Processa um arquivo CSV aplicando resumo e classificação por linha.

    Lê o CSV fornecido (detectando codificação), aplica summarize_pt e
    classify_zero_shot_pt na coluna indicada e salva um CSV com as colunas
    adicionais 'resumo' e 'categoria_llm' em um diretório temporário.

    Args:
        file: Objeto de arquivo ou caminho fornecido pelo componente Gradio.
        col_texto (str): Nome da coluna no CSV que contém o texto a ser processado.
        categorias_texto (str): Categorias em formato string separadas por vírgula
            (opcional). Se vazio, usa DEFAULT_CATEGORIES.
        sep (str): Separador do CSV (padrão ';').

    Returns:
        Tuple[pandas.DataFrame, str]: Prévia (até 15 linhas) do DataFrame de saída
        e o caminho absoluto do CSV processado salvo em diretório temporário.

    Raises:
        gr.Error: Se ocorrer qualquer erro durante a leitura ou processamento do CSV.
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

        resumos = [summarize_pt(t) for t in textos]  
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