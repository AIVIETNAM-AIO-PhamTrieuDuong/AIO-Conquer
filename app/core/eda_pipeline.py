from __future__ import annotations

import io
import os
import tempfile
import asyncio
import uuid

import numpy as np
import pandas as pd

from app.memory.redis_client import memory
from app.model.llm_client import llm


# ---------------------------------------------------------------------------
# Step 1 — analyze_and_clean_data  (mirrors notebook cell-8)
# ---------------------------------------------------------------------------

def analyze_and_clean_data(data_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict, io.StringIO]:
    # Duplicate Handling
    df_clean = data_frame.drop_duplicates().copy()
    removed_duplicates = len(data_frame) - len(df_clean)

    # Column type separation
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    cat_cols = df_clean.select_dtypes(include=["object", "bool", "category"]).columns

    # LLM Buffer — profile text (mirrors notebook llm_buffer)
    llm_buffer = io.StringIO()
    llm_buffer.write("=== Data Shape ===\n")
    llm_buffer.write(f"Rows: {df_clean.shape[0]}, Columns: {df_clean.shape[1]}\n")
    llm_buffer.write(f"Duplicates removed: {removed_duplicates}\n\n")

    info_summary = pd.DataFrame({
        "DataType": df_clean.dtypes,
        "MissingValues": df_clean.isnull().sum(),
        "UniqueValues": df_clean.nunique(),
    })
    llm_buffer.write("=== Missing Values & Data Types ===\n")
    llm_buffer.write(info_summary.to_string() + "\n\n")

    # Null stats before filling
    null_stats = df_clean.isnull().sum().to_dict()

    # Fill nulls
    for col in num_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    for col in cat_cols:
        mode_val = df_clean[col].mode()
        df_clean[col] = df_clean[col].fillna(mode_val[0] if not mode_val.empty else "Unknown")

    # Numerical stats + IQR outlier detection (mirrors notebook exactly)
    num_stats: dict = {}
    df_clean["outlier_col_name"] = [[] for _ in range(len(df_clean))]

    for col in num_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        is_outlier = (df_clean[col] < (Q1 - 1.5 * IQR)) | (df_clean[col] > (Q3 + 1.5 * IQR))

        df_clean.loc[is_outlier, "outlier_col_name"] = df_clean.loc[
            is_outlier, "outlier_col_name"
        ].apply(lambda x: x + [col])

        num_stats[col] = {
            "mean": float(df_clean[col].mean()),
            "std": float(df_clean[col].std()),
            "min": float(df_clean[col].min()),
            "max": float(df_clean[col].max()),
            "uniques": int(df_clean[col].nunique()),
            "nulls_fixed": int(null_stats.get(col, 0)),
            "outliers": int(is_outlier.sum()),
            "outlier_perc": float((is_outlier.sum() / len(df_clean)) * 100),
        }

    df_clean["outlier_col_name"] = df_clean["outlier_col_name"].apply(
        lambda x: x if len(x) > 0 else "0"
    )

    # Categorical stats
    cat_stats: dict = {}
    for col in cat_cols:
        counts = df_clean[col].value_counts()
        cat_stats[col] = {
            "mode": str(counts.index[0]) if not counts.empty else None,
            "uniques": int(df_clean[col].nunique()),
            "nulls_fixed": int(null_stats.get(col, 0)),
            "top_5": {str(k): int(v) for k, v in counts.head(5).items()},
        }

    return df_clean, num_stats, cat_stats, llm_buffer


# ---------------------------------------------------------------------------
# Step 2 — generate_summary_md  (mirrors notebook cell-8 generate_summary_md)
# ---------------------------------------------------------------------------

def generate_summary_md(df: pd.DataFrame, num_stats: dict, cat_stats: dict) -> str:
    lines: list[str] = [
        "# CONQUER QA REPORT: DATA FOOTPRINT\n\n",
        "## 1. Numerical Analysis\n\n",
    ]

    for col, s in num_stats.items():
        warning = " ⚠️ **[HIGH ANOMALY RATE]**" if s["outlier_perc"] > 5 else ""
        lines.append(
            f"### {col}\n"
            f"- **Mean:** {s['mean']:.2f} | **Std:** {s['std']:.2f}\n"
            f"- **Min:** {s['min']:.2f} | **Max:** {s['max']:.2f}\n"
            f"- **Outliers:** {s['outliers']} ({s['outlier_perc']:.2f}%){warning}\n"
            f"- **Nulls fixed:** {s['nulls_fixed']}\n\n"
        )

    lines.append("## 2. Categorical Landscape\n\n")
    for col, s in cat_stats.items():
        top_str = ", ".join(f"{k} ({v})" for k, v in s["top_5"].items())
        lines.append(
            f"### {col}\n"
            f"- **Mode:** {s['mode']}\n"
            f"- **Unique values:** {s['uniques']}\n"
            f"- **Top 5:** {top_str}\n"
            f"- **Nulls fixed:** {s['nulls_fixed']}\n\n"
        )

    total_outlier_rows = len(df[df["outlier_col_name"] != "0"])
    lines.append(
        "## 3. System Alerts & Metadata\n\n"
        f"- **Total rows with flagged anomalies:** {total_outlier_rows}\n"
        f"- **Total rows:** {df.shape[0]} | **Total columns:** {df.shape[1]}\n"
        "- **Data integrity:** Outlier sources preserved in `outlier_col_name` column.\n"
    )

    return "".join(lines)


# ---------------------------------------------------------------------------
# Step 3 — call_llm_for_insight  (mirrors notebook cell-11, uses existing llm)
# ---------------------------------------------------------------------------

async def call_llm_for_insight(summary_text: str, profile_text: str) -> str:
    prompt = f"""You are a Senior Data-Centric AI & QA Specialist.
Analyze the statistical summary below and provide insights in Markdown format.

Guidelines:
1. Analyze outliers — explain how anomalies affect data quality.
2. Bias detection — evaluate categorical distributions for imbalance.
3. Provide 2 actionable recommendations for the modeling team.
4. Describe what each feature likely represents based on its statistics.

STATISTICAL REPORT:
{summary_text}

DATASET PROFILE:
{profile_text}

Respond in professional English, structured Markdown only. No JSON wrapper."""

    try:
        insight = await llm.generate_text(prompt, max_tokens=2048)
        return f"\n\n## 4. LLM Semantic Insights\n\n{insight.strip()}\n"
    except Exception as e:
        return f"\n\n## 4. LLM Semantic Insights\n\n*Insight generation skipped: {e}*\n"


# ---------------------------------------------------------------------------
# Orchestrator — run_eda  (ties all steps together, runs as background task)
# ---------------------------------------------------------------------------

async def run_eda(job_id: str, file_path: str) -> None:
    try:
        # Read file
        if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
            df = await asyncio.get_event_loop().run_in_executor(None, pd.read_excel, file_path)
        else:
            df = await asyncio.get_event_loop().run_in_executor(None, pd.read_csv, file_path)

        # Step 1 — CPU-bound, run in thread pool
        df_clean, num_stats, cat_stats, llm_buffer = await asyncio.get_event_loop().run_in_executor(
            None, analyze_and_clean_data, df
        )

        # Step 2 — fast, run in thread pool
        summary_md = await asyncio.get_event_loop().run_in_executor(
            None, generate_summary_md, df_clean, num_stats, cat_stats
        )

        # Step 3 — async LLM call
        insight = await call_llm_for_insight(summary_md, llm_buffer.getvalue())
        summary_md += insight

        # Save cleaned CSV to temp dir
        tmp_dir = os.path.join(tempfile.gettempdir(), "eda", job_id)
        os.makedirs(tmp_dir, exist_ok=True)
        cleaned_path = os.path.join(tmp_dir, "cleaned.csv")
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: df_clean.to_csv(cleaned_path, index=False)
        )

        payload = {
            "summary_md": summary_md,
            "num_stats": num_stats,
            "cat_stats": cat_stats,
            "profile_text": llm_buffer.getvalue(),
            "shape": {"rows": int(df_clean.shape[0]), "cols": int(df_clean.shape[1])},
            "cleaned_file_path": cleaned_path,
        }
        await memory.set_eda_result(job_id, payload)
        await memory.set_eda_status(job_id, "done")

    except Exception as e:
        await memory.set_eda_status(job_id, f"error:{e}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
