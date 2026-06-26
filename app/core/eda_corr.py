"""Multi-type association ("Correlation Truth Table") for the EDA pipeline.

Computes a mixed-type association matrix over every column pair:
  - Numeric  × Numeric      → Spearman correlation
  - Categorical × Categorical → Cramér's V
  - Numeric  × Categorical  → Eta (correlation ratio)

Ported from the standalone ``run_corr_v2.py`` script — file I/O removed; returns
plain objects so the EDA pipeline can feed ``truth_table_md`` into the LLM prompt.
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# Threshold for listing a pair in the truth table (cut noise for the LLM).
TRUTH_TABLE_THRESHOLD = 0.20

# Helper column injected by analyze_and_clean_data — not real data.
_HELPER_COLS = {"outlier_col_name"}

_ID_KEYWORDS = ["name", "description", "note", "comment", "address", "email"]
_TARGET_KEYWORDS = ["churn", "attrition", "target", "label", "default", "fraud"]


# ---------------------------------------------------------------------------
# Association metrics
# ---------------------------------------------------------------------------

def cramers_v(col_a: pd.Series, col_b: pd.Series) -> float:
    """Cramér's V for categorical × categorical."""
    contingency = pd.crosstab(col_a.fillna("__NA__"), col_b.fillna("__NA__"))
    chi2 = chi2_contingency(contingency)[0]
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    if min_dim == 0 or n == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * max(min_dim, 1))))


def eta_squared(categorical_col: pd.Series, numeric_col: pd.Series) -> float:
    """Eta (correlation ratio, sqrt of eta-squared) for categorical × numeric."""
    groups: dict = {}
    for cat, val in zip(categorical_col, numeric_col):
        if pd.isna(cat) or pd.isna(val):
            continue
        groups.setdefault(cat, []).append(val)

    if len(groups) < 2:
        return 0.0

    all_vals: list = []
    for vals in groups.values():
        all_vals.extend(vals)
    grand_mean = np.mean(all_vals)

    ss_between = sum(len(vals) * (np.mean(vals) - grand_mean) ** 2 for vals in groups.values())
    ss_total = sum((v - grand_mean) ** 2 for v in all_vals)

    if ss_total == 0:
        return 0.0
    return float(np.sqrt(ss_between / ss_total))


# ---------------------------------------------------------------------------
# Column classification + target detection
# ---------------------------------------------------------------------------

def classify_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return (valid_cols, numeric_cols, categorical_cols, excluded_cols)."""
    excluded_cols: list[str] = []
    valid_cols: list[str] = []

    for col in df.columns:
        if col in _HELPER_COLS:
            excluded_cols.append(col)
            continue

        col_lower = col.lower().strip()

        # High-cardinality identifier detection
        is_id = False
        if col_lower in ("id", "index"):
            is_id = True
        elif col_lower.endswith(" id") or col_lower.endswith("_id"):
            is_id = True
        elif any(kw in col_lower for kw in _ID_KEYWORDS):
            is_id = True

        # Object columns with >50% unique values look like free-text / ids
        if not is_id and pd.api.types.is_object_dtype(df[col]):
            unique_ratio = df[col].nunique() / len(df) if len(df) else 0
            if unique_ratio > 0.5:
                is_id = True

        (excluded_cols if is_id else valid_cols).append(col)

    numeric_cols = [c for c in valid_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in valid_cols if c not in numeric_cols]
    return valid_cols, numeric_cols, categorical_cols, excluded_cols


def detect_target(df: pd.DataFrame, valid_cols: list[str]) -> str | None:
    """Heuristic: a boolean 2-value column, or a name matching target keywords."""
    # 1) Name match wins (most explicit signal).
    for col in valid_cols:
        if any(kw in col.lower() for kw in _TARGET_KEYWORDS):
            return col
    # 2) Otherwise the first boolean / 2-unique-value column.
    for col in valid_cols:
        series = df[col]
        if pd.api.types.is_bool_dtype(series) or series.nunique(dropna=True) == 2:
            return col
    return None


# ---------------------------------------------------------------------------
# Association matrix + truth table
# ---------------------------------------------------------------------------

def _pair_type(a_is_num: bool, b_is_num: bool) -> str:
    if a_is_num and b_is_num:
        return "Num×Num (Spearman)"
    if not a_is_num and not b_is_num:
        return "Cat×Cat (Cramér's V)"
    return "Mixed (Eta)"


def build_association(data_frame: pd.DataFrame) -> dict:
    """Compute the association matrix + markdown truth table for a cleaned df."""
    df = data_frame.drop(columns=list(_HELPER_COLS), errors="ignore")
    valid_cols, numeric_cols, categorical_cols, excluded_cols = classify_columns(df)
    target_col = detect_target(df, valid_cols)

    n = len(valid_cols)
    matrix = pd.DataFrame(np.zeros((n, n)), index=valid_cols, columns=valid_cols)

    for i in range(n):
        for j in range(i, n):
            if i == j:
                matrix.iloc[i, j] = 1.0
                continue

            col_a_name, col_b_name = valid_cols[i], valid_cols[j]
            a_is_num = col_a_name in numeric_cols
            b_is_num = col_b_name in numeric_cols
            col_a, col_b = df[col_a_name], df[col_b_name]

            try:
                if a_is_num and b_is_num:
                    score = abs(col_a.corr(col_b, method="spearman"))
                elif not a_is_num and not b_is_num:
                    score = cramers_v(col_a, col_b)
                else:
                    score = eta_squared(col_b, col_a) if a_is_num else eta_squared(col_a, col_b)
            except Exception:
                score = 0.0

            score = round(score, 4) if not (score is None or np.isnan(score)) else 0.0
            matrix.iloc[i, j] = score
            matrix.iloc[j, i] = score

    pairs: list[tuple[str, str, float, str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            score = matrix.iloc[i, j]
            if score >= TRUTH_TABLE_THRESHOLD:
                a, b = valid_cols[i], valid_cols[j]
                pairs.append((a, b, score, _pair_type(a in numeric_cols, b in numeric_cols)))
    pairs.sort(key=lambda x: x[2], reverse=True)

    return {
        "matrix": matrix,
        "truth_table_md": _render_truth_table(pairs, n, target_col),
        "valid_cols": valid_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "excluded_cols": excluded_cols,
        "target_col": target_col,
        "pairs": pairs,
    }


def _render_truth_table(pairs: list[tuple], n: int, target_col: str | None) -> str:
    lines = [
        "# Association Truth Table",
        "",
        f"**Threshold:** |association| >= {TRUTH_TABLE_THRESHOLD}",
        "**Methods:** Spearman (Num×Num), Cramér's V (Cat×Cat), Eta (Mixed)",
        f"**Detected TARGET:** {target_col or 'None'}",
        "",
        "## Significant Pairs (upper triangle only)",
        "",
        "| Column A | Column B | Score | Type |",
        "|----------|----------|-------|------|",
    ]
    for a, b, score, pair_type in pairs:
        lines.append(f"| `{a}` | `{b}` | {score:.4f} | {pair_type} |")
    lines += [
        "",
        f"**Total significant pairs:** {len(pairs)}",
        f"**Total valid columns:** {n}",
        f"**Upper-triangle pairs evaluated:** {n * (n - 1) // 2}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File output (no Redis for now — downloadable via dev endpoint)
# ---------------------------------------------------------------------------

def _job_dir(job_id: str) -> str:
    path = os.path.join(tempfile.gettempdir(), "eda", job_id)
    os.makedirs(path, exist_ok=True)
    return path


def multivariate_path(job_id: str) -> str:
    return os.path.join(_job_dir(job_id), "multivariate.json")


def truth_table_path(job_id: str) -> str:
    return os.path.join(_job_dir(job_id), "truth_table.md")


def write_json_file(job_id: str, items: list[dict]) -> str:
    path = multivariate_path(job_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    return path


def write_truth_table(job_id: str, truth_table_md: str) -> str:
    path = truth_table_path(job_id)
    with open(path, "w", encoding="utf-8") as f:
        f.write(truth_table_md)
    return path
