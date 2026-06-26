"""Prompt builder for the Semantic Multivariate Potential Analysis step.

Takes the two EDA artifacts (Dataset Profile + Correlation Truth Table) and asks
the LLM to emit a Multivariate Use-Case Dictionary as a single raw JSON array.
"""
from __future__ import annotations

_PROMPT_TEMPLATE = """\
Please perform a comprehensive Semantic Multivariate Potential Analysis based on the TWO attached documents:
1. Dataset Profile — column metadata, types, and business context.
2. Correlation Truth Table — pre-computed correlation pairs (Spearman/Cramér's V/Eta).

Focus exclusively on executing the core analytical logic to output a production-ready Multivariate Use-Case Dictionary strictly formatted as a single raw JSON array.

DETECTED TARGET VARIABLE: {target_col}

---

### PAIR EVALUATION LOGIC (INTERNAL PROCESSING)
Before generating the JSON payload, evaluate pairs using these constraints:
1. Exclude high-cardinality identifiers, completely empty columns, or severe noise/constant columns.
2. Group remaining columns mentally by Query Role (Temporal, Categorical/Spatial, Continuous Metrics, Boolean Target). Count the valid columns as N.
3. NON-NEGOTIABLE RULE: If a TARGET variable exists, it MUST be paired with EVERY valid non-TARGET column. Expected TARGET pairs = N - 1. Do NOT skip any.
4. Evaluate every valid upper-triangular pair and assign a Confidence Score (1-5):
   - Score 4-5 (Critical KPI): Essential for core business monitoring (e.g., Target × Any Feature, or Core Category × Primary Metric). Target pairs CANNOT score below 4.
   - Score 2-3 (Secondary Insight): Useful for EDA or deep-dive analysis.
   - Score 0: Irrelevant / structural duplicates / multi-collinear noise.

---

### OUTPUT FORMATTING & REQUIREMENTS
Return ONLY a raw, valid JSON array string containing the valid pairs (Score > 0) sorted in descending order by their Confidence Score. Do NOT wrap it in markdown code blocks (```json ... ```). Do not include any conversational text, introduction, or wrap-up filler.

#### SymPy Integration Rule:
For the sympy_calculation_code field, you must provide executable Python code using the sympy library.
- Define symbols using sympy.symbols or sympy.Symbol.
- Symbolically represent the core metric calculation formula, incorporating any filters or constraints.
- Express logic explicitly using piecewise functions or relational symbols.

*Templates by Pair Type:*
- TARGET × CATEGORICAL: Piecewise((attrition_rate('Yes'), Eq(overtime_flag, 1)), ...)
- TARGET × METRIC: Piecewise((avg_income('attrited'), Eq(attrition_flag, 1)), ...)
- METRIC × CATEGORICAL: avg_income(job_role)

#### JSON Schema per Object:
[
  {{
    "comparison_pair": {{
      "variable_a": "string (exact column name from profile)",
      "variable_b": "string (exact column name from profile)"
    }},
    "evaluation": {{
      "confidence_score": "integer (1-5 scale)",
      "proposed_analysis_metric": "string (clear title of the analytical use-case)"
    }},
    "business_value": "string (detailed explanation of why this matters to stakeholders and what it solves)",
    "sympy_calculation_code": "string (valid, executable Python SymPy code block outlining the symbolic expression and conditional logic/constraints)",
    "metrics_and_significance": {{
      "statistical_test_type": "string (e.g., Pearson, Chi-Square, ANOVA, T-Test, Logistic Regression based on variable roles)",
      "expected_metrics": ["list", "of", "target", "metrics", "to", "measure"]
    }},
    "interpretation_instructions": "string (numbered step-by-step instructions clarifying how data teams must isolate variables, apply filters, and interpret the resulting correlation/metrics)"
  }}
]

---

=== DOCUMENT 1: DATASET PROFILE ===
{dataset_profile_md}
=== END DOCUMENT 1 ===

=== DOCUMENT 2: CORRELATION TRUTH TABLE ===
{truth_table_md}
=== END DOCUMENT 2 ===
"""


def build_multivariate_prompt(
    dataset_profile_md: str,
    truth_table_md: str,
    target_col: str | None,
) -> str:
    return _PROMPT_TEMPLATE.format(
        target_col=target_col or "None (no clear target detected)",
        dataset_profile_md=dataset_profile_md,
        truth_table_md=truth_table_md,
    )
