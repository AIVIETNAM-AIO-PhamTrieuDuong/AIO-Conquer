# EDA Workflow Template

This document describes the automated, domain-agnostic Exploratory Data Analysis (EDA) steps executed by the Conquer QA system for every uploaded dataset.

## Step 1: Duplicate Handling
- The system scans the entire dataset and drops all exact duplicate rows.
- The number of removed duplicates is recorded and reported in the data profile.

## Step 2: Automatic Date Detection
- The system scans all `object`-type columns for date/time patterns.
- Columns whose names contain keywords like `date`, `time`, or `timestamp` are tested with `pd.to_datetime()`.
- Confirmed date columns are separated from the categorical pool and reported in the data profile.

## Step 3: Column Type Separation
The system automatically identifies and separates all features into three groups:
- **Numerical:** Columns containing numeric data types (int, float).
- **Categorical:** Columns containing strings (object), booleans, or category types — **excluding** detected date columns.
- **Temporal:** Date/time columns identified in Step 2.

## Step 4: Missing Values Imputation
- **Numerical columns:** All Null/NaN values are filled with the column **mean**.
- **Categorical columns:** All Null/NaN values are filled with the column **mode**. If no clear mode exists, the value `"Unknown"` is used.
- **Date columns:** All Null/NaN values are filled with the column **mode** (most frequent date).
- The count of nulls fixed per column is recorded for reporting.

## Step 5: Outlier Detection & Flagging
Uses the **Interquartile Range (IQR)** method, applied only to numerical columns:
- Compute the first quartile ($Q1$) and third quartile ($Q3$).
- $IQR = Q3 - Q1$
- A value is flagged as an outlier if it falls below $(Q1 - 1.5 \times IQR)$ or above $(Q3 + 1.5 \times IQR)$.
- **Data integrity is preserved:** Outlier rows are NOT dropped. Instead, a metadata column `outlier_col_name` is appended to the DataFrame, storing a list of the column names responsible for the anomaly in each row.

## Step 6: Statistical Profiling
- **Numerical Profile:** Extracts Mean, Standard Deviation (Std), Min, Max, unique count, outlier count, outlier percentage, **Skewness**, and **Kurtosis**. A `⚠️ HIGH ANOMALY RATE` warning is triggered if outlier rate exceeds 5%. Skewness and Kurtosis values are displayed when |Skewness| > 1 to flag heavily skewed distributions.
- **Categorical Profile:** Extracts Mode, unique count, and Top 5 most frequent values with their occurrence counts.
- **Dataset-level Metadata:** Total flagged rows, total rows, and total column count.

## Step 7: High Correlation Detection
- The system computes the absolute Pearson correlation matrix across all numerical columns.
- Any pair of columns with correlation > **0.85** is flagged as a `🚩 Correlation Alert`.
- This alerts the downstream QA Agent to avoid using both columns in the same aggregation or model to prevent multicollinearity issues.

## Step 8: LLM-Powered Semantic Profiling
- All statistical metadata from Steps 6-7 is formatted into a structured Markdown report.
- This report is passed to an LLM (Gemini) using the `LLM_Summary_Template` as a strict output schema.
- The LLM infers for **every single column**: Query Role, Business Context, Null/NaN Handling Rule, Aggregation Safety, and Grouping Directive.
- The LLM also receives date detection results and correlation alerts to produce more accurate temporal and relationship insights.
- This semantic layer is the foundation that prevents the downstream QA Agent from generating hallucinated or incorrect queries.
