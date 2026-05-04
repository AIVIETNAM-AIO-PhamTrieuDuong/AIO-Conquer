## 3. Exhaustive Column-Level QA Profiling
*Goal: Define the exact query role, semantic meaning, and safety constraints for EVERY feature in the dataset.*
*LLM Directive: You MUST repeat the following bulleted structure for EVERY column present in the statistical summary.*

* **Column:** `[Insert Column Name]`
    * **Query Role:** [Infer role: 'Identifier' / 'Categorical Dimension' / 'Continuous Metric' / 'Temporal (Date/Time)' / 'Spatial' / 'Free Text']
    * **Business Context:** [Infer what this column represents in the context of the dataset's overarching domain]
    * **Null/NaN Handling Rule:** [Define exact SQL/Pandas treatment: e.g., 'COALESCE to 0', 'Filter out', 'Leave as NULL']
    * **Aggregation Safety:** [State 'SAFE for SUM/AVG' OR 'UNSAFE. Contains [X]% outliers. MUST apply `WHERE [Column] < [Threshold]` before aggregation']
    * **Grouping Directive:** [State 'Ideal for GROUP BY' OR 'DO NOT use for GROUP BY (High Cardinality: [X] unique values). Use for exact WHERE lookups only']

## 4. Temporal & Entity Relationships
*Goal: Identify primary axes for time-series and entity-based user queries.*

* **Primary Time Axis:** * [Identify the main Date/Time column, if any. Format: "Use `[Column Name]` for all 'When', 'Daily', 'Monthly' trend queries. Granularity is [Day/Month/Year]"].
* **Primary Entity/Granularity:** * [Identify what one row represents. Format: "One row represents a unique [Entity, e.g., Transaction, User Session, Employee]. COUNT(*) translates to 'Total [Entity]'"].

## 5. Domain Semantic Dictionary
*Goal: Map implicit natural language questions to explicit dataset columns and conditions.*
*LLM Directive: Infer 3-5 common business terms users might ask based on this specific schema.*

* **Semantic Mapping 1:** * [Term: e.g., 'Active Users' / 'High Value Transactions'] -> [SQL/Pandas Logic: e.g., `WHERE [Column A] = 'Yes' AND [Column B] > 100`]
* **Semantic Mapping 2:** * [Term] -> [Logic]
* **Semantic Mapping 3:** * [Term] -> [Logic]

## 6. Global Execution & Query Caveats
*Goal: Global safety rules for the Text-to-SQL/Pandas generation node.*

* **Cross-Column Conflicts:** * [Identify logical impossibilities. Format: "Ensure queries NEVER combination [Condition A] with [Condition B] as they are mutually exclusive"].
* **Metric Calculation Rules:** * [Infer 1-2 core calculated metrics. Format: "To calculate [Derived Metric, e.g., Conversion Rate], strictly use: `[Math Formula combining Column X and Y]`"].
* **Default Pre-filtering:** * [Identify if any global filter should apply to ALL queries. Format: "Always apply `WHERE [Column] != [Junk Value]` unless the user explicitly asks for raw data"].