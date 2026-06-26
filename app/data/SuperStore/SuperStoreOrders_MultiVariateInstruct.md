# Semantic Multivariate Potential Analysis — SuperStoreOrders

This semantic analysis evaluates the `SuperStoreOrders` dataset's column-level attributes based on the provided profiling data. Strict business logic is applied to eliminate noise, handle data types correctly, and identify highly actionable multivariate relationships.

---

### STEP 1 — Profile Ingestion & Feature Classification

### Excluded Identifiers & Free Text
* **`order_id`**: A unique alphanumeric tracking code (25,035 unique values). Excluded because it provides no grouping or aggregation value beyond exact WHERE lookups.
* **`customer_name`**: Free text full name of the customer (795 unique values). Excluded because high cardinality makes it unsuitable for group-by analysis.
* **`product_id`**: Internal SKU tracking code (10,292 unique values). Excluded due to extremely high cardinality.
* **`product_name`**: Free text human-readable name of the product (3,788 unique values). Excluded due to extremely high cardinality.

### Filtered Columns for Analysis Matrix

**1. Temporal Dimensions** (Dates/Times)
* **`order_date`**: Calendar date the customer placed the order. Stored as a string and must be cast to Date type before use.
* **`ship_date`**: Calendar date when the physical product was dispatched to the customer.
* **`year`**: Pre-extracted temporal category (4 unique values: 2011 to 2014).

**2. Categorical & Spatial Dimensions** (Booleans, Categories, Locations)
* **`ship_mode`**: Shipping speed/tier selected (4 unique values).
* **`segment`**: Business classification of the customer (3 unique values: Consumer, Corporate, Home Office).
* **`state`**: Province or state for shipping (1,094 unique values). Spatial dimension. Not globally unique on its own.
* **`country`**: Nation for shipping (147 unique values). Spatial dimension.
* **`market`**: Macro-level global market region (7 unique values). Spatial dimension.
* **`region`**: Geographic sub-division within a market or country (13 unique values). Spatial dimension.
* **`category`**: Highest-level product taxonomy (3 unique values).
* **`sub_category`**: Secondary product taxonomy grouping (17 unique values).
* **`order_priority`**: Urgency flag for the order handling process (4 unique values).

**3. Continuous Metrics** (Aggregation-safe numeric values)
* **`sales`**: Gross revenue. Requires string-cleaning (`CAST(REPLACE(sales, ',', '') AS FLOAT)`) before aggregation. Contains outliers.
* **`quantity`**: Number of units purchased. Safe for aggregation.
* **`discount`**: Percentage discount represented as a decimal. Avoid SUM, use AVG cautiously.
* **`profit`**: Net profit or loss. Highly variable with extreme outliers.
* **`shipping_cost`**: Monetary cost to transport the order. Contains outliers.

---

### STEP 2 — Multivariate Potential Matrix with Confidence Scoring

| ID | Feature | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | C11 | C12 | C13 | C14 | C15 | C16 | C17 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | order_date | - | [[order_date]], [[ship_date]] | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C2 | ship_date | 0 | - | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C3 | year | 0 | 0 | - | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | [[year]], [[sales]] | 0 | 0 | [[year]], [[profit]] | 0 |
| C4 | ship_mode | 0 | 0 | 0 | - | 0 | 0 | 0 | 0 | 0 | 0 | 0 | [[ship_mode]], [[order_priority]] | 0 | 0 | 0 | 0 | [[ship_mode]], [[shipping_cost]] |
| C5 | segment | 0 | 0 | 0 | 0 | - | 0 | 0 | [[segment]], [[market]] | 0 | [[segment]], [[category]] | 0 | 0 | [[segment]], [[sales]] | 0 | 0 | [[segment]], [[profit]] | 0 |
| C6 | state | 0 | 0 | 0 | 0 | 0 | - | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C7 | country | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 | 0 | [[country]], [[category]] | 0 | 0 | [[country]], [[sales]] | 0 | 0 | [[country]], [[profit]] | 0 |
| C8 | market | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 | 0 | 0 | 0 | [[market]], [[sales]] | 0 | 0 | [[market]], [[profit]] | 0 |
| C9 | region | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | [[region]], [[category]] | 0 | 0 | [[region]], [[sales]] | 0 | 0 | [[region]], [[profit]] | 0 |
| C10 | category | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 | 0 | [[category]], [[sales]] | 0 | [[category]], [[discount]] | [[category]], [[profit]] | 0 |
| C11 | sub_category | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 | [[sub_category]], [[sales]] | 0 | 0 | [[sub_category]], [[profit]] | 0 |
| C12 | order_priority | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 | 0 | 0 | 0 | [[order_priority]], [[shipping_cost]] |
| C13 | sales | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 | [[sales]], [[discount]] | [[sales]], [[profit]] | 0 |
| C14 | quantity | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 | 0 | 0 |
| C15 | discount | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | [[discount]], [[profit]] | 0 |
| C16 | profit | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 |
| C17 | shipping_cost | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - |

---

### STEP 3 — Multivariate Use-Case Dictionary

| Column Pair | Score | Proposed Analysis / Metric | Business Value | Calculation Method |
|---|---|---|---|---|
| `[[order_date]], [[ship_date]]` | 5 | Fulfillment Time Tracking | Primary KPI for operational efficiency, measuring the exact lag between order placement and dispatch. | 1. `CAST(order_date AS DATE)` and `CAST(ship_date AS DATE)`.<br>2. Calculate `DATEDIFF(day, order_date, ship_date)`.<br>3. Average the result to find typical fulfillment speed. |
| `[[ship_mode]], [[shipping_cost]]` | 5 | Shipping Tier Cost Impact | Core business monitoring to evaluate if expedited shipping tiers are exponentially increasing logistics spend. | 1. `GROUP BY ship_mode`.<br>2. Calculate `AVG(shipping_cost)`.<br>3. Handle outliers if necessary. |
| `[[segment]], [[category]]` | 5 | Customer Segment Affinity | Strict Dimension x Dimension mapping to reveal which customer types are the primary buyers of specific product categories. | 1. `GROUP BY segment, category`.<br>2. Calculate `COUNT(DISTINCT order_id)` to measure order volume per segment-category pair. |
| `[[segment]], [[sales]]` | 5 | Revenue by Customer Segment | Primary Metric x Core Category to identify which business segment generates the most gross revenue. | 1. `GROUP BY segment`.<br>2. `CAST(REPLACE(sales, ',', '') AS FLOAT)`.<br>3. Calculate `SUM(sales)`. |
| `[[segment]], [[profit]]` | 5 | Profitability by Segment | Evaluates which customer classification yields the highest net returns. | 1. `GROUP BY segment`.<br>2. Calculate `SUM(profit)`. |
| `[[country]], [[category]]` | 5 | Geographic Category Demand | Spatial Dimension x Dimension mapping to reveal macro-level product preferences across different nations. | 1. `GROUP BY country, category`.<br>2. Calculate `COUNT(DISTINCT order_id)` or `SUM(quantity)`. |
| `[[country]], [[sales]]` | 5 | Gross Revenue by Nation | Primary KPI identifying the highest revenue-generating geographies. | 1. `GROUP BY country`.<br>2. `CAST(REPLACE(sales, ',', '') AS FLOAT)`.<br>3. Calculate `SUM(sales)`. |
| `[[country]], [[profit]]` | 5 | Net Profit by Nation | Evaluates national-level profitability, exposing countries operating at a loss. | 1. `GROUP BY country`.<br>2. Calculate `SUM(profit)`. |
| `[[market]], [[sales]]` | 5 | Global Market Revenue | High-level executive KPI tracking gross revenue across macro markets (APAC, EU, etc.). | 1. `GROUP BY market`.<br>2. `CAST(REPLACE(sales, ',', '') AS FLOAT)`.<br>3. Calculate `SUM(sales)`. |
| `[[market]], [[profit]]` | 5 | Global Market Profitability | Identifies which global markets are sustaining the business versus draining resources. | 1. `GROUP BY market`.<br>2. Calculate `SUM(profit)`. |
| `[[region]], [[sales]]` | 5 | Regional Revenue Generation | Core spatial insight to guide localized sales strategies based on gross revenue. | 1. `GROUP BY region`.<br>2. `CAST(REPLACE(sales, ',', '') AS FLOAT)`.<br>3. Calculate `SUM(sales)`. |
| `[[region]], [[profit]]` | 5 | Regional Profit Margins | Evaluates the net financial health of localized regions. | 1. `GROUP BY region`.<br>2. Calculate `SUM(profit)`. |
| `[[category]], [[sales]]` | 5 | Revenue by Taxonomy | Essential for core business monitoring to see which high-level product tier drives top-line growth. | 1. `GROUP BY category`.<br>2. `CAST(REPLACE(sales, ',', '') AS FLOAT)`.<br>3. Calculate `SUM(sales)`. |
| `[[category]], [[profit]]` | 5 | Profitability by Taxonomy | Identifies if certain product categories are loss leaders or cash cows. | 1. `GROUP BY category`.<br>2. Calculate `SUM(profit)`. |
| `[[category]], [[discount]]` | 5 | Discount Strategy by Category | Tracks how aggressive discounting strategies are applied to different product tiers. | 1. `GROUP BY category`.<br>2. Calculate `AVG(discount)`. |
| `[[sales]], [[profit]]` | 5 | Overall Profit Margin | Critical Metric x Metric ratio that determines the true mathematical health of the business. | 1. `CAST(REPLACE(sales, ',', '') AS FLOAT)`.<br>2. Calculate Global Margin: `SUM(profit) / SUM(sales)`. |
| `[[discount]], [[profit]]` | 5 | Discount Impact on Net Profit | Metric x Metric correlation strictly measuring how percentage markdowns erode bottom-line returns. | 1. Filter out zero discounts: `WHERE discount > 0`.<br>2. Correlate `discount` tiers against `AVG(profit)`. |
| `[[year]], [[sales]]` | 4 | Year-over-Year Revenue Growth | Tracks top-line trajectory across the available 4-year span. | 1. `GROUP BY year`.<br>2. `CAST(REPLACE(sales, ',', '') AS FLOAT)`.<br>3. Calculate `SUM(sales)`.<br>4. Sort by year chronologically. |
| `[[year]], [[profit]]` | 4 | Year-over-Year Profit Growth | Tracks bottom-line trajectory to ensure scaling is healthy. | 1. `GROUP BY year`.<br>2. Calculate `SUM(profit)`.<br>3. Sort by year chronologically. |
| `[[ship_mode]], [[order_priority]]` | 4 | SLA Operational Alignment | Checks if highly critical orders are successfully paired with expedited shipping modes. | 1. `GROUP BY ship_mode, order_priority`.<br>2. Calculate `COUNT(*)` to verify mapping consistency. |
| `[[segment]], [[market]]` | 4 | B2B/B2C Market Distribution | Spatial cross-tabulation to see if certain global markets are dominated by Corporate vs Consumer buyers. | 1. `GROUP BY market, segment`.<br>2. Calculate `COUNT(DISTINCT order_id)`. |
| `[[region]], [[category]]` | 4 | Localized Product Demand | Spatial insight mapping specific regional preferences to high-level product taxonomies. | 1. `GROUP BY region, category`.<br>2. Calculate `SUM(quantity)`. |
| `[[sub_category]], [[sales]]` | 4 | Granular Revenue Drivers | Secondary insight identifying top-selling specific product lines (e.g., Phones, Chairs). | 1. `GROUP BY sub_category`.<br>2. `CAST(REPLACE(sales, ',', '') AS FLOAT)`.<br>3. Calculate `SUM(sales)`. |
| `[[sub_category]], [[profit]]` | 4 | Granular Profitability | Pinpoints specific product lines that are dragging down overall category margins. | 1. `GROUP BY sub_category`.<br>2. Calculate `SUM(profit)`. |
| `[[order_priority]], [[shipping_cost]]` | 4 | Cost of Urgency | Measures the logistical cost premium associated with fulfilling high-priority orders. | 1. `GROUP BY order_priority`.<br>2. Calculate `AVG(shipping_cost)`. |
| `[[sales]], [[discount]]` | 4 | Markdowns vs Revenue Volume | Metric x Metric ratio examining if higher discounts successfully stimulate larger gross sales volumes. | 1. `CAST(REPLACE(sales, ',', '') AS FLOAT)`.<br>2. Bin discounts (e.g., 0-10%, 10-30%).<br>3. Calculate `AVG(sales)` per bin. |