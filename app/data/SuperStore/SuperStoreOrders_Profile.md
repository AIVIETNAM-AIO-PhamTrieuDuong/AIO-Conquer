# SuperStoreOrders Dataset Profile

## 3. Exhaustive Column-Level QA Profiling

* **Column:** `order_id`
    * **Query Role:** Identifier
    * **Business Context:** The unique alphanumeric tracking code assigned to a customer's purchase event. Multiple rows can share the same `order_id` if multiple products were bought together.
    * **Null/NaN Handling Rule:** No nulls detected.
    * **Aggregation Safety:** N/A (String).
    * **Grouping Directive:** DO NOT use for GROUP BY (High Cardinality: 25,035 unique values). Use for exact WHERE lookups or `COUNT(DISTINCT order_id)` to calculate total unique orders.

* **Column:** `order_date`
    * **Query Role:** Temporal (Date/Time)
    * **Business Context:** The calendar date when the customer placed the order.
    * **Null/NaN Handling Rule:** No nulls detected. 
    * **Aggregation Safety:** N/A (Temporal).
    * **Grouping Directive:** Ideal for GROUP BY (Daily/Monthly trends). Note: Format is stored as a String (DD/MM/YYYY or D/M/YYYY) and MUST be cast to Date type before temporal sorting or date-math.

* **Column:** `ship_date`
    * **Query Role:** Temporal (Date/Time)
    * **Business Context:** The calendar date when the physical product was dispatched to the customer.
    * **Null/NaN Handling Rule:** No nulls.
    * **Aggregation Safety:** N/A.
    * **Grouping Directive:** Can be used for GROUP BY to track fulfillment throughput, but `order_date` is preferred for revenue queries.

* **Column:** `ship_mode`
    * **Query Role:** Categorical Dimension
    * **Business Context:** The shipping speed/tier selected for the order line (e.g., Standard Class, First Class).
    * **Null/NaN Handling Rule:** No nulls.
    * **Aggregation Safety:** N/A.
    * **Grouping Directive:** Ideal for GROUP BY (4 unique values).

* **Column:** `customer_name`
    * **Query Role:** Categorical Dimension / Free Text
    * **Business Context:** The full name of the customer who made the purchase. 
    * **Null/NaN Handling Rule:** No nulls.
    * **Aggregation Safety:** N/A.
    * **Grouping Directive:** DO NOT use for GROUP BY (High Cardinality: 795 unique values). Use for specific text matching (`LIKE '%Name%'`) or top 10 rankings (`LIMIT 10`).

* **Column:** `segment`
    * **Query Role:** Categorical Dimension
    * **Business Context:** The business classification of the customer (Consumer, Corporate, Home Office).
    * **Null/NaN Handling Rule:** No nulls.
    * **Aggregation Safety:** N/A.
    * **Grouping Directive:** Ideal for GROUP BY (3 unique values). 

* **Column:** `state`
    * **Query Role:** Spatial / Categorical Dimension
    * **Business Context:** The province or state where the order is being shipped.
    * **Null/NaN Handling Rule:** No nulls.
    * **Aggregation Safety:** N/A.
    * **Grouping Directive:** DO NOT use for GROUP BY globally without pairing with `country`, as state names may overlap across countries (1,094 unique values). 

* **Column:** `country`
    * **Query Role:** Spatial / Categorical Dimension
    * **Business Context:** The nation where the order is being shipped.
    * **Null/NaN Handling Rule:** No nulls.
    * **Aggregation Safety:** N/A.
    * **Grouping Directive:** Ideal for GROUP BY (147 unique values). Highly recommended for geographic segmentation.

* **Column:** `market`
    * **Query Role:** Spatial / Categorical Dimension
    * **Business Context:** The macro-level global market region (e.g., APAC, LATAM, EU).
    * **Null/NaN Handling Rule:** No nulls.
    * **Aggregation Safety:** N/A.
    * **Grouping Directive:** Ideal for GROUP BY (7 unique values).

* **Column:** `region`
    * **Query Role:** Spatial / Categorical Dimension
    * **Business Context:** A geographic sub-division within a given `market` or `country` (e.g., Central, South, EMEA).
    * **Null/NaN Handling Rule:** No nulls.
    * **Aggregation Safety:** N/A.
    * **Grouping Directive:** Ideal for GROUP BY (13 unique values).

* **Column:** `product_id`
    * **Query Role:** Identifier
    * **Business Context:** The internal SKU or unique catalog tracking code for a specific product.
    * **Null/NaN Handling Rule:** No nulls.
    * **Aggregation Safety:** N/A.
    * **Grouping Directive:** DO NOT use for GROUP BY generally (High Cardinality: 10,292 unique values), unless generating product-level performance tables.

* **Column:** `category`
    * **Query Role:** Categorical Dimension
    * **Business Context:** The highest-level taxonomy of the product sold (Office Supplies, Technology, Furniture).
    * **Null/NaN Handling Rule:** No nulls.
    * **Aggregation Safety:** N/A.
    * **Grouping Directive:** Ideal for GROUP BY (3 unique values).

* **Column:** `sub_category`
    * **Query Role:** Categorical Dimension
    * **Business Context:** The secondary taxonomy grouping of the product (e.g., Binders, Storage, Art).
    * **Null/NaN Handling Rule:** No nulls.
    * **Aggregation Safety:** N/A.
    * **Grouping Directive:** Ideal for GROUP BY (17 unique values).

* **Column:** `product_name`
    * **Query Role:** Free Text / Categorical
    * **Business Context:** The literal, human-readable name and description of the product.
    * **Null/NaN Handling Rule:** No nulls.
    * **Aggregation Safety:** N/A.
    * **Grouping Directive:** DO NOT use for GROUP BY (High Cardinality: 3,788 unique values). Use for keyword searching.

* **Column:** `sales`
    * **Query Role:** Continuous Metric
    * **Business Context:** The gross revenue generated from the specific order line item.
    * **Null/NaN Handling Rule:** No nulls.
    * **Aggregation Safety:** UNSAFE (Typing constraint). Currently formatted as an `object/string` in the raw data (may contain commas e.g., "1,200"). MUST apply `CAST(REPLACE(sales, ',', '') AS FLOAT)` before SUM/AVG. Contains ~11.03% high-value outliers. 
    * **Grouping Directive:** DO NOT use for GROUP BY. 

* **Column:** `quantity`
    * **Query Role:** Continuous Metric
    * **Business Context:** The number of individual units of the specific product purchased in this order line.
    * **Null/NaN Handling Rule:** No nulls.
    * **Aggregation Safety:** SAFE for SUM/AVG. Contains ~1.71% minor outliers (Max is 14 units).
    * **Grouping Directive:** DO NOT use for GROUP BY.

* **Column:** `discount`
    * **Query Role:** Continuous Metric
    * **Business Context:** The percentage discount applied to the item, represented as a decimal (e.g., 0.20 = 20%).
    * **Null/NaN Handling Rule:** No nulls. Leave as 0.0 when no discount is applied.
    * **Aggregation Safety:** SAFE for AVG. Do NOT use SUM(discount) as it makes no mathematical sense. Contains ~8.13% outliers.
    * **Grouping Directive:** Can be used for GROUP BY to see "Sales across different discount tiers", but generally avoid.

* **Column:** `profit`
    * **Query Role:** Continuous Metric
    * **Business Context:** The net profit (or loss) realized from the sale of this order line item, expressed in local currency.
    * **Null/NaN Handling Rule:** No nulls.
    * **Aggregation Safety:** UNSAFE. Highly variable. Contains ~12.15% extreme outliers ranging from severe negative losses (-6,599) to massive gains (+8,399). Applying `WHERE profit BETWEEN -1000 AND 1000` may be required for stable averages.
    * **Grouping Directive:** DO NOT use for GROUP BY.

* **Column:** `shipping_cost`
    * **Query Role:** Continuous Metric
    * **Business Context:** The monetary cost required to physically transport the order.
    * **Null/NaN Handling Rule:** No nulls.
    * **Aggregation Safety:** UNSAFE. Contains ~11.52% outliers (Max value 933.57). 
    * **Grouping Directive:** DO NOT use for GROUP BY.

* **Column:** `order_priority`
    * **Query Role:** Categorical Dimension
    * **Business Context:** The urgency flag assigned to the order handling process (e.g., Critical, High, Medium, Low).
    * **Null/NaN Handling Rule:** No nulls.
    * **Aggregation Safety:** N/A.
    * **Grouping Directive:** Ideal for GROUP BY (4 unique values).

* **Column:** `year`
    * **Query Role:** Temporal (Date/Time) / Categorical
    * **Business Context:** The pre-extracted year of the order, ranging from 2011 to 2014.
    * **Null/NaN Handling Rule:** No nulls.
    * **Aggregation Safety:** N/A (Numeric, but acts as temporal category).
    * **Grouping Directive:** Ideal for GROUP BY (4 unique values). Use for high-level year-over-year (YoY) analysis.

## 4. Temporal & Entity Relationships

* **Primary Time Axis:** * Use `order_date` for all 'When', 'Daily', 'Monthly' sales and profit trend queries. Granularity is Day. Note that you can use the pre-extracted `year` column for faster YoY grouping.
* **Primary Entity/Granularity:** * One row represents a unique **Order Line Item** (a specific product within an overarching order). `COUNT(*)` translates to 'Total Items Sold'. To find 'Total Orders', you MUST use `COUNT(DISTINCT order_id)`.

## 5. Domain Semantic Dictionary

* **Semantic Mapping 1:** * 'Total Revenue' / 'Total Sales' -> `SUM(CAST(REPLACE(sales, ',', '') AS FLOAT))`
* **Semantic Mapping 2:** * 'Profitable Transactions' -> `WHERE profit > 0`
* **Semantic Mapping 3:** * 'Loss-making / Unprofitable Sales' -> `WHERE profit < 0`
* **Semantic Mapping 4:** * 'B2B (Business-to-Business) Customers' -> `WHERE segment IN ('Corporate', 'Home Office')`
* **Semantic Mapping 5:** * 'Expedited / Fast Shipping' -> `WHERE ship_mode IN ('First Class', 'Same Day')`

## 6. Global Execution & Query Caveats

* **Cross-Column Conflicts:** * Ensure queries NEVER aggregate `sales` without first cleaning the string format (removing commas and casting to numeric). The `sales` column is typed as a string object, and a raw `SUM(sales)` will fail in most SQL engines or Pandas scripts.
* **Metric Calculation Rules:** * To calculate `Profit Margin`, strictly use: `SUM(profit) / SUM(CAST(REPLACE(sales, ',', '') AS FLOAT))`. Do NOT calculate the average of the `profit` column directly as a representation of overall margin health, as transaction sizes vary wildly.
* **Default Pre-filtering:** * No junk pre-filters required, but whenever looking at "Fulfillment Time" or "Days to Ship", the global formula is `DATEDIFF(day, CAST(order_date AS DATE), CAST(ship_date AS DATE))`. Beware of DD/MM/YYYY formatting conflicts based on the SQL dialect used.
