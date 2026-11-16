# Online Retail Sales Analysis & Customer Segmentation

**Python · Pandas · NumPy · Matplotlib · Seaborn · Plotly · Dash · RFM Analysis**

An end-to-end exploratory data analysis of an online retailer's transactions, focused on understanding **sales performance, customer behavior, product demand, and geographic trends**. The project moves from data-quality checks and transaction-level feature engineering to customer segmentation, business insights, and a dashboard.

> **Dataset:** [UCI Machine Learning Repository — Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail). The transactions cover **December 2010–December 2011** and belong to a UK-based, non-store retailer. This is an independent analysis of a public dataset, not a client engagement.

## Business questions

- How much merchandise revenue does the business generate, and how does it change over time?
- Which products contribute the most **revenue**, and which sell the most **units**?
- Which countries contribute the most orders and sales?
- Which customers are high-value, loyal, recently active, or at risk of becoming inactive?
- Which products tend to appear together on the same invoice?
- How should unusual transactions and wholesale-sized orders be interpreted?

## What the notebook covers

| Stage | Work performed |
| --- | --- |
| Data understanding | Review transaction structure, columns, coverage, and data types |
| Data quality | Inspect duplicate rows, missing customer IDs, cancellations, invalid sales values, and non-merchandise items |
| Cleaning | Preserve an audited dataset (`df_clean`) and create a positive-merchandise-sales dataset (`df_sales`) |
| Feature engineering | Calculate `Revenue = Quantity × UnitPrice` and extract time attributes from invoice dates |
| Customer analysis | Aggregate identified-customer order frequency, spending, recency, and average order value |
| RFM segmentation | Score customers by **Recency, Frequency, Monetary value** and assign transparent, rule-based segments |
| Product analysis | Compare top products by sales revenue and quantity and measure revenue concentration |
| Time and geography | Explore monthly, weekday, hourly, and country-level patterns |
| Product relationships | Examine frequently co-purchased products using invoice-level combinations |
| Outlier review | Flag extreme quantities, prices, and order values for investigation rather than deleting them automatically |
| Reporting | Generate findings, an executive summary, dashboard visuals, and reusable CSV tables |

## Key metrics

- **Revenue:** sum of positive merchandise sales after the notebook's cleaning rules.
- **Orders:** number of distinct `InvoiceNo` values.
- **Customers:** number of identifiable customers (`CustomerID`).
- **Average order value:** revenue divided by distinct orders.
- **RFM metrics:** recency in days, distinct purchase frequency, and monetary value.

**Important:** The analysis focuses on **positive merchandise sales**, not gross profit or net revenue after returns. Customers with missing IDs remain in general sales analysis but cannot be individually segmented.

## Dashboard

The notebook has a Matplotlib sales overview. An **optional interactive Dash + Plotly dashboard** can replace its final dashboard cell, with a custom **Espresso / Terracotta / Olive / Dusty Rose** theme rather than the usual blue/purple palette.

The interactive version supports:

- **Filters:** country, year, and number of leading products to display.
- **Live KPIs:** total revenue, number of orders, unique customers, and average order value.
- **Charts:** monthly revenue, best-selling products by revenue, leading countries, customer segments, and a weekday-by-hour revenue heatmap.

**Note:** GitHub displays a notebook preview but does not host a live Dash application. Run the notebook locally to use the interactive filters. Add the separate Dash dashboard code to the notebook before describing the interactive version as part of the published notebook.

## Dataset and setup

1. Download the **Online Retail** dataset from [UCI](https://archive.ics.uci.edu/dataset/352/online+retail).
2. Put the data beside the notebook using **one** of these supported filenames:
   - `online_retail.csv.gz`
   - `online_retail.csv`
   - `Online Retail.xlsx`
3. Install dependencies:

   ```bash
   python -m pip install jupyter pandas numpy matplotlib seaborn plotly missingno openpyxl dash
   ```

4. Start Jupyter Notebook:

   ```bash
   jupyter notebook "Online_Retail_Sales_Questions_Blank_Fixed.ipynb"
   ```

5. Choose **Run All**. If you added the Dash dashboard cell, execute it after the analysis. The supplied interactive version uses port **8050**.

## Project files

```text
Online_Retail_Sales/
├── README.md
├── Online_Retail_Sales_Questions_Blank_Fixed.ipynb
├── online_retail.csv.gz                 # Download separately; one supported data format
└── outputs/                             # Generated when the notebook is run
    ├── retail_clean_sales.csv
    ├── retail_customer_summary.csv
    ├── retail_rfm_segments.csv
    └── retail_country_summary.csv
```

> The file tree shows a recommended repository structure. The source dataset and generated outputs are not included in this README download.

## Analytical limitations

- The dataset covers **one retailer** over a historical period, so results are not automatically generalizable.
- **Revenue is not profit**: product costs, shipping costs, and margins are unavailable.
- Missing `CustomerID` values limit customer-level insights.
- Partial months at the start and end of the dataset affect comparisons.
- Co-purchases and correlations do not establish causation.
- Cancelled transactions and returns are excluded from the positive-sales KPIs.

## Data credit

**UCI Machine Learning Repository — Online Retail**, originally contributed by Daqing Chen. [Dataset page](https://archive.ics.uci.edu/dataset/352/online+retail).
