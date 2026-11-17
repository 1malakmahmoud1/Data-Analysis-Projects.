"""Exact standalone Dash version of the notebook's final dashboard.
Run: python retail_dashboard.py
Deploy with gunicorn retail_dashboard:server.
"""
from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "online_retail.csv.gz"
df_raw = pd.read_csv(DATA_PATH, compression="gzip")
df_raw["InvoiceDate"] = pd.to_datetime(df_raw["InvoiceDate"], errors="coerce")
non_product_codes = {"POST", "D", "M", "BANK CHARGES", "DOT", "C2", "CRUK", "PADS", "AMAZONFEE", "S"}

def clean_data(df):
    out = df.drop_duplicates().copy()
    out['IsCancelled'] = out['InvoiceNo'].astype(str).str.upper().str.startswith('C')
    out['IsNonProduct'] = out['StockCode'].astype(str).str.upper().isin(non_product_codes)
    out['IsZeroOrNegPrice'] = out['UnitPrice'] <= 0
    out['Description'] = out['Description'].fillna('Unknown')

    # Returns, negative adjustments and invalid dates are not positive merchandise sales.
    sales_mask = (
        ~out['IsCancelled'] & ~out['IsNonProduct'] & ~out['IsZeroOrNegPrice']
        & out['Quantity'].gt(0) & out['InvoiceDate'].notna()
        & out['InvoiceNo'].notna() & out['StockCode'].notna()
    )
    df_sales = out.loc[sales_mask].copy()
    return out, df_sales

df_clean, df_sales = clean_data(df_raw)
print(f'Cleaned rows (including cancellations): {len(df_clean):,}')
print(f'Positive merchandise sales rows: {len(df_sales):,}')


df_sales["Revenue"]=df_sales['Quantity'] * df_sales['UnitPrice']
df_sales
dt = df_sales['InvoiceDate']
df_sales['year'] = dt.dt.year
df_sales['Quarter'] = dt.dt.quarter
df_sales['month'] = dt.dt.month
df_sales['MonName'] = dt.dt.strftime('%b')
df_sales['Day'] = dt.dt.day
df_sales['DayofWeek'] = dt.dt.dayofweek
df_sales['DayName'] = dt.dt.strftime('%a')
df_sales['Hour'] = dt.dt.hour
df_sales['week'] = dt.dt.isocalendar().week
df_sales['IsWeekend'] = df_sales['DayofWeek'].isin([5,6])
print(df_sales['Revenue'].sum())
df_sales.head()

snapshot_date = df_sales['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df_sales.groupby('CustomerID').agg(
    LastPurchase=('InvoiceDate', 'max'),
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('Revenue', 'sum')
)

rfm['Recency'] = (snapshot_date - rfm['LastPurchase']).dt.days
rfm = rfm.drop(columns=['LastPurchase'])

rfm['R_Score'] = pd.qcut(rfm['Recency'], q=4, labels=[4, 3, 2, 1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=4, labels=[1, 2, 3, 4])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=4, labels=[1, 2, 3, 4])

rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

rfm.head()

def segment_customer(df):
    r = int(df['R_Score'])
    f = int(df['F_Score'])
    
    if r == 4 and f == 4:
        return 'Champions'
    elif r >= 3 and f >= 3:
        return 'Loyal Customers'
    elif r >= 3 and f <= 2:
        return 'Potential Loyalists'
    elif r <= 2 and f >= 3:
        return 'At Risk'
    elif r <= 2 and f <= 2:
        return 'Hibernating / Lost'
    else:
        return 'Others'

rfm['Segment'] = rfm.apply(segment_customer, axis=1)
rfm['Segment'].value_counts()

from dash import Dash, dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sales = df_sales.copy()
sales["InvoiceDate"] = pd.to_datetime(
    sales["InvoiceDate"], errors="coerce"
)
sales["Revenue"] = pd.to_numeric(
    sales["Revenue"], errors="coerce"
)
sales = sales.dropna(subset=["InvoiceDate", "Revenue"])
sales["Year"] = sales["InvoiceDate"].dt.year
sales["Weekday"] = sales["InvoiceDate"].dt.dayofweek
sales["Hour"] = sales["InvoiceDate"].dt.hour

# CUSTOM COLOR PALETTE
C = {
    "bg": "#211D1A",
    "panel": "#302923",
    "text": "#F9EEDF",
    "muted": "#C4B5A3",
    "terra": "#D99572",
    "olive": "#A9B17C",
    "rose": "#C78C8D",
    "gold": "#E3BB79",
    "grid": "#50463D"
}
accents = [C["terra"], C["olive"], C["rose"], C["gold"]]

def style(fig, height=355):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=C["panel"],
        plot_bgcolor=C["panel"],
        font=dict(color=C["text"], family="Trebuchet MS, Arial"),
        title_font=dict(size=17),
        height=height,
        margin=dict(l=35, r=25, t=60, b=45),
        colorway=accents,
        legend=dict(orientation="h", y=-0.25),
        hoverlabel=dict(
            bgcolor=C["bg"], font_color=C["text"]
        )
    )
    fig.update_xaxes(gridcolor=C["grid"])
    fig.update_yaxes(gridcolor=C["grid"])
    return fig

def metric(label, component_id, color):
    return html.Div([
        html.Div(
            label,
            style={"fontSize": "12px", "color": C["muted"]}
        ),
        html.H2(
            id=component_id,
            style={"margin": "9px 0 0", "color": color}
        )
    ], style={
        "background": C["panel"],
        "padding": "21px",
        "borderRadius": "15px",
        "borderTop": f"3px solid {color}",
        "flex": "1",
        "minWidth": "165px"
    })

def panel(graph_id):
    return html.Div(
        dcc.Graph(
            id=graph_id,
            config={"displaylogo": False}
        ),
        style={
            "background": C["panel"],
            "borderRadius": "15px",
            "overflow": "hidden"
        }
    )

retail_app = Dash("retail_sales_dashboard")

retail_app.layout = html.Div([
    html.Div([
        html.Div(
            "COMMERCE / ANALYTICS",
            style={
                "letterSpacing": "3px",
                "color": C["terra"],
                "fontSize": "12px"
            }
        ),
        html.H1(
            "Online Retail Sales | Interactive Dashboard",
            style={"margin": "8px 0"}
        ),
        html.P(
            "Explore revenue, orders, products, countries and customer segments.",
            style={"color": C["muted"], "margin": "0"}
        )
    ], style={"padding": "16px 0 24px"}),

    # INTERACTIVE FILTERS
    html.Div([
        html.Div([
            html.Label("Country"),
            dcc.Dropdown(
                id="ret-country",
                options=[
                    {"label": "All countries", "value": "ALL"}
                ] + [
                    {"label": c, "value": c}
                    for c in sorted(
                        sales["Country"].dropna().astype(str).unique()
                    )
                ],
                value="ALL",
                clearable=False,
                style={"color": "#292520"}
            )
        ], style={"flex": "2", "minWidth": "200px"}),

        html.Div([
            html.Label("Year"),
            dcc.Dropdown(
                id="ret-year",
                options=[
                    {"label": "All years", "value": "ALL"}
                ] + [
                    {"label": str(y), "value": str(y)}
                    for y in sorted(sales["Year"].dropna().unique())
                ],
                value="ALL",
                clearable=False,
                style={"color": "#292520"}
            )
        ], style={"flex": "1", "minWidth": "150px"}),

        html.Div([
            html.Label("Top products"),
            dcc.Dropdown(
                id="ret-top",
                options=[5, 10, 15],
                value=10,
                clearable=False,
                style={"color": "#292520"}
            )
        ], style={"flex": "1", "minWidth": "150px"})

    ], style={
        "display": "flex",
        "flexWrap": "wrap",
        "gap": "16px",
        "padding": "19px",
        "background": C["panel"],
        "borderRadius": "14px",
        "marginBottom": "19px"
    }),

    # KPI CARDS
    html.Div([
        metric("REVENUE", "ret-revenue", C["terra"]),
        metric("ORDERS", "ret-orders", C["olive"]),
        metric("CUSTOMERS", "ret-customers", C["rose"]),
        metric("AVG. ORDER VALUE", "ret-aov", C["gold"])
    ], style={
        "display": "flex",
        "flexWrap": "wrap",
        "gap": "13px",
        "marginBottom": "20px"
    }),

    # CHARTS
    html.Div([
        panel("ret-trend"),
        panel("ret-products"),
        panel("ret-countries"),
        panel("ret-segments"),
        panel("ret-heatmap")
    ], style={
        "display": "grid",
        "gridTemplateColumns": "repeat(auto-fit, minmax(360px, 1fr))",
        "gap": "17px"
    }),

    html.P(
        "Revenue excludes cancellations. RFM segment labels use full-history data.",
        style={
            "color": C["muted"],
            "fontSize": "12px",
            "marginTop": "22px"
        }
    )
], style={
    "background": C["bg"],
    "color": C["text"],
    "fontFamily": "Trebuchet MS, Arial",
    "padding": "24px",
    "minHeight": "100vh"
})


# INTERACTIVE CALLBACK
@retail_app.callback(
    Output("ret-revenue", "children"),
    Output("ret-orders", "children"),
    Output("ret-customers", "children"),
    Output("ret-aov", "children"),
    Output("ret-trend", "figure"),
    Output("ret-products", "figure"),
    Output("ret-countries", "figure"),
    Output("ret-segments", "figure"),
    Output("ret-heatmap", "figure"),
    Input("ret-country", "value"),
    Input("ret-year", "value"),
    Input("ret-top", "value")
)
def update_retail(country, year, top_n):
    d = sales

    if country != "ALL":
        d = d[d["Country"].astype(str) == country]

    if year != "ALL":
        d = d[d["Year"] == int(year)]

    # KPIs
    revenue = d["Revenue"].sum()
    orders = d["InvoiceNo"].nunique()
    customers = d["CustomerID"].nunique()
    aov = revenue / orders if orders else 0

    # MONTHLY REVENUE
    monthly = (
        d.groupby(
            pd.Grouper(key="InvoiceDate", freq="MS")
        )["Revenue"].sum().reset_index()
    )

    trend = px.area(
        monthly,
        x="InvoiceDate",
        y="Revenue",
        title="Revenue Over Time"
    )
    trend.update_traces(
        line_color=C["terra"],
        fillcolor="rgba(217,149,114,.20)",
        hovertemplate=(
            "%{x|%b %Y}<br>£%{y:,.2f}<extra></extra>"
        )
    )

    # TOP PRODUCTS
    top_products = (
        d.groupby("Description", dropna=True)["Revenue"]
        .sum()
        .nlargest(int(top_n))
        .sort_values()
        .reset_index()
    )

    products = px.bar(
        top_products,
        x="Revenue",
        y="Description",
        orientation="h",
        title=f"Top {top_n} Products by Revenue",
        color_discrete_sequence=[C["olive"]]
    )
    products.update_layout(yaxis=dict(automargin=True))

    # COUNTRY REVENUE
    countries = (
        d.groupby("Country")["Revenue"]
        .sum()
        .nlargest(8)
        .sort_values()
        .reset_index()
    )

    country_fig = px.bar(
        countries,
        x="Revenue",
        y="Country",
        orientation="h",
        title="Revenue by Country",
        color_discrete_sequence=[C["gold"]]
    )

    # CUSTOMER SEGMENTATION
    if "rfm" in globals() and "Segment" in rfm.columns:
        seg = d.assign(
            Segment=d["CustomerID"]
            .map(rfm["Segment"])
            .fillna("Unidentified")
        )
        seg_data = (
            seg.groupby("Segment")["Revenue"]
            .sum()
            .reset_index()
        )
    else:
        seg_data = pd.DataFrame({
            "Segment": ["Not available"],
            "Revenue": [0]
        })

    segments = px.pie(
        seg_data,
        names="Segment",
        values="Revenue",
        hole=0.65,
        title="Revenue by Customer Segment",
        color_discrete_sequence=accents
    )
    segments.update_traces(
        textinfo="percent+label",
        textposition="outside"
    )

    # SALES HEATMAP
    heat_data = (
        d.groupby(["Weekday", "Hour"])["Revenue"]
        .sum()
        .unstack(fill_value=0)
        .reindex(
            index=range(7),
            columns=range(24),
            fill_value=0
        )
    )

    heat = go.Figure(go.Heatmap(
        z=heat_data.values,
        x=list(range(24)),
        y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        colorscale=[
            [0, C["panel"]],
            [0.35, C["rose"]],
            [0.7, C["terra"]],
            [1, C["gold"]]
        ],
        hovertemplate=(
            "%{y} at %{x}:00"
            "<br>Revenue: £%{z:,.0f}<extra></extra>"
        )
    ))
    heat.update_layout(
        title="Revenue Heatmap · Day × Hour",
        xaxis_title="Hour of Day"
    )

    return (
        f"£{revenue:,.0f}",
        f"{orders:,}",
        f"{customers:,}",
        f"£{aov:,.2f}",
        style(trend),
        style(products, 420),
        style(country_fig),
        style(segments),
        style(heat)
    )


server = retail_app.server

if __name__ == "__main__":
    retail_app.run(debug=False, port=8050)

