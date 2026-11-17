"""Online Retail Sales · Streamlit website
Adapted from the user's Online Retail Sales notebook (same data cleaning, RFM, charts,
filter controls, KPI definitions and Espresso / Terracotta / Olive / Rose / Gold palette).
"""
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Online Retail Sales | Interactive Dashboard",
                   page_icon="🛍️", layout="wide")

C = {
    "bg": "#211D1A", "panel": "#302923", "text": "#F9EEDF",
    "muted": "#C4B5A3", "terra": "#D99572", "olive": "#A9B17C",
    "rose": "#C78C8D", "gold": "#E3BB79", "grid": "#50463D"
}
accents = [C["terra"], C["olive"], C["rose"], C["gold"]]
NON_PRODUCT_CODES = {"POST", "D", "M", "BANK CHARGES", "DOT", "C2",
                     "CRUK", "PADS", "AMAZONFEE", "S"}

st.markdown("""<style>
.stApp,[data-testid="stAppViewContainer"],section.main{background:#211D1A;color:#F9EEDF}
[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1480px;padding:30px 34px 58px}
h1,h2,h3,p{font-family:"Trebuchet MS",Arial,sans-serif}
div[data-testid="stSelectbox"] label p{color:#C4B5A3;font-size:14px;font-weight:600}
[data-baseweb="select"]>div{background:#3B332C;border:1px solid #57493C;border-radius:9px;color:#F9EEDF}
[data-testid="stPlotlyChart"]{background:#302923;border-radius:15px;overflow:hidden;padding:3px}
[data-testid="stMarkdownContainer"] p{color:#C4B5A3}
@media(max-width:700px){.block-container{padding:17px 13px 35px}.ret-title{font-size:29px!important}}
</style>""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Loading and preparing the original retail dataset...")
def load_data():
    path = Path(__file__).resolve().parent / "online_retail.csv.gz"
    raw = pd.read_csv(path, compression="gzip")
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"], errors="coerce")
    # Same cleaning as notebook: drop duplicate rows, tag cancellations/adjustments.
    clean = raw.drop_duplicates().copy()
    clean["IsCancelled"] = clean["InvoiceNo"].astype(str).str.upper().str.startswith("C")
    clean["IsNonProduct"] = clean["StockCode"].astype(str).str.upper().isin(NON_PRODUCT_CODES)
    clean["IsZeroOrNegPrice"] = clean["UnitPrice"] <= 0
    clean["Description"] = clean["Description"].fillna("Unknown")
    mask = (~clean["IsCancelled"] & ~clean["IsNonProduct"]
            & ~clean["IsZeroOrNegPrice"] & clean["Quantity"].gt(0)
            & clean["InvoiceDate"].notna() & clean["InvoiceNo"].notna()
            & clean["StockCode"].notna())
    sales = clean.loc[mask].copy()
    sales["Revenue"] = sales["Quantity"] * sales["UnitPrice"]
    sales["Year"] = sales["InvoiceDate"].dt.year
    sales["Weekday"] = sales["InvoiceDate"].dt.dayofweek
    sales["Hour"] = sales["InvoiceDate"].dt.hour

    # Same full-history RFM method and rules from notebook cells 22-24.
    snapshot = sales["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = sales.groupby("CustomerID").agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum")
    )
    rfm["Recency"] = (snapshot - rfm["LastPurchase"]).dt.days
    rfm = rfm.drop(columns=["LastPurchase"])
    rfm["R_Score"] = pd.qcut(rfm["Recency"], q=4, labels=[4, 3, 2, 1])
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=4,
                             labels=[1, 2, 3, 4])
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], q=4, labels=[1, 2, 3, 4])
    rfm["RFM_Score"] = (rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str)
                        + rfm["M_Score"].astype(str))
    def assign_segment(row):
        r, f = int(row["R_Score"]), int(row["F_Score"])
        if r == 4 and f == 4:
            return "Champions"
        elif r >= 3 and f >= 3:
            return "Loyal Customers"
        elif r >= 3 and f <= 2:
            return "Potential Loyalists"
        elif r <= 2 and f >= 3:
            return "At Risk"
        elif r <= 2 and f <= 2:
            return "Hibernating / Lost"
        return "Others"
    rfm["Segment"] = rfm.apply(assign_segment, axis=1)
    return sales, rfm

def style(fig, height=355):
    fig.update_layout(template="plotly_dark", paper_bgcolor=C["panel"],
        plot_bgcolor=C["panel"], font=dict(color=C["text"],family="Trebuchet MS, Arial"),
        title_font=dict(size=17),height=height,
        margin=dict(l=35,r=25,t=60,b=45),colorway=accents,
        legend=dict(orientation="h",y=-0.25),
        hoverlabel=dict(bgcolor=C["bg"],font_color=C["text"]))
    fig.update_xaxes(gridcolor=C["grid"])
    fig.update_yaxes(gridcolor=C["grid"])
    return fig

def chart(fig,height=355):
    st.plotly_chart(style(fig,height),use_container_width=True,
                    config={"displaylogo":False,"responsive":True})

def metric_card(label,value,color):
    st.markdown(f"""<div style="background:{C['panel']};padding:21px;
    border-radius:15px;border-top:3px solid {color};min-height:113px;
    font-family:Trebuchet MS,Arial">
    <div style="font-size:12px;color:{C['muted']}">{label}</div>
    <div style="font-size:28px;font-weight:700;color:{color};margin-top:9px">{value}</div>
    </div>""",unsafe_allow_html=True)

def empty_chart(title):
    fig=go.Figure()
    fig.update_layout(title=title)
    fig.add_annotation(text="No data for these filters",xref="paper",yref="paper",
                       x=.5,y=.5,showarrow=False,font_color=C["muted"])
    return fig

try:
    sales,rfm=load_data()
except Exception as e:
    st.error(f"Data preparation failed: {e}")
    st.info("The source file 'online_retail.csv.gz' must be beside streamlit_app.py.")
    st.stop()

st.markdown(f"""<div style="letter-spacing:3px;color:{C['terra']};font-size:12px;
font-family:Trebuchet MS,Arial">COMMERCE / ANALYTICS</div>
<h1 class="ret-title" style="margin:8px 0;color:{C['text']};font-size:39px;
font-family:Trebuchet MS,Arial">Online Retail Sales | Interactive Dashboard</h1>
<p style="color:{C['muted']};margin:0 0 24px;font-size:15px">
Explore revenue, orders, products, countries and customer segments.</p>""",unsafe_allow_html=True)

col1,col2,col3=st.columns([2,1,1])
with col1:
    country=st.selectbox("Country",["All countries"]+
           sorted(sales["Country"].dropna().astype(str).unique().tolist()))
with col2:
    year=st.selectbox("Year",["All years"]+
           [str(int(x)) for x in sorted(sales["Year"].dropna().unique())])
with col3:
    top_n=st.selectbox("Top products",[5,10,15],index=1)

d=sales
if country!="All countries":
    d=d[d["Country"].astype(str)==country]
if year!="All years":
    d=d[d["Year"]==int(year)]

revenue=float(d["Revenue"].sum())
orders=int(d["InvoiceNo"].nunique())
customers=int(d["CustomerID"].nunique())
aov=revenue/orders if orders else 0.0
st.markdown("<div style='height:9px'></div>",unsafe_allow_html=True)
a,b,c,e=st.columns(4)
with a:metric_card("REVENUE",f"£{revenue:,.0f}",C["terra"])
with b:metric_card("ORDERS",f"{orders:,}",C["olive"])
with c:metric_card("CUSTOMERS",f"{customers:,}",C["rose"])
with e:metric_card("AVG. ORDER VALUE",f"£{aov:,.2f}",C["gold"])
st.markdown("<div style='height:12px'></div>",unsafe_allow_html=True)

if d.empty:
    trend=empty_chart("Revenue Over Time")
    products=empty_chart(f"Top {top_n} Products by Revenue")
    countries=empty_chart("Revenue by Country")
    segments=empty_chart("Revenue by Customer Segment")
    heat=empty_chart("Revenue Heatmap · Day × Hour")
else:
    monthly=d.groupby(pd.Grouper(key="InvoiceDate",freq="MS"))["Revenue"].sum().reset_index()
    trend=px.area(monthly,x="InvoiceDate",y="Revenue",title="Revenue Over Time")
    trend.update_traces(line_color=C["terra"],fillcolor="rgba(217,149,114,.20)",
        hovertemplate="%{x|%b %Y}<br>£%{y:,.2f}<extra></extra>")

    top=(d.groupby("Description",dropna=True)["Revenue"].sum()
         .nlargest(int(top_n)).sort_values().reset_index())
    products=px.bar(top,x="Revenue",y="Description",orientation="h",
                    title=f"Top {top_n} Products by Revenue",
                    color_discrete_sequence=[C["olive"]])
    products.update_layout(yaxis=dict(automargin=True))

    by_country=(d.groupby("Country")["Revenue"].sum().nlargest(8)
                .sort_values().reset_index())
    countries=px.bar(by_country,x="Revenue",y="Country",orientation="h",
                     title="Revenue by Country",
                     color_discrete_sequence=[C["gold"]])

    with_segment=d.assign(Segment=d["CustomerID"].map(rfm["Segment"]).fillna("Unidentified"))
    segment_data=with_segment.groupby("Segment")["Revenue"].sum().reset_index()
    segments=px.pie(segment_data,names="Segment",values="Revenue",hole=.65,
                    title="Revenue by Customer Segment",
                    color_discrete_sequence=accents)
    segments.update_traces(textinfo="percent+label",textposition="outside")

    heat_data=(d.groupby(["Weekday","Hour"])["Revenue"].sum()
               .unstack(fill_value=0).reindex(index=range(7),
               columns=range(24),fill_value=0))
    heat=go.Figure(go.Heatmap(z=heat_data.values,x=list(range(24)),
        y=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
        colorscale=[[0,C["panel"]],[.35,C["rose"]],[.7,C["terra"]],[1,C["gold"]]],
        hovertemplate="%{y} at %{x}:00<br>Revenue: £%{z:,.0f}<extra></extra>"))
    heat.update_layout(title="Revenue Heatmap · Day × Hour",xaxis_title="Hour of Day")

left,right=st.columns(2)
with left: chart(trend)
with right: chart(products,420)
left,right=st.columns(2)
with left: chart(countries)
with right: chart(segments)
chart(heat)
st.markdown(f"""<p style="color:{C['muted']};font:12px Trebuchet MS,Arial;
margin-top:22px">Revenue excludes cancellations and non-merchandise lines.
RFM segments use the full transaction history. Dec 2010–Dec 2011.
Revenue is not profit. Data: UCI Online Retail.</p>""",unsafe_allow_html=True)
