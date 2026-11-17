"""PREDICTIVE. | Interactive web app based on the AI4I Predictive Maintenance notebook.
Run: streamlit run Predictive-Maintenance/streamlit_app.py
"""
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="PREDICTIVE. | Machine Health",page_icon="⚙️",layout="wide")
SAGE,RUST,OCHRE,BLUE="#BFD3A7","#E08E70","#DCC18A","#96B9BB"
COLS=["Type","Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]"]
MODES=["TWF","HDF","PWF","OSF","RNF"]
st.markdown("""<style>
.stApp{background:#111814;color:#EAEFE7}
section[data-testid="stSidebar"]{background:#1B261F;border-right:1px solid #405443}
.block-container{max-width:1520px;padding-top:1.7rem}
[data-testid="stMetric"]{background:#202D24;border:1px solid #455748;border-radius:15px;padding:18px}
h1,h2,h3{letter-spacing:-.03em}
.hero{background:linear-gradient(120deg,#354732,#1D2B22);border:1px solid #53664F;border-radius:18px;padding:29px 31px;margin-bottom:23px}
.hero h1{font-size:clamp(29px,3vw,49px);line-height:1.06;color:#F0F5E9;margin:.3em 0}
.hero p{color:#B9CAB9;margin:0;font-size:15px}
.eyebrow{font-size:11px;letter-spacing:.19em;font-weight:800;color:#C4D8AC}
</style>""",unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data():
    path=Path(__file__).parent/"ai4i2020.csv"
    data=pd.read_csv(path)
    data.columns=data.columns.str.replace("\ufeff","",regex=False).str.strip()
    data["Type"]=data["Type"].astype("string").str.strip().str.upper()
    for col in COLS[1:]+["Machine failure"]+MODES:
        data[col]=pd.to_numeric(data[col],errors="coerce")
    data=data.drop_duplicates().dropna(subset=COLS+["Machine failure"]).copy()
    data=data[data["Machine failure"].isin([0,1]) & data["Type"].isin(["L","M","H"])]
    data["Machine failure"]=data["Machine failure"].astype(int)
    return data.reset_index(drop=True)

@st.cache_resource(show_spinner="Training models on the AI4I benchmark...")
def train():
    df=load_data()
    xtr,xte,ytr,yte=train_test_split(df[COLS],df["Machine failure"],test_size=.2,
                                   random_state=42,stratify=df["Machine failure"])
    transformer=ColumnTransformer([
        ("numeric",StandardScaler(),COLS[1:]),
        ("grade",OneHotEncoder(handle_unknown="ignore"),["Type"])
    ])
    classifiers={
        "Logistic Regression":LogisticRegression(max_iter=1000,class_weight="balanced",random_state=42),
        "Decision Tree":DecisionTreeClassifier(max_depth=5,class_weight="balanced",random_state=42),
        "Random Forest":RandomForestClassifier(n_estimators=300,max_depth=15,min_samples_leaf=4,
                                              class_weight="balanced_subsample",random_state=42,n_jobs=-1)
    }
    fitted={}
    metrics=[]
    for name,clf in classifiers.items():
        pipeline=Pipeline([("transform",transformer),("classifier",clf)])
        pipeline.fit(xtr,ytr)
        predictions=pipeline.predict(xte)
        fitted[name]=pipeline
        metrics.append({"Model":name,"Accuracy":accuracy_score(yte,predictions),
                        "Precision":precision_score(yte,predictions,zero_division=0),
                        "Recall":recall_score(yte,predictions,zero_division=0),
                        "F1":f1_score(yte,predictions,zero_division=0)})
    cm=confusion_matrix(yte,fitted["Random Forest"].predict(xte),labels=[0,1])
    return fitted["Random Forest"],pd.DataFrame(metrics),cm,len(xtr),len(xte)

def draw(fig,height=350):
    fig.update_layout(height=height,margin=dict(l=12,r=12,t=22,b=12),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#EAEFE7",size=12),legend=dict(orientation="h",y=1.15,x=0),
        xaxis=dict(gridcolor="#35453B",zeroline=False),yaxis=dict(gridcolor="#35453B",zeroline=False))
    st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})

try:
    df=load_data()
except Exception as error:
    st.error(f"Cannot read Predictive-Maintenance/ai4i2020.csv: {error}")
    st.stop()

st.sidebar.title("PREDICTIVE.")
st.sidebar.caption("MACHINE HEALTH STUDIO / MALAK MAHMOUD")
page=st.sidebar.radio("WORKSPACE",["Overview","Machine Explorer","Model Lab","Prediction Studio","Data Desk"])
st.sidebar.divider()
st.sidebar.caption("GLOBAL FILTERS")
types=st.sidebar.multiselect("Machine grade",["L","M","H"],default=["L","M","H"])
tool_low,tool_high=int(df["Tool wear [min]"].min()),int(df["Tool wear [min]"].max())
wear=st.sidebar.slider("Tool wear · min",tool_low,tool_high,(tool_low,tool_high))
status=st.sidebar.selectbox("Outcome",["All","No failure","Recorded failure"])
filtered=df[df["Type"].isin(types) & df["Tool wear [min]"].between(*wear)].copy()
if status!="All":
    filtered=filtered[filtered["Machine failure"]==(1 if status=="Recorded failure" else 0)]
filtered["Outcome"]=filtered["Machine failure"].map({0:"No failure",1:"Recorded failure"})
st.sidebar.divider()
st.sidebar.caption("SOURCE: UCI AI4I 2020 (SYNTHETIC BENCHMARK)")
st.markdown('<div class="hero"><div class="eyebrow">ENGINEERING DATA / LIVE ANALYSIS</div>'
            '<h1>Machine health, made readable.</h1>'
            '<p>Explore operating signals, recorded failures, and classifier performance.</p></div>',
            unsafe_allow_html=True)
st.caption(f"{page.upper()} / {len(filtered):,} FILTERED RECORDS / MODEL EVALUATION IS ALWAYS ON THE FULL DATASET")

if page=="Overview":
    n=len(filtered);f=int(filtered["Machine failure"].sum())
    a,b,c,d=st.columns(4)
    a.metric("OBSERVATIONS",f"{n:,}")
    b.metric("RECORDED FAILURES",f"{f:,}")
    c.metric("FAILURE RATE",f"{100*f/n:.2f}%" if n else "—")
    d.metric("MEAN TOOL WEAR",f"{filtered['Tool wear [min]'].mean():.1f} min" if n else "—")
    if not n:
        st.info("No machines match these filters.")
    else:
        left,right=st.columns([1.2,.8])
        with left:
            st.subheader("Torque vs. rotational speed")
            sample=filtered.sample(min(len(filtered),4000),random_state=42)
            draw(px.scatter(sample,x="Rotational speed [rpm]",y="Torque [Nm]",color="Outcome",
                color_discrete_map={"No failure":SAGE,"Recorded failure":RUST},
                hover_data=["Type","Tool wear [min]"],opacity=.63))
        with right:
            st.subheader("Failure rate by grade")
            rates=(filtered.groupby("Type")["Machine failure"].mean()*100).reset_index(name="Rate")
            fig=px.bar(rates,x="Type",y="Rate",color="Type",text="Rate",
                color_discrete_map={"L":RUST,"M":OCHRE,"H":SAGE})
            fig.update_traces(texttemplate="%{text:.2f}%")
            draw(fig)
        left,right=st.columns(2)
        with left:
            st.subheader("Failure-mode indicators")
            counts=filtered[MODES].sum().sort_values().reset_index()
            counts.columns=["Mode","Count"]
            draw(px.bar(counts,x="Count",y="Mode",orientation="h",color_discrete_sequence=[RUST]))
        with right:
            st.subheader("Tool-wear distribution")
            draw(px.histogram(filtered,x="Tool wear [min]",color="Outcome",nbins=35,opacity=.8,
                color_discrete_map={"No failure":SAGE,"Recorded failure":RUST},barmode="overlay"))

elif page=="Machine Explorer":
    st.subheader("Explore operating conditions")
    if filtered.empty:
        st.info("No matching observations.")
    else:
        a,b=st.columns(2)
        with a:xcol=st.selectbox("X-axis",COLS[1:],index=2)
        with b:ycol=st.selectbox("Y-axis",COLS[1:],index=3)
        sample=filtered.sample(min(len(filtered),4000),random_state=42)
        draw(px.scatter(sample,x=xcol,y=ycol,color="Outcome",symbol="Type",opacity=.55,
            color_discrete_map={"No failure":SAGE,"Recorded failure":RUST}),height=420)
        a,b=st.columns(2)
        with a:
            draw(px.box(filtered,x="Outcome",y=ycol,color="Outcome",points=False,
                color_discrete_map={"No failure":SAGE,"Recorded failure":RUST}))
        with b:
            group=filtered.groupby(["Type","Outcome"]).size().reset_index(name="Observations")
            draw(px.bar(group,x="Type",y="Observations",color="Outcome",barmode="group",
                color_discrete_map={"No failure":SAGE,"Recorded failure":RUST}))
    st.caption("These associations do not prove what causes a failure.")

elif page=="Model Lab":
    st.subheader("Model comparison / held-out test set")
    st.info("Models use machine grade and five operating features. IDs and failure-mode labels are excluded to prevent target leakage.")
    model,scores,cm,ntrain,ntest=train()
    a,b=st.columns(2);a.metric("TRAIN ROWS",f"{ntrain:,}");b.metric("TEST ROWS",f"{ntest:,}")
    st.dataframe(scores.style.format({"Accuracy":"{:.3f}","Precision":"{:.3f}",
                                      "Recall":"{:.3f}","F1":"{:.3f}"}),
                 use_container_width=True,hide_index=True)
    a,b=st.columns([1.2,.8])
    with a:
        draw(px.bar(scores,x="Model",y=["F1","Recall","Precision"],barmode="group",
                    color_discrete_sequence=[SAGE,RUST,BLUE]))
    with b:
        fig=go.Figure(go.Heatmap(z=cm,x=["Pred 0","Pred 1"],y=["Actual 0","Actual 1"],
            text=cm,texttemplate="%{text}",showscale=False,
            colorscale=[[0,"#202B24"],[1,SAGE]]))
        fig.update_layout(yaxis_autorange="reversed")
        draw(fig)
    st.caption("Split: stratified 80/20, random_state=42. RF uses settings from the original notebook, retrained for this app.")

elif page=="Prediction Studio":
    st.subheader("Test a machine profile")
    st.warning("Educational demonstration only. Not a calibrated real-world failure risk or a maintenance recommendation.")
    with st.form("predict"):
        a,b,c=st.columns(3)
        with a:
            typ=st.selectbox("Machine grade",["L","M","H"])
            air=st.number_input("Air temperature (K)",250.0,360.0,300.5,.1)
        with b:
            process=st.number_input("Process temperature (K)",250.0,400.0,310.2,.1)
            speed=st.number_input("Rotational speed (rpm)",100,5000,1400,10)
        with c:
            torque=st.number_input("Torque (Nm)",0.0,150.0,55.0,.5)
            wear_input=st.number_input("Tool wear (min)",0,1000,180,5)
        submitted=st.form_submit_button("Calculate estimate",type="primary")
    if submitted:
        model,_,_,_,_=train()
        entry=pd.DataFrame([[typ,air,process,speed,torque,wear_input]],columns=COLS)
        prediction=int(model.predict(entry)[0])
        score=float(model.predict_proba(entry)[0,1])
        a,b=st.columns(2)
        a.metric("Predicted class","Recorded failure" if prediction else "No failure")
        b.metric("Uncalibrated model score",f"{score:.1%}")
        st.caption("The benchmark does not provide remaining useful life or safe-operation guarantees.")

elif page=="Data Desk":
    st.subheader("Data inspection & export")
    st.caption("Original source file is never modified. Download the filtered prepared version below.")
    st.dataframe(filtered.drop(columns="Outcome"),use_container_width=True,hide_index=True,height=420)
    st.download_button("Download filtered CSV",filtered.drop(columns="Outcome").to_csv(index=False).encode("utf-8-sig"),
        file_name="ai4i_filtered.csv",mime="text/csv",type="primary")
    st.info("AI4I 2020 is a synthetic benchmark with no timestamps. No fabricated time-series claims are made.")

st.divider()
st.caption("PREDICTIVE. / Malak Mahmoud / UCI AI4I 2020 (Matzka) / CC BY 4.0 / For educational use.")
