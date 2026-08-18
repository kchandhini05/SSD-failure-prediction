import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

# -----------------------
# LOAD METRICS
# -----------------------

try:
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
except:
    metrics = {"accuracy":0, "precision":0, "recall":0, "f1":0}

accuracy = metrics.get("accuracy", 0) * 100
precision = metrics.get("precision", 0) * 100
recall = metrics.get("recall", 0) * 100
f1 = metrics.get("f1", 0) * 100

# -----------------------
# PAGE CONFIG
# -----------------------

st.set_page_config(
    page_title="SSD AI Monitoring System",
    page_icon="💾",
    layout="wide"
)

# -----------------------
# LOGIN SYSTEM
# -----------------------

def login():
    st.markdown("""
    <h1 style='text-align:center;color:#00BFFF'>
    💾 SSD Monitoring System
    </h1>
    <h4 style='text-align:center'>
    AI Predictive Maintenance Dashboard
    </h4>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username=="admin" and password=="admin123":
                st.session_state["logged"]=True
                st.success("✅ Login successful")
                st.rerun()
            else:
                st.error("❌ Invalid credentials")

if "logged" not in st.session_state:
    st.session_state["logged"]=False

if not st.session_state["logged"]:
    login()
    st.stop()

# -----------------------
# LOAD DATA
# -----------------------

try:
    data = pd.read_csv("results.csv")
except:
    st.error("⚠ Run main.py first")
    st.stop()

# -----------------------
# ADD STATUS
# -----------------------

data["Status"] = data["prediction"].apply(
    lambda x: "Failure Risk" if x==1 else "Healthy"
)

# -----------------------
# FEATURE MAPPING (SAFE)
# -----------------------

feature_temp = "smart_9_raw"     # Power-on hours
feature_wear = "smart_5_raw"     # Wear
feature_error = "smart_187_raw"  # Errors

# -----------------------
# SIDEBAR FILTERS
# -----------------------

st.sidebar.title("⚙ Filters")

temp = st.sidebar.slider(
    "Usage (Power-On Hours)",
    int(data[feature_temp].min()),
    int(data[feature_temp].max()),
    (int(data[feature_temp].min()), int(data[feature_temp].max()))
)

wear = st.sidebar.slider(
    "Wear Level",
    int(data[feature_wear].min()),
    int(data[feature_wear].max()),
    (int(data[feature_wear].min()), int(data[feature_wear].max()))
)

errors = st.sidebar.slider(
    "Read Errors",
    int(data[feature_error].min()),
    int(data[feature_error].max()),
    (int(data[feature_error].min()), int(data[feature_error].max()))
)

status = st.sidebar.selectbox(
    "SSD Status",
    ["All","Healthy","Failure Risk"]
)

# -----------------------
# FILTER DATA
# -----------------------

filtered = data[
    (data[feature_temp]>=temp[0]) &
    (data[feature_temp]<=temp[1]) &
    (data[feature_wear]>=wear[0]) &
    (data[feature_wear]<=wear[1]) &
    (data[feature_error]>=errors[0]) &
    (data[feature_error]<=errors[1])
]

if status=="Healthy":
    filtered = filtered[filtered["prediction"]==0]

if status=="Failure Risk":
    filtered = filtered[filtered["prediction"]==1]

# -----------------------
# TITLE
# -----------------------

st.title("💾 SSD Failure Prediction Dashboard")

# -----------------------
# KPI METRICS
# -----------------------

total = len(filtered)
fail = len(filtered[filtered["prediction"]==1])
healthy = total - fail

c1,c2,c3,c4 = st.columns(4)

c1.metric("Total SSD", total)
c2.metric("Healthy", healthy)
c3.metric("Failure Risk", fail)
c4.metric("Accuracy", f"{accuracy:.2f}%")

# -----------------------
# MODEL PERFORMANCE
# -----------------------

st.subheader("🤖 AI Model Performance")

m1,m2,m3,m4 = st.columns(4)

m1.metric("Accuracy", f"{accuracy:.2f}%")
m2.metric("Precision", f"{precision:.2f}%")
m3.metric("Recall", f"{recall:.2f}%")
m4.metric("F1 Score", f"{f1:.2f}%")

# -----------------------
# ALERTS
# -----------------------

st.subheader("🚨 Failure Alerts")

if fail > 0:
    st.error(f"🚨 {fail} SSDs at high failure risk!")
    st.warning("⚠ Backup data immediately!")
else:
    st.success("✅ All SSDs are healthy")

# -----------------------
# SYSTEM HEALTH
# -----------------------

health = int((healthy/total)*100) if total>0 else 0

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=health,
    title={"text":"System Health"},
    gauge={"axis":{"range":[0,100]}}
))

st.plotly_chart(fig, use_container_width=True)

# -----------------------
# SSD LIFE
# -----------------------

st.subheader("🧠 Remaining SSD Life")

filtered = filtered.copy()
filtered["Remaining_Life"] = 100 - filtered[feature_wear]

fig2 = px.line(filtered.head(1000), y="Remaining_Life", title="Remaining Life Prediction")
st.plotly_chart(fig2, use_container_width=True)

# -----------------------
# ANALYTICS
# -----------------------

st.subheader("📊 Analytics")

col1,col2 = st.columns(2)

fig3 = px.scatter(
    filtered.sample(min(2000,len(filtered))),
    x=feature_temp,
    y=feature_wear,
    color="Status",
    title="Usage vs Wear"
)
col1.plotly_chart(fig3, use_container_width=True)

fig4 = px.histogram(
    filtered.sample(min(2000,len(filtered))),
    x=feature_error,
    color="Status",
    title="Error Distribution"
)
col2.plotly_chart(fig4, use_container_width=True)

# -----------------------
# MODEL INFO
# -----------------------

st.subheader("🤖 Model Info")
st.write("Hybrid Model: Isolation Forest + Random Forest + Mahalanobis Distance")
st.write("SMOTE applied for class imbalance handling")

# -----------------------
# DATA TABLE (SAFE)
# -----------------------

st.subheader("📄 SSD Data (Sample)")

st.dataframe(filtered.head(500), use_container_width=True)

# -----------------------
# DOWNLOAD
# -----------------------

csv = filtered.to_csv(index=False).encode()

st.download_button(
    "Download CSV",
    csv,
    "ssd_data.csv",
    "text/csv"
)

# -----------------------
# LOGOUT
# -----------------------

if st.sidebar.button("Logout"):
    st.session_state["logged"]=False
    st.rerun()