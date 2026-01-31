import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Blockchain Anomaly Detection", layout="wide")

st.title("🚨 Blockchain Block Anomaly Detection Dashboard")
st.markdown("Detects suspicious blockchain blocks using percentile-based analysis.")

# =========================
# STEP 1: LOAD DATA
# =========================
df = pd.read_csv("data.csv.gz")
df.columns = df.columns.str.strip()

st.success("✅ Data Loaded Successfully")
st.subheader("Sample Data")
st.dataframe(df.head())

# =========================
# STEP 2: DATA CLEANING
# =========================
df = df.dropna()
df = df.drop_duplicates()

numeric_cols = [
    "size", "tx_count", "difficulty",
    "median_fee_rate", "avg_fee_rate", "total_fees",
    "fee_range_min", "fee_range_max",
    "input_count", "output_count", "output_amount"
]

numeric_cols = [col for col in numeric_cols if col in df.columns]

for col in numeric_cols:
    df = df[df[col] >= 0]

st.info(f"🧹 Rows after cleaning: {len(df)}")

# =========================
# STEP 3: FEATURE ENGINEERING
# =========================
df["fee_spread"] = df["fee_range_max"] - df["fee_range_min"]
df["tx_complexity"] = df["input_count"] + df["output_count"]
df["fee_per_tx"] = df["total_fees"] / (df["tx_count"] + 1)

st.subheader("🧠 Engineered Features")
st.dataframe(df[["fee_spread", "tx_complexity", "fee_per_tx"]].head())

# =========================
# STEP 4: INTERACTIVE THRESHOLD
# =========================
st.sidebar.header("⚙️ Anomaly Settings")

percentile = st.sidebar.slider(
    "Anomaly Percentile Threshold",
    min_value=90,
    max_value=99,
    value=95
)

# =========================
# STEP 5: ANOMALY DETECTION
# =========================
fee_spread_thresh = df["fee_spread"].quantile(percentile / 100)
complexity_thresh = df["tx_complexity"].quantile(percentile / 100)
fee_tx_thresh = df["fee_per_tx"].quantile(percentile / 100)

df["suspicious"] = (
    (df["fee_spread"] > fee_spread_thresh) |
    (df["tx_complexity"] > complexity_thresh) |
    (df["fee_per_tx"] > fee_tx_thresh)
)

suspicious_blocks = df[df["suspicious"]]

st.subheader("🚨 Suspicious Blocks")
st.write(f"Detected **{len(suspicious_blocks)}** suspicious blocks")

st.dataframe(
    suspicious_blocks[
        ["height", "fee_spread", "tx_complexity", "fee_per_tx"]
    ].head(20)
)

# =========================
# STEP 6: RISK SCORING
# =========================
df["risk_score"] = (
    0.4 * (df["fee_spread"] / df["fee_spread"].max()) +
    0.3 * (df["tx_complexity"] / df["tx_complexity"].max()) +
    0.3 * (df["fee_per_tx"] / df["fee_per_tx"].max())
)

df["risk_score"] = df["risk_score"].round(3)

st.subheader("🔥 Highest Risk Blocks")
st.dataframe(
    df.sort_values("risk_score", ascending=False)[
        ["height", "risk_score"]
    ].head(10)
)

# =========================
# STEP 7: VISUALIZATION
# =========================
st.subheader("📈 Visual Anomaly Map")

fee_cap = df["fee_spread"].quantile(0.99)
df_plot = df[df["fee_spread"] <= fee_cap]

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
    df_plot["tx_complexity"],
    df_plot["fee_spread"],
    c=df_plot["risk_score"]
)

ax.set_xlabel("Transaction Complexity")
ax.set_ylabel("Fee Spread (capped at 99th percentile)")
ax.set_title("Blockchain Block Anomaly Detection")

plt.colorbar(scatter, ax=ax, label="Risk Score")
st.pyplot(fig)
