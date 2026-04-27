import streamlit as st
import pandas as pd
import numpy as np

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Business Intelligence Agent", layout="wide")


# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    try:
        deals = pd.read_csv("Deal funnel.csv")
        work_orders = pd.read_csv("work order.csv")
        return deals, work_orders
    except Exception as e:
        st.error(f"Error loading CSV files: {e}")
        st.stop()


# ----------------------------
# Clean Data
# ----------------------------
def clean_data(df):
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("unknown").astype(str).str.strip()
        else:
            df[col] = df[col].fillna(0)

    return df


# ----------------------------
# Detect Numeric Column
# ----------------------------
def detect_value_column(df):
    priority_names = [
        "value", "amount", "revenue", "deal_value",
        "price", "total", "sales", "deal_amount"
    ]

    for col in priority_names:
        if col in df.columns:
            return col

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if numeric_cols:
        return numeric_cols[0]

    return None


# ----------------------------
# Detect Status Column
# ----------------------------
def detect_status_column(df):
    possible = ["status", "stage", "deal_stage"]

    for col in possible:
        if col in df.columns:
            return col

    return None


# ----------------------------
# BI Agent
# ----------------------------
class BusinessIntelligenceAgent:

    def __init__(self, deals, work_orders):
        self.deals = deals
        self.work_orders = work_orders

        self.value_col = detect_value_column(deals)
        self.stage_col = detect_status_column(deals)
        self.work_status_col = detect_status_column(work_orders)

    def total_revenue(self):
        if self.value_col:
            return round(self.deals[self.value_col].sum(), 2)
        return "No revenue column found"

    def average_deal_size(self):
        if self.value_col:
            return round(self.deals[self.value_col].mean(), 2)
        return "No numeric data"

    def top_deals(self):
        if self.value_col:
            return self.deals.sort_values(
                by=self.value_col,
                ascending=False
            ).head(5)
        return "No top deals data"

    def pipeline_status(self):
        if self.stage_col:
            return self.deals[self.stage_col].value_counts()
        return "No pipeline data"

    def win_rate(self):
        if self.stage_col:
            total = len(self.deals)

            won = self.deals[
                self.deals[self.stage_col]
                .astype(str)
                .str.lower()
                .str.contains("won|closed")
            ].shape[0]

            if total > 0:
                return round((won / total) * 100, 2)

        return 0

    def work_order_status(self):
        if self.work_status_col:
            return self.work_orders[self.work_status_col].value_counts()
        return "No work order data"

    def ask(self, q):

        q = q.lower()

        if "revenue" in q:
            return self.total_revenue()

        elif "top" in q or "deal" in q:
            return self.top_deals()

        elif "pipeline" in q:
            return self.pipeline_status()

        elif "average" in q:
            return self.average_deal_size()

        elif "win" in q:
            return self.win_rate()

        elif "work" in q or "order" in q:
            return self.work_order_status()

        else:
            return "Try queries like revenue, top deals, pipeline, win rate"


# ----------------------------
# Main App
# ----------------------------
def main():

    st.title("📊 Business Intelligence Agent")
    st.write("Smart analytics using your uploaded CSV datasets.")

    deals, work_orders = load_data()

    deals = clean_data(deals)
    work_orders = clean_data(work_orders)

    agent = BusinessIntelligenceAgent(deals, work_orders)

    # Sidebar
    st.sidebar.title("Available Queries")
    st.sidebar.write("• total revenue")
    st.sidebar.write("• top deals")
    st.sidebar.write("• pipeline status")
    st.sidebar.write("• average deal size")
    st.sidebar.write("• win rate")
    st.sidebar.write("• work order status")

    # Show Columns
    with st.expander("📁 Dataset Columns"):
        st.write("Deals Columns:", deals.columns.tolist())
        st.write("Work Orders Columns:", work_orders.columns.tolist())

    # Input
    question = st.text_input("Ask a business question:")

    if st.button("Submit"):

        result = agent.ask(question)

        st.subheader("Result")

        if isinstance(result, pd.DataFrame):
            st.dataframe(result, use_container_width=True)

        elif isinstance(result, pd.Series):
            st.bar_chart(result)

        else:
            st.success(result)

    # Dashboard Metrics
    st.divider()
    st.subheader("📌 Quick Dashboard")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Total Revenue", agent.total_revenue())

    with c2:
        st.metric("Average Deal Size", agent.average_deal_size())

    with c3:
        st.metric("Win Rate", f"{agent.win_rate()}%")


if __name__ == "__main__":
    main()
