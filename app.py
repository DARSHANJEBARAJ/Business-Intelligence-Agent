import streamlit as st
import pandas as pd
import numpy as np
import os

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Business Intelligence AI Agent",
    page_icon="📊",
    layout="wide"
)

# -------------------------------------------------
# LOAD DATA (AUTO FILE DETECTION)
# -------------------------------------------------
@st.cache_data
def load_data():

    files = os.listdir()

    deals_file = None
    work_file = None

    for file in files:
        name = file.lower()

        if file.endswith(".csv"):

            if "deal" in name:
                deals_file = file

            elif "work" in name or "order" in name:
                work_file = file

    if deals_file is None:
        st.error("Deals CSV file not found.")
        st.stop()

    if work_file is None:
        st.error("Work Orders CSV file not found.")
        st.stop()

    deals = pd.read_csv(deals_file)
    work_orders = pd.read_csv(work_file)

    return deals, work_orders, deals_file, work_file


# -------------------------------------------------
# CLEAN DATA
# -------------------------------------------------
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


# -------------------------------------------------
# DETECT BEST NUMERIC COLUMN
# -------------------------------------------------
def detect_value_column(df):

    priority = [
        "value",
        "deal_value",
        "amount",
        "revenue",
        "sales",
        "price",
        "total",
        "contract_value"
    ]

    for col in df.columns:
        if col in priority:
            return col

    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if num_cols:
        return max(num_cols, key=lambda c: df[c].sum())

    return None


# -------------------------------------------------
# DETECT TEXT COLUMN
# -------------------------------------------------
def detect_column(df, names):

    for col in df.columns:
        if col in names:
            return col

    return None


# -------------------------------------------------
# BI AGENT
# -------------------------------------------------
class BIAgent:

    def __init__(self, deals, work_orders):

        self.deals = deals
        self.work_orders = work_orders

        self.value_col = detect_value_column(deals)

        self.stage_col = detect_column(
            deals,
            ["stage", "status", "deal_stage"]
        )

        self.owner_col = detect_column(
            deals,
            ["owner", "salesperson", "manager"]
        )

        self.sector_col = detect_column(
            deals,
            ["sector", "industry", "category"]
        )

        self.work_status_col = detect_column(
            work_orders,
            ["status", "stage"]
        )

    # ---------------------------------------------
    def revenue(self):

        if self.value_col:
            return round(self.deals[self.value_col].sum(), 2)

        return 0

    # ---------------------------------------------
    def avg_deal(self):

        if self.value_col:
            return round(self.deals[self.value_col].mean(), 2)

        return 0

    # ---------------------------------------------
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

    # ---------------------------------------------
    def ask(self, q):

        q = q.lower()

        # Revenue
        if "revenue" in q:
            return f"💰 Total Revenue = {self.revenue():,.2f}"

        # Average
        elif "average" in q:
            return f"📊 Average Deal Size = {self.avg_deal():,.2f}"

        # Win Rate
        elif "win rate" in q:
            return f"🏆 Win Rate = {self.win_rate()}%"

        # Top Deals
        elif "top deals" in q:

            if self.value_col:
                return self.deals.sort_values(
                    by=self.value_col,
                    ascending=False
                ).head(5)

        # Pipeline
        elif "pipeline" in q:

            if self.stage_col:
                return self.deals[self.stage_col].value_counts()

        # Work Orders
        elif "work order" in q:

            if self.work_status_col:
                return self.work_orders[
                    self.work_status_col
                ].value_counts()

        # Sector Revenue
        elif "sector revenue" in q:

            if self.sector_col and self.value_col:

                return self.deals.groupby(
                    self.sector_col
                )[self.value_col].sum()

        # Top Owner
        elif "top owner" in q:

            if self.owner_col and self.value_col:

                return self.deals.groupby(
                    self.owner_col
                )[self.value_col].sum().sort_values(
                    ascending=False
                ).head(5)

        else:

            return """
Try asking:

• total revenue  
• average deal size  
• top deals  
• pipeline status  
• win rate  
• work order status  
• sector revenue  
• top owner
"""


# -------------------------------------------------
# MAIN APP
# -------------------------------------------------
def main():

    st.title("📊 Business Intelligence AI Agent")
    st.write("Founder-level business answers from multiple datasets.")

    # Load files
    deals, work_orders, deals_file, work_file = load_data()

    deals = clean_data(deals)
    work_orders = clean_data(work_orders)

    agent = BIAgent(deals, work_orders)

    # Sidebar
    st.sidebar.success("Files Loaded")
    st.sidebar.write(f"Deals File: {deals_file}")
    st.sidebar.write(f"Work File: {work_file}")

    # Query Box
    question = st.text_input(
        "Ask a business question:",
        placeholder="total revenue"
    )

    if st.button("Ask Agent"):

        result = agent.ask(question)

        st.subheader("Answer")

        if isinstance(result, pd.DataFrame):
            st.dataframe(result, use_container_width=True)

        elif isinstance(result, pd.Series):
            st.bar_chart(result)

        else:
            st.success(result)

    # Dashboard
    st.divider()
    st.subheader("📌 Executive Dashboard")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Revenue",
            f"{agent.revenue():,.2f}"
        )

    with c2:
        st.metric(
            "Avg Deal Size",
            f"{agent.avg_deal():,.2f}"
        )

    with c3:
        st.metric(
            "Win Rate",
            f"{agent.win_rate()}%"
        )

    # Insights
    st.divider()
    st.subheader("🧠 Auto Insights")

    st.info(
        f"""
• Revenue = {agent.revenue():,.2f}

• Average Deal Size = {agent.avg_deal():,.2f}

• Win Rate = {agent.win_rate()}%

• Ask deeper questions for more insights.
"""
    )


# -------------------------------------------------
if __name__ == "__main__":
    main()
