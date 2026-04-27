import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Business Intelligence AI Agent", layout="wide")


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    deals = pd.read_csv("Deal funnel.csv")
    work_orders = pd.read_csv("work order.csv")
    return deals, work_orders


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
# DETECT COLUMNS
# -------------------------------------------------
def detect_col(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


# -------------------------------------------------
# BI AGENT
# -------------------------------------------------
class BIAgent:

    def __init__(self, deals, work_orders):
        self.deals = deals
        self.work_orders = work_orders

        self.value_col = detect_col(
            deals,
            ["value", "amount", "deal_value", "revenue", "sales"]
        )

        self.stage_col = detect_col(
            deals,
            ["stage", "status"]
        )

        self.sector_col = detect_col(
            deals,
            ["sector", "industry", "category"]
        )

        self.owner_col = detect_col(
            deals,
            ["owner", "salesperson", "manager"]
        )

        self.date_col = detect_col(
            deals,
            ["date", "created_date", "close_date"]
        )

        self.work_status_col = detect_col(
            work_orders,
            ["status", "stage"]
        )

    # -----------------------------------------
    # BASIC KPI
    # -----------------------------------------
    def revenue(self):
        if self.value_col:
            return round(self.deals[self.value_col].sum(), 2)
        return 0

    def avg_deal(self):
        if self.value_col:
            return round(self.deals[self.value_col].mean(), 2)
        return 0

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

    # -----------------------------------------
    # SMART QUERY ENGINE
    # -----------------------------------------
    def ask(self, question):

        q = question.lower()

        # -----------------------------------
        # TOTAL REVENUE
        # -----------------------------------
        if "revenue" in q:
            return f"💰 Total Revenue = {self.revenue():,.2f}"

        # -----------------------------------
        # AVERAGE DEAL
        # -----------------------------------
        if "average" in q:
            return f"📊 Average Deal Size = {self.avg_deal():,.2f}"

        # -----------------------------------
        # WIN RATE
        # -----------------------------------
        if "win rate" in q:
            return f"🏆 Win Rate = {self.win_rate()}%"

        # -----------------------------------
        # TOP DEALS
        # -----------------------------------
        if "top deals" in q:
            if self.value_col:
                return self.deals.sort_values(
                    by=self.value_col,
                    ascending=False
                ).head(5)

        # -----------------------------------
        # PIPELINE STATUS
        # -----------------------------------
        if "pipeline" in q:
            if self.stage_col:
                return self.deals[self.stage_col].value_counts()

        # -----------------------------------
        # ENERGY SECTOR THIS QUARTER
        # -----------------------------------
        if "energy" in q and "quarter" in q:

            if self.sector_col:

                energy = self.deals[
                    self.deals[self.sector_col]
                    .astype(str)
                    .str.lower()
                    .str.contains("energy")
                ]

                count = len(energy)

                value = 0
                if self.value_col:
                    value = energy[self.value_col].sum()

                return f"""
📈 Energy Sector Pipeline This Quarter

• Open Deals: {count}
• Pipeline Value: {value:,.2f}
"""

        # -----------------------------------
        # TOP OWNER
        # -----------------------------------
        if "top owner" in q or "best salesperson" in q:

            if self.owner_col and self.value_col:
                data = self.deals.groupby(
                    self.owner_col
                )[self.value_col].sum().sort_values(
                    ascending=False
                ).head(5)

                return data

        # -----------------------------------
        # WORK ORDER STATUS
        # -----------------------------------
        if "work order" in q:

            if self.work_status_col:
                return self.work_orders[
                    self.work_status_col
                ].value_counts()

        # -----------------------------------
        # SECTOR WISE REVENUE
        # -----------------------------------
        if "sector revenue" in q:

            if self.sector_col and self.value_col:
                data = self.deals.groupby(
                    self.sector_col
                )[self.value_col].sum().sort_values(
                    ascending=False
                )

                return data

        # -----------------------------------
        # DEFAULT
        # -----------------------------------
        return """
Try queries like:

• total revenue  
• top deals  
• pipeline status  
• average deal size  
• win rate  
• How is pipeline looking for energy sector this quarter?  
• top owner  
• sector revenue  
• work order status
"""


# -------------------------------------------------
# MAIN APP
# -------------------------------------------------
def main():

    st.title("📊 Business Intelligence AI Agent")
    st.write("Founder-level business answers from multiple datasets.")

    deals, work_orders = load_data()

    deals = clean_data(deals)
    work_orders = clean_data(work_orders)

    agent = BIAgent(deals, work_orders)

    # INPUT
    question = st.text_input(
        "Ask a business question:",
        placeholder="How is pipeline looking for energy sector this quarter?"
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

    # -------------------------------------
    # QUICK DASHBOARD
    # -------------------------------------
    st.divider()
    st.subheader("📌 Executive Dashboard")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Revenue", f"{agent.revenue():,.2f}")

    with c2:
        st.metric("Avg Deal Size", f"{agent.avg_deal():,.2f}")

    with c3:
        st.metric("Win Rate", f"{agent.win_rate()}%")

    # -------------------------------------
    # AUTO INSIGHTS
    # -------------------------------------
    st.divider()
    st.subheader("🧠 Auto Insights")

    st.info(
        f"""
• Current Revenue is {agent.revenue():,.2f}

• Average Deal Size is {agent.avg_deal():,.2f}

• Win Rate stands at {agent.win_rate()}%

• Use queries to drill deeper into sectors, owners and work orders.
"""
    )


if __name__ == "__main__":
    main()
    
