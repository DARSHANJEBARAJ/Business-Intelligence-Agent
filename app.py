import streamlit as st
import pandas as pd
import numpy as np

# --------------------------------
# PAGE SETTINGS
# --------------------------------
st.set_page_config(page_title="Business Intelligence Agent", layout="wide")


# --------------------------------
# LOAD DATA
# --------------------------------
@st.cache_data
def load_data():
    deals = pd.read_csv("Deal funnel.csv")
    work_orders = pd.read_csv("work order.csv")
    return deals, work_orders


# --------------------------------
# CLEAN DATA
# --------------------------------
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


# --------------------------------
# FIND IMPORTANT COLUMNS
# --------------------------------
def detect_value_column(df):
    names = [
        "value", "amount", "revenue", "deal_value",
        "price", "sales", "total"
    ]

    for col in names:
        if col in df.columns:
            return col

    nums = df.select_dtypes(include=np.number).columns.tolist()

    if nums:
        return nums[0]

    return None


def detect_status_column(df):
    names = ["status", "stage", "deal_stage"]

    for col in names:
        if col in df.columns:
            return col

    return None


# --------------------------------
# AGENT
# --------------------------------
class BIAgent:

    def __init__(self, deals, work_orders):
        self.deals = deals
        self.work_orders = work_orders

        self.value_col = detect_value_column(deals)
        self.stage_col = detect_status_column(deals)
        self.work_col = detect_status_column(work_orders)

    def total_revenue(self):
        if self.value_col:
            return f"{round(self.deals[self.value_col].sum(),2):,.2f}"
        return "No revenue data"

    def average_deal(self):
        if self.value_col:
            return f"{round(self.deals[self.value_col].mean(),2):,.2f}"
        return "No data"

    def top_deals(self):
        if self.value_col:
            data = self.deals.sort_values(
                by=self.value_col,
                ascending=False
            ).head(5)

            return data

        return "No deals data"

    def pipeline(self):
        if self.stage_col:
            return self.deals[self.stage_col].value_counts()

        return "No pipeline data"

    def work_orders_status(self):
        if self.work_col:
            return self.work_orders[self.work_col].value_counts()

        return "No work order data"

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

    def ask(self, question):

        q = question.lower()

        if "revenue" in q:
            return f"💰 Total Revenue = {self.total_revenue()}"

        elif "average" in q:
            return f"📊 Average Deal Size = {self.average_deal()}"

        elif "top" in q:
            return self.top_deals()

        elif "pipeline" in q:
            return self.pipeline()

        elif "win" in q:
            return f"🏆 Win Rate = {self.win_rate()}%"

        elif "work" in q or "order" in q:
            return self.work_orders_status()

        else:
            return """
Try asking:

• total revenue  
• top deals  
• pipeline status  
• win rate  
• average deal size  
• work order status
"""


# --------------------------------
# MAIN APP
# --------------------------------
def main():

    st.title("📊 Business Intelligence Agent")
    st.write("Ask questions and get answers from your dataset.")

    deals, work_orders = load_data()

    deals = clean_data(deals)
    work_orders = clean_data(work_orders)

    agent = BIAgent(deals, work_orders)

    # Query Input Only
    question = st.text_input("Ask your question:")

    if st.button("Get Answer"):

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
    st.subheader("📌 Quick Insights")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Revenue", agent.total_revenue())

    with c2:
        st.metric("Avg Deal", agent.average_deal())

    with c3:
        st.metric("Win Rate", f"{agent.win_rate()}%")


if __name__ == "__main__":
    main()
