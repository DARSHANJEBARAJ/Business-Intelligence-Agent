import streamlit as st
import pandas as pd


# ----------------------------
# Load Data
# ----------------------------
def load_data():
    try:
        deals = pd.read_csv("Deal funnel.csv")
        work_orders = pd.read_csv("work order.csv")
    except Exception as e:
        st.error(f"Error loading files: {e}")
        st.stop()

    return deals, work_orders


# ----------------------------
# Clean Data
# ----------------------------
def clean_data(df):
    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Fill missing values safely
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("unknown").astype(str).str.strip()
        else:
            df[col] = df[col].fillna(0)

    return df


# ----------------------------
# BI Agent Class
# ----------------------------
class BusinessIntelligenceAgent:
    def __init__(self, deals, work_orders):
        self.deals = deals
        self.work_orders = work_orders

    def total_revenue(self):
        if "value" in self.deals.columns:
            return round(self.deals["value"].sum(), 2)
        return "Revenue data not available"

    def pipeline_health(self):
        if "stage" in self.deals.columns:
            return self.deals["stage"].value_counts().to_dict()
        return "Pipeline data not available"

    def top_deals(self, n=5):
        if "value" in self.deals.columns:
            return self.deals.sort_values(by="value", ascending=False).head(n)
        return "Top deals not available"

    def work_order_summary(self):
        if "status" in self.work_orders.columns:
            return self.work_orders["status"].value_counts().to_dict()
        return "Work order data not available"

    def average_deal_size(self):
        if "value" in self.deals.columns:
            return round(self.deals["value"].mean(), 2)
        return "Not available"

    def win_rate(self):
        if "stage" in self.deals.columns:
            total = len(self.deals)
            won = self.deals[
                self.deals["stage"].astype(str).str.lower().str.contains("won")
            ].shape[0]

            if total > 0:
                return round((won / total) * 100, 2)
        return 0

    def query(self, user_input):
        q = user_input.lower()

        if "revenue" in q:
            return f"Total Revenue: {self.total_revenue()}"

        elif "pipeline" in q:
            return self.pipeline_health()

        elif "top deals" in q:
            return self.top_deals()

        elif "work order" in q or "orders" in q:
            return self.work_order_summary()

        elif "average deal" in q:
            return f"Average Deal Size: {self.average_deal_size()}"

        elif "win rate" in q:
            return f"Win Rate: {self.win_rate()}%"

        else:
            return "Sorry, I couldn't understand the query."


# ----------------------------
# Streamlit App
# ----------------------------
def main():
    st.set_page_config(page_title="Business Intelligence Agent", layout="wide")

    st.title("📊 Business Intelligence Agent")
    st.write("Ask business questions using your CSV data.")

    # Load and clean data
    deals, work_orders = load_data()

    deals = clean_data(deals)
    work_orders = clean_data(work_orders)

    agent = BusinessIntelligenceAgent(deals, work_orders)

    # Sidebar
    st.sidebar.header("Available Queries")
    st.sidebar.write("- total revenue")
    st.sidebar.write("- pipeline status")
    st.sidebar.write("- top deals")
    st.sidebar.write("- work order status")
    st.sidebar.write("- average deal size")
    st.sidebar.write("- win rate")

    # User Input
    user_input = st.text_input("Ask a business question:")

    if st.button("Submit"):
        if user_input.strip() == "":
            st.warning("Please enter a question.")
        else:
            result = agent.query(user_input)

            st.subheader("Result")

            if isinstance(result, pd.DataFrame):
                st.dataframe(result)
            elif isinstance(result, dict):
                st.json(result)
            else:
                st.success(result)


if __name__ == "__main__":
    main()






