import pandas as pd

def load_data():
    try:
        deals = pd.read_excel("Deal_funnel_data.xlsx")
        work_orders = pd.read_excel("Work_Order_Tracker_Data.xlsx")
    except Exception:
        deals = pd.read_csv("Deal_funnel_Data.csv")
        work_orders = pd.read_csv("Work_Order_Tracker_Data.csv")
    return deals, work_orders


def clean_data(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()
    
    df.fillna("unknown", inplace=True)
    return df


class BusinessIntelligenceAgent:
    def __init__(self, deals, work_orders):
        self.deals = deals
        self.work_orders = work_orders

    def total_revenue(self):
        if 'value' in self.deals.columns:
            return round(self.deals['value'].sum(), 2)
        return "Revenue data not available"

    def pipeline_health(self):
        if 'stage' in self.deals.columns:
            return self.deals['stage'].value_counts().to_dict()
        return "Pipeline data not available"

    def top_deals(self, n=5):
        if 'value' in self.deals.columns:
            return self.deals.sort_values(by='value', ascending=False).head(n)
        return "Top deals not available"

    def work_order_summary(self):
        if 'status' in self.work_orders.columns:
            return self.work_orders['status'].value_counts().to_dict()
        return "Work order data not available"

    def average_deal_size(self):
        if 'value' in self.deals.columns:
            return round(self.deals['value'].mean(), 2)
        return "Not available"

    def win_rate(self):
        if 'stage' in self.deals.columns:
            total = len(self.deals)
            won = self.deals[self.deals['stage'].str.lower().str.contains("won")].shape[0]
            return round((won / total) * 100, 2) if total > 0 else 0
        return "Not available"

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


def main():
    print("Initializing BI Agent...\n")

    deals, work_orders = load_data()

    deals = clean_data(deals)
    work_orders = clean_data(work_orders)

    agent = BusinessIntelligenceAgent(deals, work_orders)

    print("Agent Ready")
    print("\nYou can ask:")
    print("- total revenue")
    print("- pipeline status")
    print("- top deals")
    print("- work order status")
    print("- average deal size")
    print("- win rate\n")

    while True:
        user_input = input("Ask a question (type 'exit' to quit): ")

        if user_input.lower() == "exit":
            print("Exiting BI Agent")
            break

        result = agent.query(user_input)
        print("\nResult:")
        print(result)
        print("-" * 50)


if __name__ == "__main__":
    main()
