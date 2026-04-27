import streamlit as st
import pandas as pd
import numpy as np
import requests
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Skylark Drones – BI Agent",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stChatMessage { border-radius: 12px; }
    div[data-testid="metric-container"] { background:#f0f4fa; padding:12px; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MONDAY.COM CONFIG
# ─────────────────────────────────────────────
MONDAY_API_URL = "https://api.monday.com/v2"

def fetch_monday_board(board_id: str, api_key: str) -> pd.DataFrame:
    query = """
    query ($boardId: [ID!]!) {
      boards(ids: $boardId) {
        name
        columns { id title type }
        items_page(limit: 500) {
          items {
            id
            name
            column_values { id text value }
          }
        }
      }
    }
    """
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json",
        "API-Version": "2024-01",
    }
    try:
        resp = requests.post(
            MONDAY_API_URL,
            json={"query": query, "variables": {"boardId": [str(board_id)]}},
            headers=headers,
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        board = data["data"]["boards"][0]
        columns = {c["id"]: c["title"] for c in board["columns"]}
        rows = []
        for item in board["items_page"]["items"]:
            row = {"name": item["name"]}
            for cv in item["column_values"]:
                col_name = columns.get(cv["id"], cv["id"])
                row[col_name] = cv["text"] or ""
            rows.append(row)
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Monday.com fetch error (board {board_id}): {e}")
        return pd.DataFrame()

# ─────────────────────────────────────────────
# LOAD CSV
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv_data():
    deals, wo = pd.DataFrame(), pd.DataFrame()
    for fname in ["Deal funnel.csv", "Deal_funnel.csv"]:
        try:
            deals = pd.read_csv(fname)
            break
        except FileNotFoundError:
            pass
    for fname in ["work order.csv", "work_order.csv"]:
        try:
            wo = pd.read_csv(fname)
            break
        except FileNotFoundError:
            pass
    return deals, wo

# ─────────────────────────────────────────────
# DATA CLEANING
# ─────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df = df[~df.apply(lambda r: r.astype(str).str.lower().eq("deal name").any(), axis=1)]
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[\s\-/()\.]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("").astype(str).str.strip()
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

def detect(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def coerce_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace(r"[₹,\s]", "", regex=True),
        errors="coerce"
    ).fillna(0)

# ─────────────────────────────────────────────
# CHART HELPERS  (native Streamlit only)
# ─────────────────────────────────────────────
def bar_chart(data: pd.Series, title: str, fmt_fn=None):
    if data.empty:
        st.info("No data available.")
        return
    st.markdown(f"**{title}**")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.bar_chart(data)
    with col2:
        df = data.reset_index()
        df.columns = ["Category", "Value"]
        if fmt_fn:
            df["Value"] = df["Value"].apply(fmt_fn)
        st.dataframe(df, hide_index=True, use_container_width=True)

def pie_table(data: pd.Series, title: str):
    if data.empty:
        st.info("No data available.")
        return
    st.markdown(f"**{title}**")
    total = data.sum()
    col1, col2 = st.columns([2, 1])
    with col1:
        st.bar_chart(data)
    with col2:
        df = data.reset_index()
        df.columns = ["Status", "Count"]
        df["Share %"] = (df["Count"] / total * 100).round(1).astype(str) + "%"
        st.dataframe(df, hide_index=True, use_container_width=True)

# ─────────────────────────────────────────────
# BI AGENT
# ─────────────────────────────────────────────
class BIAgent:
    def __init__(self, deals: pd.DataFrame, wo: pd.DataFrame):
        self.deals = deals
        self.wo    = wo

        self.d_value   = detect(deals, ["masked_deal_value", "value", "amount", "deal_value"])
        self.d_status  = detect(deals, ["deal_status", "status"])
        self.d_stage   = detect(deals, ["deal_stage", "stage"])
        self.d_sector  = detect(deals, ["sector_service", "sector", "industry"])
        self.d_owner   = detect(deals, ["owner_code", "owner", "salesperson"])
        self.d_prob    = detect(deals, ["closure_probability", "probability"])
        self.d_created = detect(deals, ["created_date", "date"])

        self.w_status  = detect(wo, ["execution_status", "exec_status", "status"])
        self.w_sector  = detect(wo, ["sector", "industry"])
        self.w_amount  = detect(wo, ["amount_in_rupees_excl_of_gst_masked",
                                      "amount_in_rupees_excl_of_gst", "amount", "value"])
        self.w_billing = detect(wo, ["billing_status", "invoice_status"])

        if self.d_value and not deals.empty:
            self.deals[self.d_value] = coerce_numeric(deals[self.d_value])
        if self.w_amount and not wo.empty:
            self.wo[self.w_amount] = coerce_numeric(wo[self.w_amount])

    def _open(self):
        if self.d_status:
            return self.deals[self.deals[self.d_status].str.lower().isin(["open", "on hold"])]
        return self.deals

    def _won(self):
        if self.d_status:
            return self.deals[self.deals[self.d_status].str.lower() == "won"]
        return pd.DataFrame()

    def _dead(self):
        if self.d_status:
            return self.deals[self.deals[self.d_status].str.lower() == "dead"]
        return pd.DataFrame()

    def _val(self, df):
        if self.d_value and not df.empty:
            return float(df[self.d_value].sum())
        return 0.0

    def fmt_cr(self, n):
        n = float(n)
        if n >= 1e7:
            return f"Rs.{n/1e7:.2f} Cr"
        elif n >= 1e5:
            return f"Rs.{n/1e5:.2f} L"
        return f"Rs.{n:,.0f}"

    def kpis(self):
        open_d = self._open()
        won_d  = self._won()
        dead_d = self._dead()
        total  = len(self.deals)
        win_r  = round(len(won_d) / total * 100, 1) if total else 0
        return {
            "open_pipeline":  self._val(open_d),
            "won_revenue":    self._val(won_d),
            "open_deals":     len(open_d),
            "won_deals":      len(won_d),
            "dead_deals":     len(dead_d),
            "win_rate":       win_r,
            "avg_deal":       self._val(open_d) / len(open_d) if len(open_d) else 0,
            "total_wo_value": self._val(self.wo) if self.w_amount else 0,
            "total_wo":       len(self.wo),
        }

    def sector_pipeline(self):
        if self.d_sector and self.d_value:
            return self._open().groupby(self.d_sector)[self.d_value].sum().sort_values(ascending=False)
        return pd.Series(dtype=float)

    def stage_funnel(self):
        if self.d_stage and self.d_value:
            return self._open().groupby(self.d_stage)[self.d_value].sum().sort_values(ascending=False)
        return pd.Series(dtype=float)

    def owner_leaderboard(self, n=10):
        if self.d_owner and self.d_value:
            return self._open().groupby(self.d_owner)[self.d_value].sum().sort_values(ascending=False).head(n)
        return pd.Series(dtype=float)

    def wo_exec_status(self):
        if self.w_status:
            return self.wo[self.w_status].value_counts()
        return pd.Series(dtype=int)

    def wo_billing_status(self):
        if self.w_billing:
            return self.wo[self.w_billing].value_counts()
        return pd.Series(dtype=int)

    def high_prob_deals(self):
        if self.d_prob:
            hp = self._open()[self._open()[self.d_prob].str.lower() == "high"]
            if self.d_value:
                return hp.sort_values(by=self.d_value, ascending=False)
            return hp
        return pd.DataFrame()

    def sector_deep_dive(self, sector_kw: str):
        if self.d_sector:
            mask = self.deals[self.d_sector].str.lower().str.contains(sector_kw.lower(), na=False)
            df = self.deals[mask]
            result = {}
            if self.d_status:
                result["open"] = df[df[self.d_status].str.lower() == "open"]
                result["won"]  = df[df[self.d_status].str.lower() == "won"]
            else:
                result["all"] = df
            return result
        return {}

    def top_deals(self, n=10):
        if self.d_value:
            return self.deals.sort_values(by=self.d_value, ascending=False).head(n)
        return self.deals.head(n)

    def monthly_trend(self):
        if self.d_created and self.d_value:
            df = self.deals.copy()
            df["_date"] = pd.to_datetime(df[self.d_created], errors="coerce")
            df = df.dropna(subset=["_date"])
            df["_month"] = df["_date"].dt.to_period("M").astype(str)
            return df.groupby("_month")[self.d_value].sum().sort_index()
        return pd.Series(dtype=float)

    def ar_summary(self):
        ar_col = detect(self.wo, ["amount_receivable_masked", "amount_receivable"])
        if ar_col:
            self.wo[ar_col] = coerce_numeric(self.wo[ar_col])
            total_ar = float(self.wo[ar_col].sum())
            priority_ar = 0.0
            if "ar_priority_account" in self.wo.columns:
                priority_ar = float(self.wo[self.wo["ar_priority_account"] == "Priority"][ar_col].sum())
            return {"total_ar": total_ar, "priority_ar": priority_ar}
        return {}

    # ── SMART QUERY ENGINE ──────────────────────
    def ask(self, question: str) -> dict:
        q = question.lower().strip()

        if any(x in q for x in ["total revenue", "revenue summary", "summary", "overview", "kpi"]):
            k = self.kpis()
            return {
                "type": "kpi_text",
                "text": (
                    f"**Revenue & Pipeline Summary**\n\n"
                    f"- Open Pipeline: **{self.fmt_cr(k['open_pipeline'])}** ({k['open_deals']} deals)\n"
                    f"- Won Revenue: **{self.fmt_cr(k['won_revenue'])}** ({k['won_deals']} deals)\n"
                    f"- Win Rate: **{k['win_rate']}%**\n"
                    f"- Avg Open Deal: **{self.fmt_cr(k['avg_deal'])}**\n"
                    f"- Total Work Order Value: **{self.fmt_cr(k['total_wo_value'])}** ({k['total_wo']} WOs)"
                ),
            }

        for kw in ["energy", "renewables", "mining", "powerline", "railways", "construction", "dsp", "tender"]:
            if kw in q:
                dd = self.sector_deep_dive(kw)
                lines = [f"**{kw.title()} Sector Deep Dive**\n"]
                for label, df in dd.items():
                    val = self._val(df)
                    lines.append(f"- **{label.title()}**: {len(df)} deals · {self.fmt_cr(val)}")
                all_df = (
                    self.deals[self.deals[self.d_sector].str.lower().str.contains(kw, na=False)]
                    if self.d_sector else pd.DataFrame()
                )
                return {"type": "sector", "text": "\n".join(lines), "df": all_df}

        if "sector" in q and any(x in q for x in ["pipeline", "breakdown", "revenue"]):
            return {"type": "chart_bar", "data": self.sector_pipeline(), "title": "Open Pipeline by Sector"}

        if any(x in q for x in ["stage", "funnel"]):
            return {"type": "chart_bar", "data": self.stage_funnel(), "title": "Open Pipeline by Deal Stage"}

        if any(x in q for x in ["top owner", "owner", "best salesperson", "bd", "kam"]):
            return {"type": "chart_bar", "data": self.owner_leaderboard(), "title": "Top Owners – Open Pipeline Value"}

        if "work order" in q or "wo " in q:
            if "billing" in q:
                return {"type": "chart_pie", "data": self.wo_billing_status(), "title": "Work Order Billing Status"}
            return {"type": "chart_pie", "data": self.wo_exec_status(), "title": "Work Order Execution Status"}

        if any(x in q for x in ["high prob", "high probability", "hot deal"]):
            df = self.high_prob_deals()
            return {"type": "dataframe", "df": df, "title": f"High Probability Deals ({len(df)})"}

        if "top" in q and "deal" in q:
            n = next((int(w) for w in q.split() if w.isdigit()), 10)
            return {"type": "dataframe", "df": self.top_deals(n), "title": f"Top {n} Deals by Value"}

        if any(x in q for x in ["win rate", "win/loss", "lost deal"]):
            k = self.kpis()
            data = pd.Series({"Won": k["won_deals"], "Open": k["open_deals"], "Dead/Lost": k["dead_deals"]})
            return {"type": "chart_pie", "data": data, "title": f"Deal Status (Win Rate: {k['win_rate']}%)"}

        if any(x in q for x in ["monthly", "trend", "over time"]):
            data = self.monthly_trend()
            return {"type": "chart_line", "data": data, "title": "Monthly Deal Value Created"}

        if "billing" in q:
            return {"type": "chart_pie", "data": self.wo_billing_status(), "title": "Billing Status"}

        if any(x in q for x in ["receivable", "ar ", "outstanding", "collection"]):
            ar = self.ar_summary()
            if ar:
                return {
                    "type": "kpi_text",
                    "text": (
                        f"**Accounts Receivable**\n\n"
                        f"- Total AR: **{self.fmt_cr(ar['total_ar'])}**\n"
                        f"- Priority AR: **{self.fmt_cr(ar['priority_ar'])}**"
                    ),
                }
            return {"type": "kpi_text", "text": "AR column not found in work orders dataset."}

        return {
            "type": "kpi_text",
            "text": (
                "**Try asking:**\n\n"
                "- `total revenue` / `kpi summary`\n"
                "- `pipeline by sector` / `stage funnel`\n"
                "- `renewables` / `mining` / `railways` / `powerline`\n"
                "- `top owners`\n"
                "- `work order status` / `work order billing`\n"
                "- `high probability deals`\n"
                "- `top 10 deals`\n"
                "- `win rate` / `monthly trend`\n"
                "- `accounts receivable`"
            ),
        }


# ─────────────────────────────────────────────
# RENDER RESPONSE
# ─────────────────────────────────────────────
def render_response(agent: BIAgent, result: dict):
    rtype = result.get("type", "kpi_text")

    if rtype in ("kpi_text", "text"):
        st.markdown(result.get("text", ""))

    elif rtype == "sector":
        st.markdown(result.get("text", ""))
        df = result.get("df", pd.DataFrame())
        if not df.empty:
            st.dataframe(df.head(30), use_container_width=True, hide_index=True)

    elif rtype == "dataframe":
        st.markdown(f"**{result.get('title', '')}**")
        df = result.get("df", pd.DataFrame())
        if not df.empty:
            st.dataframe(df.head(20), use_container_width=True, hide_index=True)
        else:
            st.info("No records found.")

    elif rtype == "chart_bar":
        bar_chart(result.get("data", pd.Series(dtype=float)), result.get("title", ""), agent.fmt_cr)

    elif rtype == "chart_pie":
        pie_table(result.get("data", pd.Series(dtype=int)), result.get("title", ""))

    elif rtype == "chart_line":
        data = result.get("data", pd.Series(dtype=float))
        title = result.get("title", "")
        if data.empty:
            st.info("No data.")
            return
        st.markdown(f"**{title}**")
        df = data.reset_index()
        df.columns = ["Month", "Value"]
        st.line_chart(df.set_index("Month")["Value"])


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.title("Skylark Drones BI")
        st.divider()
        source = st.radio("Data Source", ["CSV Files (offline)", "Monday.com (live)"], index=0)

        api_key, deals_board, wo_board = "", "", ""
        if source == "Monday.com (live)":
            api_key     = st.text_input("Monday.com API Key", type="password")
            deals_board = st.text_input("Deals Board ID",      placeholder="e.g. 1234567890")
            wo_board    = st.text_input("Work Orders Board ID", placeholder="e.g. 9876543210")
            st.session_state["monday_api_key"] = api_key
            if api_key and deals_board and wo_board:
                st.success("Ready — fetching live from Monday.com")
            else:
                st.warning("Enter API key + both Board IDs")

        st.divider()
        st.markdown("**Quick Queries**")
        quick = [
            "total revenue",
            "pipeline by sector",
            "energy sector pipeline",
            "renewables deals",
            "mining sector",
            "railways pipeline",
            "powerline sector",
            "top owners",
            "work order status",
            "work order billing",
            "high probability deals",
            "top 10 deals",
            "win rate",
            "monthly trend",
            "accounts receivable",
        ]
        for q in quick:
            if st.button(q, use_container_width=True):
                st.session_state["prefill_q"] = q

        st.divider()
        if st.button("Reload Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    return source, api_key, deals_board, wo_board


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    source, api_key, deals_board, wo_board = sidebar()

    st.title("Skylark Drones — Business Intelligence Agent")
    st.caption("Founder-level answers from Work Orders & Deal Funnel · Monday.com integrated")

    deals_raw = pd.DataFrame()
    wo_raw    = pd.DataFrame()

    if source == "Monday.com (live)" and api_key and deals_board and wo_board:
        with st.spinner("Fetching from Monday.com…"):
            deals_raw = fetch_monday_board(deals_board, api_key)
            wo_raw    = fetch_monday_board(wo_board, api_key)
        if deals_raw.empty and wo_raw.empty:
            st.warning("Monday.com returned empty data. Check Board IDs and API key.")
    else:
        deals_raw, wo_raw = load_csv_data()
        if deals_raw.empty:
            st.info("Place 'Deal funnel.csv' and 'work order.csv' in the same folder as app.py, or use Monday.com source.")

    deals = clean(deals_raw)
    wo    = clean(wo_raw)

    if deals.empty and wo.empty:
        st.warning("No data loaded. Add CSV files or connect Monday.com.")
        st.stop()

    agent = BIAgent(deals, wo)
    k = agent.kpis()

    # KPI row
    st.subheader("Executive Dashboard")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Open Pipeline",  agent.fmt_cr(k["open_pipeline"]))
    c2.metric("Won Revenue",    agent.fmt_cr(k["won_revenue"]))
    c3.metric("Win Rate",       f"{k['win_rate']}%")
    c4.metric("Avg Deal Size",  agent.fmt_cr(k["avg_deal"]))
    c5.metric("Work Orders",    k["total_wo"])
    c6.metric("WO Value",       agent.fmt_cr(k["total_wo_value"]))

    st.divider()

    # Auto charts
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Sector Pipeline (Open Deals)")
        s_data = agent.sector_pipeline()
        if not s_data.empty:
            st.bar_chart(s_data)
            df_s = s_data.reset_index()
            df_s.columns = ["Sector", "Value"]
            df_s["Value"] = df_s["Value"].apply(agent.fmt_cr)
            st.dataframe(df_s, hide_index=True, use_container_width=True)

    with col_r:
        st.subheader("Work Order Execution Status")
        wo_data = agent.wo_exec_status()
        if not wo_data.empty:
            st.bar_chart(wo_data)
            total_wo = wo_data.sum()
            df_wo = wo_data.reset_index()
            df_wo.columns = ["Status", "Count"]
            df_wo["Share"] = (df_wo["Count"] / total_wo * 100).round(1).astype(str) + "%"
            st.dataframe(df_wo, hide_index=True, use_container_width=True)

    st.divider()

    # Chat agent
    st.subheader("Ask the BI Agent")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and isinstance(msg["content"], dict):
                render_response(agent, msg["content"])
            else:
                st.markdown(msg["content"])

    prefill    = st.session_state.pop("prefill_q", None)
    user_input = st.chat_input("Ask a business question…")
    if prefill:
        user_input = prefill

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        result = agent.ask(user_input)
        st.session_state.messages.append({"role": "assistant", "content": result})
        with st.chat_message("assistant"):
            render_response(agent, result)

    with st.expander("Raw Data Explorer"):
        tab1, tab2 = st.tabs(["Deal Funnel", "Work Orders"])
        with tab1:
            st.dataframe(deals, use_container_width=True, hide_index=True)
        with tab2:
            st.dataframe(wo, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
