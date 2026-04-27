import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, date
import plotly.express as px
import plotly.graph_objects as go

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
    .metric-card {
        background: #f0f4fa;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MONDAY.COM CONFIG  (set via env or sidebar)
# ─────────────────────────────────────────────
MONDAY_API_URL = "https://api.monday.com/v2"

def get_api_key():
    return st.session_state.get("monday_api_key", os.environ.get("MONDAY_API_KEY", ""))

# ─────────────────────────────────────────────
# MONDAY.COM DATA FETCHER
# ─────────────────────────────────────────────
def fetch_monday_board(board_id: str, api_key: str) -> pd.DataFrame:
    """Fetch all items + column values from a Monday.com board."""
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
# FALLBACK: LOAD FROM CSV
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
    # Remove duplicate header rows
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

def detect(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(r"[₹,\s]", "", regex=True), errors="coerce").fillna(0)

# ─────────────────────────────────────────────
# BI AGENT CLASS
# ─────────────────────────────────────────────
class BIAgent:
    def __init__(self, deals: pd.DataFrame, wo: pd.DataFrame):
        self.deals = deals
        self.wo    = wo

        # --- deals column mapping ---
        self.d_value   = detect(deals, ["masked_deal_value", "value", "amount", "deal_value"])
        self.d_status  = detect(deals, ["deal_status", "status"])
        self.d_stage   = detect(deals, ["deal_stage", "stage"])
        self.d_sector  = detect(deals, ["sector_service", "sector/service", "sector", "industry"])
        self.d_owner   = detect(deals, ["owner_code", "owner", "salesperson"])
        self.d_prob    = detect(deals, ["closure_probability", "probability"])
        self.d_created = detect(deals, ["created_date", "date"])
        self.d_close   = detect(deals, ["tentative_close_date", "close_date_a", "close_date"])

        # --- work-order column mapping ---
        self.w_status  = detect(wo, ["execution_status", "exec_status", "status"])
        self.w_sector  = detect(wo, ["sector", "industry"])
        self.w_amount  = detect(wo, ["amount_in_rupees_excl_of_gst_masked",
                                      "amount_in_rupees_excl_of_gst",
                                      "amount", "value"])
        self.w_billing = detect(wo, ["billing_status", "invoice_status"])
        self.w_collect = detect(wo, ["collection_status", "wo_status_billed"])
        self.w_wo_stat = detect(wo, ["wo_status_billed", "wo_status"])
        self.w_type    = detect(wo, ["type_of_work", "nature_of_work"])

        # Coerce value columns
        if self.d_value and not deals.empty:
            self.deals[self.d_value] = coerce_numeric(deals[self.d_value])
        if self.w_amount and not wo.empty:
            self.wo[self.w_amount] = coerce_numeric(wo[self.w_amount])

    # ── helpers ──────────────────────────────
    def _open(self):
        if self.d_status:
            return self.deals[self.deals[self.d_status].str.lower().isin(["open","on hold"])]
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
            return df[self.d_value].sum()
        return 0

    def fmt_cr(self, n):
        if n >= 1e7:
            return f"₹{n/1e7:.2f} Cr"
        elif n >= 1e5:
            return f"₹{n/1e5:.2f} L"
        return f"₹{n:,.0f}"

    # ── KPIs ─────────────────────────────────
    def kpis(self):
        open_d  = self._open()
        won_d   = self._won()
        dead_d  = self._dead()
        total   = len(self.deals)
        win_r   = round(len(won_d)/total*100, 1) if total else 0
        return {
            "open_pipeline":   self._val(open_d),
            "won_revenue":     self._val(won_d),
            "open_deals":      len(open_d),
            "won_deals":       len(won_d),
            "dead_deals":      len(dead_d),
            "win_rate":        win_r,
            "avg_deal":        self._val(open_d) / len(open_d) if len(open_d) else 0,
            "total_wo_value":  self._val(self.wo) if self.w_amount else 0,
            "total_wo":        len(self.wo),
        }

    # ── sector pipeline chart data ────────────
    def sector_pipeline(self, status_filter="Open"):
        if self.d_sector and self.d_value:
            df = self._open() if status_filter == "Open" else self._won()
            return df.groupby(self.d_sector)[self.d_value].sum().sort_values(ascending=False)
        return pd.Series()

    # ── stage funnel ─────────────────────────
    def stage_funnel(self):
        if self.d_stage and self.d_value:
            return self._open().groupby(self.d_stage)[self.d_value].sum().sort_values(ascending=False)
        return pd.Series()

    # ── owner leaderboard ─────────────────────
    def owner_leaderboard(self, n=10):
        if self.d_owner and self.d_value:
            return self._open().groupby(self.d_owner)[self.d_value].sum().sort_values(ascending=False).head(n)
        return pd.Series()

    # ── wo status ────────────────────────────
    def wo_exec_status(self):
        if self.w_status:
            return self.wo[self.w_status].value_counts()
        return pd.Series()

    def wo_billing_status(self):
        if self.w_billing:
            return self.wo[self.w_billing].value_counts()
        return pd.Series()

    # ── high-prob deals ───────────────────────
    def high_prob_deals(self):
        if self.d_prob:
            return self._open()[self._open()[self.d_prob].str.lower() == "high"].sort_values(
                by=self.d_value, ascending=False
            ) if self.d_value else self._open()[self._open()[self.d_prob].str.lower() == "high"]
        return pd.DataFrame()

    # ── sector filter ─────────────────────────
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

    # ── top N deals ───────────────────────────
    def top_deals(self, n=10, status=None):
        df = self.deals.copy()
        if status and self.d_status:
            df = df[df[self.d_status].str.lower() == status.lower()]
        if self.d_value:
            return df.sort_values(by=self.d_value, ascending=False).head(n)
        return df.head(n)

    # ── monthly trend ─────────────────────────
    def monthly_trend(self):
        if self.d_created and self.d_value:
            df = self.deals.copy()
            df["_date"] = pd.to_datetime(df[self.d_created], errors="coerce")
            df = df.dropna(subset=["_date"])
            df["_month"] = df["_date"].dt.to_period("M").astype(str)
            return df.groupby("_month")[self.d_value].sum().sort_index()
        return pd.Series()

    # ── receivables / AR ─────────────────────
    def ar_summary(self):
        ar_col = detect(self.wo, ["amount_receivable_masked", "amount_receivable"])
        if ar_col:
            self.wo[ar_col] = coerce_numeric(self.wo[ar_col])
            total_ar = self.wo[ar_col].sum()
            priority = self.wo[self.wo.get("ar_priority_account", "ar_priority_account") == "Priority"][ar_col].sum() if "ar_priority_account" in self.wo.columns else 0
            return {"total_ar": total_ar, "priority_ar": priority}
        return {}

    # ─────────────────────────────────────────
    # SMART QUERY ENGINE
    # ─────────────────────────────────────────
    def ask(self, question: str):
        q = question.lower().strip()

        # ── revenue / summary ──────────────────
        if any(x in q for x in ["total revenue", "revenue summary", "summary", "overview", "kpi"]):
            k = self.kpis()
            return {
                "type": "kpi",
                "text": (
                    f"**Revenue & Pipeline Summary**\n\n"
                    f"- Open Pipeline: **{self.fmt_cr(k['open_pipeline'])}** ({k['open_deals']} deals)\n"
                    f"- Won Revenue: **{self.fmt_cr(k['won_revenue'])}** ({k['won_deals']} deals)\n"
                    f"- Win Rate: **{k['win_rate']}%**\n"
                    f"- Avg Open Deal: **{self.fmt_cr(k['avg_deal'])}**\n"
                    f"- Total Work Order Value: **{self.fmt_cr(k['total_wo_value'])}** ({k['total_wo']} WOs)"
                ),
            }

        # ── energy / specific sector ───────────
        for kw in ["energy", "renewables", "mining", "powerline", "railways", "construction", "dsp"]:
            if kw in q:
                dd = self.sector_deep_dive(kw)
                parts = []
                for label, df in dd.items():
                    if df.empty:
                        continue
                    val = self._val(df)
                    parts.append(f"**{label.title()} ({len(df)} deals):** {self.fmt_cr(val)}")
                    if not df.empty and self.d_stage:
                        top_stage = df[self.d_stage].value_counts().idxmax() if self.d_stage in df else "-"
                        parts.append(f"  Top Stage: {top_stage}")
                text = f"**{kw.title()} Sector Deep Dive**\n\n" + "\n".join(parts) if parts else f"No data found for sector: {kw}"
                display_df = self.deals[self.deals[self.d_sector].str.lower().str.contains(kw, na=False)] if self.d_sector else pd.DataFrame()
                return {"type": "sector", "text": text, "df": display_df}

        # ── pipeline by sector ─────────────────
        if "sector" in q and any(x in q for x in ["pipeline", "breakdown", "revenue"]):
            data = self.sector_pipeline()
            return {"type": "chart_bar", "data": data, "title": "Open Pipeline by Sector"}

        # ── stage funnel ──────────────────────
        if any(x in q for x in ["stage", "funnel", "pipeline stage"]):
            data = self.stage_funnel()
            return {"type": "chart_bar", "data": data, "title": "Open Pipeline by Stage"}

        # ── top owners ────────────────────────
        if any(x in q for x in ["top owner", "best salesperson", "owner", "bd", "kam"]):
            data = self.owner_leaderboard()
            return {"type": "chart_bar", "data": data, "title": "Top Owners by Open Pipeline Value"}

        # ── work orders ───────────────────────
        if "work order" in q or "wo" in q:
            if "billing" in q:
                data = self.wo_billing_status()
                return {"type": "chart_pie", "data": data, "title": "Work Order Billing Status"}
            data = self.wo_exec_status()
            return {"type": "chart_pie", "data": data, "title": "Work Order Execution Status"}

        # ── high probability ──────────────────
        if any(x in q for x in ["high prob", "high probability", "likely to close", "hot deals"]):
            df = self.high_prob_deals()
            return {"type": "dataframe", "df": df, "title": f"High Probability Deals ({len(df)} found)"}

        # ── top deals ─────────────────────────
        if "top" in q and "deal" in q:
            n = 10
            for word in q.split():
                if word.isdigit():
                    n = int(word)
            df = self.top_deals(n=n)
            return {"type": "dataframe", "df": df, "title": f"Top {n} Deals by Value"}

        # ── win rate ──────────────────────────
        if "win rate" in q or "win/loss" in q or "lost" in q:
            k = self.kpis()
            won, dead, open_ = k["won_deals"], k["dead_deals"], k["open_deals"]
            total = won + dead + open_
            data = pd.Series({"Won": won, "Open": open_, "Dead/Lost": dead})
            return {"type": "chart_pie", "data": data, "title": f"Deal Status Distribution (Win Rate: {k['win_rate']}%)"}

        # ── monthly trend ────────────────────
        if any(x in q for x in ["monthly", "trend", "over time", "month"]):
            data = self.monthly_trend()
            return {"type": "chart_line", "data": data, "title": "Monthly Deal Value Created"}

        # ── billing status ────────────────────
        if "billing" in q:
            data = self.wo_billing_status()
            return {"type": "chart_pie", "data": data, "title": "Billing Status"}

        # ── AR / receivables ──────────────────
        if any(x in q for x in ["receivable", "ar ", "outstanding", "collection"]):
            ar = self.ar_summary()
            if ar:
                return {
                    "type": "text",
                    "text": (
                        f"**Accounts Receivable Summary**\n\n"
                        f"- Total AR: **{self.fmt_cr(ar.get('total_ar', 0))}**\n"
                        f"- Priority AR: **{self.fmt_cr(ar.get('priority_ar', 0))}**"
                    ),
                }
            return {"type": "text", "text": "AR data not detected in work orders."}

        # ── default ───────────────────────────
        return {
            "type": "text",
            "text": (
                "I can answer questions like:\n\n"
                "- `total revenue` / `kpi summary`\n"
                "- `pipeline by sector` / `stage funnel`\n"
                "- `energy sector` / `renewables` / `mining` / `railways`\n"
                "- `top owners` / `owner leaderboard`\n"
                "- `work order status` / `billing status`\n"
                "- `high probability deals`\n"
                "- `top 10 deals`\n"
                "- `win rate` / `monthly trend`\n"
                "- `accounts receivable`"
            ),
        }


# ─────────────────────────────────────────────
# RENDER AGENT RESPONSE
# ─────────────────────────────────────────────
def render_response(agent: BIAgent, result: dict):
    rtype = result.get("type", "text")

    if rtype in ("text", "kpi", "sector"):
        st.markdown(result.get("text", ""))
        if "df" in result and not result["df"].empty:
            cols_to_show = [c for c in result["df"].columns if c not in ("",)][:12]
            st.dataframe(result["df"][cols_to_show].head(20), use_container_width=True)

    elif rtype == "dataframe":
        st.markdown(f"**{result.get('title','')}**")
        df = result.get("df", pd.DataFrame())
        if not df.empty:
            st.dataframe(df.head(20), use_container_width=True)
        else:
            st.info("No data found.")

    elif rtype == "chart_bar":
        data: pd.Series = result.get("data", pd.Series())
        title = result.get("title", "")
        if data.empty:
            st.info("No data available.")
            return
        df_plot = data.reset_index()
        df_plot.columns = ["Category", "Value"]
        df_plot["Label"] = df_plot["Value"].apply(agent.fmt_cr)
        fig = px.bar(
            df_plot, x="Value", y="Category", orientation="h",
            text="Label", title=title,
            color="Value", color_continuous_scale="Blues",
        )
        fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"), height=max(300, len(df_plot)*40+100))
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    elif rtype == "chart_pie":
        data: pd.Series = result.get("data", pd.Series())
        title = result.get("title", "")
        if data.empty:
            st.info("No data available.")
            return
        fig = px.pie(values=data.values, names=data.index, title=title, hole=0.4)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    elif rtype == "chart_line":
        data: pd.Series = result.get("data", pd.Series())
        title = result.get("title", "")
        if data.empty:
            st.info("No data available.")
            return
        df_plot = data.reset_index()
        df_plot.columns = ["Month", "Value"]
        fig = px.line(df_plot, x="Month", y="Value", title=title, markers=True)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.title("🚁 Skylark Drones BI")
        st.divider()

        source = st.radio("Data Source", ["CSV Files (offline)", "Monday.com (live)"], index=0)

        api_key, deals_board, wo_board = "", "", ""
        if source == "Monday.com (live)":
            api_key = st.text_input("Monday.com API Key", type="password",
                                     help="Your personal Monday.com API token")
            deals_board = st.text_input("Deals Board ID", placeholder="e.g. 1234567890")
            wo_board    = st.text_input("Work Orders Board ID", placeholder="e.g. 9876543210")
            st.session_state["monday_api_key"] = api_key
            if api_key and deals_board and wo_board:
                st.success("Ready to fetch from Monday.com")
            else:
                st.warning("Fill API key + both Board IDs")

        st.divider()
        st.markdown("**Quick Queries**")
        quick = [
            "total revenue",
            "pipeline by sector",
            "energy sector pipeline",
            "renewables deals",
            "mining sector",
            "railways pipeline",
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
        if st.button("🔄 Reload Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    return source, api_key, deals_board, wo_board


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    source, api_key, deals_board, wo_board = sidebar()

    st.title("📊 Skylark Drones — Business Intelligence Agent")
    st.caption("Founder-level answers from Work Orders & Deal Funnel data")

    # ── Load data ─────────────────────────────
    deals_raw = pd.DataFrame()
    wo_raw    = pd.DataFrame()

    if source == "Monday.com (live)" and api_key and deals_board and wo_board:
        with st.spinner("Fetching from Monday.com…"):
            deals_raw = fetch_monday_board(deals_board, api_key)
            wo_raw    = fetch_monday_board(wo_board, api_key)
        if deals_raw.empty and wo_raw.empty:
            st.warning("Monday.com returned empty data. Check board IDs / API key.")
    else:
        deals_raw, wo_raw = load_csv_data()
        if deals_raw.empty:
            st.warning("CSV not found. Place 'Deal funnel.csv' and 'work order.csv' in the app folder, or switch to Monday.com source.")

    deals = clean(deals_raw)
    wo    = clean(wo_raw)

    if deals.empty and wo.empty:
        st.stop()

    agent = BIAgent(deals, wo)
    k = agent.kpis()

    # ── KPI Dashboard ─────────────────────────
    st.subheader("📌 Executive Dashboard")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Open Pipeline",   agent.fmt_cr(k["open_pipeline"]))
    c2.metric("Won Revenue",     agent.fmt_cr(k["won_revenue"]))
    c3.metric("Win Rate",        f"{k['win_rate']}%")
    c4.metric("Avg Deal Size",   agent.fmt_cr(k["avg_deal"]))
    c5.metric("Work Orders",     k["total_wo"])
    c6.metric("WO Value",        agent.fmt_cr(k["total_wo_value"]))

    st.divider()

    # ── Auto Insights Row ─────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("🏭 Sector Pipeline")
        s_data = agent.sector_pipeline()
        if not s_data.empty:
            df_s = s_data.reset_index()
            df_s.columns = ["Sector", "Value"]
            df_s["Label"] = df_s["Value"].apply(agent.fmt_cr)
            fig = px.bar(df_s, x="Value", y="Sector", orientation="h",
                         text="Label", color="Value", color_continuous_scale="Blues")
            fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"),
                              height=max(300, len(df_s)*40+80), margin=dict(l=0,r=0,t=20,b=0))
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("⚙️ Work Order Status")
        wo_data = agent.wo_exec_status()
        if not wo_data.empty:
            fig2 = px.pie(values=wo_data.values, names=wo_data.index, hole=0.45)
            fig2.update_traces(textposition="inside", textinfo="percent+label")
            fig2.update_layout(height=340, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Conversational Agent ─────────────────
    st.subheader("🤖 Ask the BI Agent")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and isinstance(msg["content"], dict):
                render_response(agent, msg["content"])
            else:
                st.markdown(msg["content"])

    # Prefill from sidebar quick-query
    prefill = st.session_state.pop("prefill_q", None)

    user_input = st.chat_input("Ask a business question…", key="chat_input")
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

    # ── Data Explorer ────────────────────────
    with st.expander("🔍 Raw Data Explorer"):
        tab1, tab2 = st.tabs(["Deal Funnel", "Work Orders"])
        with tab1:
            st.dataframe(deals, use_container_width=True)
        with tab2:
            st.dataframe(wo, use_container_width=True)


if __name__ == "__main__":
    main()
