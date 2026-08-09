import io
import json
import pandas as pd
import streamlit as st
import services.database as database
import services
import config

# Set up page configurations
st.set_page_config(
    page_title="MarketPulse AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Run database migrations on startup
try:
    database.run_migrations()
except Exception as e:
    st.error(f"Failed to execute database migrations: {e}")

# ==========================================
# SIDEBAR NAVIGATION & HEALTH STATUS
# ==========================================

st.sidebar.title("MarketPulse AI 📊")
st.sidebar.markdown("**Unified Portfolio RAG Sentinel**")
st.sidebar.markdown("---")

# Page Routing Radio Menu
selected_page = st.sidebar.radio(
    "Navigation Menu",
    [
        "💼 Portfolio Positions",
        "💬 AI Research Chatbot",
        "📰 Market News Portal",
        "🛠️ Developer Diagnostics"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("⚠️ **Disclaimer**: Educational simulation only. Non-custodial research support. No broker connections.")

# ==========================================
# PAGE RENDERERS
# ==========================================

def render_portfolio_page():
    st.header("💼 Portfolio Positions Manager")
    st.markdown("Manage stock holdings, quantities, and cost basis in CockroachDB Serverless utilizing ACID transactions.")
    
    # Safely initialize the background price scheduler daemon
    services.initialize_polling_scheduler()
    
    # Calculate portfolio performance metrics
    metrics = services.calculate_performance_metrics()
    
    st.markdown("---")
    
    # 1. Metric Cards Summary Header
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric(
        label="Total Portfolio Value",
        value=f"${metrics['total_value']:,.2f}"
    )
    metric_col2.metric(
        label="Daily Portfolio Change",
        value=f"${metrics['daily_change']:+,.2f}",
        delta=f"{metrics['daily_change_pct']:+.2f}%"
    )
    metric_col3.metric(
        label="Total Gain / Loss",
        value=f"${metrics['total_gain_loss']:+,.2f}",
        delta=f"{metrics['total_gain_loss_pct']:+.2f}%"
    )
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("➕ Add Asset Position")
        with st.form("add_asset_form", clear_on_submit=True):
            ticker_input = st.text_input("Stock Ticker", placeholder="e.g. MSFT, AAPL, NVDA").upper().strip()
            shares_input = st.number_input("Shares", min_value=0.0001, step=1.0, format="%.4f")
            cost_input = st.number_input("Average Cost Basis ($)", min_value=0.01, step=10.0, format="%.2f")
            submit_asset = st.form_submit_button("Save Asset Position")
            
            if submit_asset:
                if not ticker_input:
                    st.error("Please provide a valid ticker symbol.")
                elif shares_input <= 0 or cost_input <= 0:
                    st.error("Shares and cost basis must be positive values.")
                else:
                    try:
                        services.add_stock_holding(ticker_input, shares_input, cost_input)
                        # Immediately log a pricing snapshot on manual update
                        import yfinance as yf
                        try:
                            t_obj = yf.Ticker(ticker_input)
                            last_p = float(t_obj.fast_info.get("lastPrice") or t_obj.info.get("regularMarketPrice") or cost_input)
                            daily_ch = float(t_obj.info.get("regularMarketChangePercent") or 0.0)
                            database.save_stock_price(ticker_input, last_p, daily_ch)
                        except Exception:
                            database.save_stock_price(ticker_input, cost_input, 0.0)
                        st.success(f"Saved {ticker_input} position to CockroachDB.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding asset: {e}")
                        
    with col2:
        st.subheader("📊 Active Portfolio Holdings")
        if metrics["holdings_details"]:
            total_cost = metrics["total_cost"]
            header_cols = st.columns([2, 2, 2, 2, 2, 2])
            header_cols[0].markdown("**Ticker**")
            header_cols[1].markdown("**Shares**")
            header_cols[2].markdown("**Cost Basis**")
            header_cols[3].markdown("**Current Price**")
            header_cols[4].markdown("**Market Value**")
            header_cols[5].markdown("**Action**")
            
            for h in metrics["holdings_details"]:
                row_cols = st.columns([2, 2, 2, 2, 2, 2])
                row_cols[0].markdown(f"**{h['ticker']}**")
                row_cols[1].write(f"{h['shares']:.2f}")
                row_cols[2].write(f"${h['cost_basis']:.2f}")
                row_cols[3].write(f"${h['current_price']:.2f}")
                row_cols[4].write(f"${h['position_value']:.2f}")
                
                if row_cols[5].button("Delete", key=f"del_{h['ticker']}"):
                    try:
                        all_holdings = database.get_holdings()
                        h_id = next((x["holding_id"] for x in all_holdings if x["ticker"] == h["ticker"]), None)
                        if h_id:
                            services.remove_stock_holding(h_id)
                            st.toast(f"Removed {h['ticker']} position.")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error removing position: {e}")
                        
            st.markdown(f"### **Total Cost Basis Value: ${total_cost:,.2f}**")
            
            # Allocation Comparison Bar Chart
            st.markdown("---")
            st.markdown("##### 📊 Asset Allocation: Cost Basis vs. Current Market Value")
            df = pd.DataFrame(metrics["holdings_details"])
            chart_df = df[["ticker", "position_cost", "position_value"]].rename(columns={
                "position_cost": "Cost Basis Value ($)",
                "position_value": "Current Market Value ($)"
            }).set_index("ticker")
            st.bar_chart(chart_df)
            
            # Historical Valuation Line Chart
            st.markdown("---")
            st.markdown("##### 📈 Historical Portfolio Valuation (30-Minute Snapshot Log)")
            try:
                snapshots = database.get_portfolio_snapshots(limit=100)
                if snapshots:
                    snap_df = pd.DataFrame(snapshots)
                    snap_df = snap_df[["recorded_at", "total_value"]].rename(columns={
                        "recorded_at": "Timestamp",
                        "total_value": "Portfolio Value ($)"
                    }).set_index("Timestamp")
                    st.line_chart(snap_df)
                else:
                    st.info("Valuation snapshot history is building. Snapshot entries log automatically every 30 minutes.")
            except Exception as e:
                st.write(f"Could not load historical valuation trends: {e}")
        else:
            st.info("Your portfolio is currently empty. Add positions to get started.")

@st.dialog("➕ Add New Strategy Rule")
def show_add_strategy_dialog():
    new_rule_text = st.text_area("Strategy Guideline Text:", height=100)
    if st.button("➕ Save to Vector Store", use_container_width=True):
        if new_rule_text.strip():
            try:
                services.save_investment_strategy(new_rule_text)
                st.toast("New strategy rule saved!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save strategy: {e}")
        else:
            st.warning("Strategy rule cannot be empty.")

def render_chatbot_page():
    col_header, col_new = st.columns([3, 1])
    with col_header:
        st.header("💬 AI Research Assistant Chatbot")
        st.markdown("Interact with MarketPulse AI. The agent interprets your intent, manages qualitative strategies, runs debates, stress tests, or ingests files.")
    
    with col_new:
        st.write("") # Spacer to align buttons slightly lower
        if st.button("➕ New Chat", use_container_width=True):
            try:
                database.clear_chat_history()
                st.session_state.chat_cleared = False
                greeting = (
                    "Hello! I am MarketPulse AI. How can I help you research your portfolio or manage your investment strategies today?\n\n"
                    "Try asking me:\n"
                    "- *'Add a strategy to limit tech to 30%'*\n"
                    "- *'Show my active strategies'*\n"
                    "- *'Delete the technology strategy rule'*\n"
                    "- *'Run a debate on MSFT'*"
                )
                database.save_chat_message(role="assistant", content=greeting)
                st.session_state.chat_history = [{"role": "assistant", "content": greeting}]
                st.toast("Started a new conversation thread.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to reset chat: {e}")

    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        
        # Initialize chat history from Database
        if "chat_history" not in st.session_state:
            try:
                db_history = database.get_chat_history(limit=20)
                if db_history:
                    st.session_state.chat_history = db_history
                else:
                    if not st.session_state.get("chat_cleared", False):
                        greeting = (
                            "Hello! I am MarketPulse AI. How can I help you research your portfolio or manage your investment strategies today?\n\n"
                            "Try asking me:\n"
                            "- *'Add a strategy to limit tech to 30%'*\n"
                            "- *'Show my active strategies'*\n"
                            "- *'Delete the technology strategy rule'*\n"
                            "- *'Run a debate on MSFT'*"
                        )
                        database.save_chat_message(role="assistant", content=greeting)
                        st.session_state.chat_history = [{"role": "assistant", "content": greeting}]
                    else:
                        st.session_state.chat_history = []
            except Exception as e:
                st.session_state.chat_history = []
            
        # Render chat history
        chat_container = st.container(height=400)
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    
        # Show pending strategy confirmation form if active
        if st.session_state.get("pending_strategy"):
            pending = st.session_state.pending_strategy
            st.markdown("🤖 **MarketPulse AI**: I drafted the following strategy update for you. Please review or tweak it before saving:")
            
            # Form style card container
            with st.container(border=True):
                updated_text = st.text_area(
                    "Proposed Strategy Text:",
                    value=pending["strategy_text"],
                    key="pending_strategy_text_input",
                    label_visibility="collapsed",
                    height=100
                )
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("✅ Confirm & Save to Vector Store", use_container_width=True):
                        try:
                            if pending["action_type"] == "add_strategy":
                                services.save_investment_strategy(updated_text)
                                confirm_msg = f"✅ **Strategy Rule Saved**: \"{updated_text}\""
                            elif pending["action_type"] == "update_strategy":
                                services.update_strategy_by_reference(pending["strategy_target"], updated_text)
                                confirm_msg = f"✅ **Strategy Rule Updated**: \"{updated_text}\""
                            
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": confirm_msg
                            })
                            try:
                                database.save_chat_message(role="assistant", content=confirm_msg)
                            except Exception as e:
                                print(f"Error persisting confirmation message: {e}")
                                
                            # Check and execute remaining actions
                            remaining = pending.get("remaining_actions")
                            if remaining:
                                with st.spinner("Executing remaining portfolio actions..."):
                                    res_remaining = services.run_remaining_actions(remaining, pending["original_prompt"])
                                    final_content = res_remaining["response"]
                                    
                                    st.session_state.chat_history.append({
                                        "role": "assistant",
                                        "content": final_content
                                    })
                                    try:
                                        database.save_chat_message(role="assistant", content=final_content)
                                    except Exception as e:
                                        print(f"Error persisting remaining actions response: {e}")
                                        
                            del st.session_state.pending_strategy
                            st.toast("Strategy saved and actions executed successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to save strategy: {e}")
                            
                with col_btn2:
                    if st.button("❌ Cancel", use_container_width=True):
                        cancel_msg = "❌ Strategy modification cancelled."
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": cancel_msg
                        })
                        try:
                            database.save_chat_message(role="assistant", content=cancel_msg)
                        except Exception as e:
                            print(f"Error persisting cancel message: {e}")
                            
                        del st.session_state.pending_strategy
                        st.toast("Strategy update cancelled.")
                        st.rerun()

        # Chat Input - disabled if strategy pending user decision
        is_pending = bool(st.session_state.get("pending_strategy"))
        
        # Requires Streamlit 1.37.0+
        user_input_data = st.chat_input(
            "Ask MarketPulse AI...", 
            disabled=is_pending,
            accept_file="multiple"
        )
        
        if user_input_data:
            user_prompt = ""
            uploaded_files_list = []
            file_context_texts = []
            
            if isinstance(user_input_data, dict):
                user_prompt = user_input_data.get("text", "").strip()
                st_files = user_input_data.get("files", [])
            else:
                user_prompt = str(user_input_data).strip()
                st_files = []
                
            for f in st_files:
                f_bytes = f.getvalue()
                uploaded_files_list.append({
                    "name": f.name,
                    "bytes": f_bytes
                })
                try:
                    if f.name.lower().endswith(".pdf"):
                        import pypdf
                        import io
                        pdf_reader = pypdf.PdfReader(io.BytesIO(f_bytes))
                        text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
                        file_context_texts.append(f"--- File: {f.name} ---\n{text}")
                    else:
                        file_context_texts.append(f"--- File: {f.name} ---\n{f_bytes.decode('utf-8')}")
                except Exception as e:
                    st.warning(f"Could not read text context from {f.name}: {e}")
                    
            file_context = "\n\n".join(file_context_texts) if file_context_texts else None
            
            display_prompt = user_prompt if user_prompt else f"[Attached {len(st_files)} file(s)]"
            
            try:
                database.save_chat_message(role="user", content=display_prompt)
            except Exception as e:
                print(f"Error persisting user message: {e}")
            st.session_state.chat_history.append({"role": "user", "content": display_prompt})
            
            with st.spinner("MarketPulse AI is processing and orchestrating tools..."):
                try:
                    res = services.run_chatbot_session(user_prompt, uploaded_files=uploaded_files_list, file_context=file_context)
                    
                    try:
                        database.save_chat_message(role="assistant", content=res["response"])
                    except Exception as e:
                        print(f"Error persisting assistant message: {e}")
                    
                    # Append assistant response
                    st.session_state.chat_history.append({"role": "assistant", "content": res["response"]})
                    st.session_state.last_router_output = res["router"]
                    if res.get("pending_strategy"):
                        st.session_state.pending_strategy = res["pending_strategy"]
                    st.rerun()
                except Exception as e:
                    st.error(f"Chatbot failed to process request: {e}")
                    
    with col2:
        st.subheader("🎯 Active Strategy Rules")
        st.markdown("Qualitative guidelines stored in CockroachDB and adhered to by the AI agent.")
        
        if st.button("➕ Add New Strategy", use_container_width=True):
            show_add_strategy_dialog()
            
        st.markdown("---")
        
        try:
            strategies = services.get_investment_strategies()
            if strategies:
                for idx, s in enumerate(strategies):
                    strat_id = s['strategy_id']
                    created_str = s['created_at'].strftime('%Y-%m-%d %H:%M') if hasattr(s['created_at'], 'strftime') else str(s['created_at'])[:16]
                    
                    with st.container(border=True):
                        # If in editing mode, show inline input form
                        if st.session_state.get("editing_strategy_id") == strat_id:
                            st.write(f"**Editing Rule #{idx+1}**")
                            edit_text = st.text_area(
                                "Edit Rule:",
                                value=s['strategy_text'],
                                key=f"edit_val_{strat_id}",
                                label_visibility="collapsed",
                                height=100
                            )
                            
                            col_save, col_cancel = st.columns(2)
                            with col_save:
                                if st.button("💾 Save", key=f"save_{strat_id}", use_container_width=True):
                                    if edit_text.strip():
                                        try:
                                            services.update_strategy_by_reference(str(strat_id), edit_text)
                                            st.session_state.editing_strategy_id = None
                                            st.toast("Strategy updated successfully!")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Failed to update strategy: {e}")
                                    else:
                                        st.warning("Strategy rule cannot be empty.")
                            with col_cancel:
                                if st.button("❌ Cancel", key=f"cancel_{strat_id}", use_container_width=True):
                                    st.session_state.editing_strategy_id = None
                                    st.rerun()
                        else:
                            # Read-only card mode
                            st.markdown(f"**Rule #{idx+1}**")
                            st.write(s['strategy_text'])
                            st.caption(f"Created: {created_str}")
                            
                            col_edit, col_del = st.columns(2)
                            with col_edit:
                                if st.button("✏️ Edit", key=f"edit_btn_{strat_id}", use_container_width=True):
                                    st.session_state.editing_strategy_id = strat_id
                                    st.rerun()
                            with col_del:
                                if st.button("🗑️ Delete", key=f"del_btn_{strat_id}", use_container_width=True):
                                    try:
                                        database.remove_strategy(strat_id)
                                        st.toast("Strategy deleted successfully!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to delete strategy: {e}")
            else:
                st.info("No qualitative strategies configured. Click 'Add New Strategy' above or tell the chatbot to add one.")
        except Exception as e:
            st.error(f"Error loading strategies: {e}")

def render_news_page():
    st.header("📰 Market News Portal")
    st.markdown("Monitor real-time market news and request on-demand AI action suggestions tailored to your portfolio.")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ News Sentinel Controls")
        
        # Get active tickers from portfolio positions
        holdings = services.get_stock_holdings()
        portfolio_tickers = sorted(list(set(h["ticker"] for h in holdings)))
        
        tickers_list = ["All Tickers"] + portfolio_tickers
        selected_ticker = st.selectbox(
            "Filter news by portfolio ticker:",
            tickers_list,
            index=0
        )
        
        custom_ticker = st.text_input(
            "Or search specific custom ticker:",
            placeholder="e.g. NVDA, TSLA, AMZN"
        ).upper().strip()
        
        target_ticker = custom_ticker if custom_ticker else (None if selected_ticker == "All Tickers" else selected_ticker)
        
        # Refresh news
        st.markdown(" ")
        if st.button("🔄 Refresh Latest News", use_container_width=True):
            # Check which tickers to refresh
            tickers_to_fetch = []
            if custom_ticker:
                tickers_to_fetch = [custom_ticker]
            elif selected_ticker == "All Tickers":
                tickers_to_fetch = portfolio_tickers
            else:
                tickers_to_fetch = [selected_ticker]
                
            if not tickers_to_fetch:
                st.warning("No tickers selected or in portfolio to refresh. Try searching a custom ticker.")
            else:
                with st.spinner("Fetching latest news from Yahoo Finance & Tavily..."):
                    new_count = 0
                    for tick in tickers_to_fetch:
                        new_count += services.fetch_and_store_news(tick)
                    
                    st.toast(f"News refreshed successfully! Added {new_count} new articles.")
                    st.rerun()
                    
    with col2:
        st.subheader("📡 News Sentinel Feed")
        
        stored_news = services.get_stored_news(ticker=target_ticker)
        
        if stored_news:
            for item in stored_news:
                with st.container(border=True):
                    header_str = f"🏷️ **{item['ticker']}** | 📰 {item['source']} | 📅 {item['published_at'].strftime('%Y-%m-%d %H:%M') if hasattr(item['published_at'], 'strftime') else str(item['published_at'])[:16]}"
                    st.markdown(header_str)
                    st.markdown(f"### {item['title']}")
                    if item.get("summary"):
                        st.write(item["summary"])
                    if item.get("url"):
                        st.markdown(f"🔗 [Read Full Article]({item['url']})")
                        
                    # Expander for suggestions
                    sugg_key = f"exp_{item['news_id']}"
                    with st.expander("AI Action Suggestions", expanded=False):
                        suggestions = item.get("action_suggestions")
                        if suggestions:
                            st.markdown(suggestions)
                            st.markdown("---")
                            if st.button("🔄 Regenerate Suggestions", key=f"regen_{item['news_id']}", use_container_width=True):
                                with st.spinner("Regenerating AI suggestions based on latest strategies..."):
                                    try:
                                        new_sugg = services.generate_suggestions_for_news(item['news_id'])
                                        st.toast("Suggestions updated successfully!")
                                        st.rerun()
                                    except Exception as ex:
                                        st.error(f"Failed to generate suggestions: {ex}")
                        else:
                            st.info("No AI action suggestions generated yet for this article.")
                            if st.button("💡 Generate Action Suggestions", key=f"gen_{item['news_id']}", use_container_width=True):
                                with st.spinner("Orchestrating Gemini Agent with your strategies & holdings..."):
                                    try:
                                        services.generate_suggestions_for_news(item['news_id'])
                                        st.toast("Action suggestions generated and saved!")
                                        st.rerun()
                                    except Exception as ex:
                                        st.error(f"Failed to generate suggestions: {ex}")
        else:
            st.info("No news articles currently stored in CockroachDB matching filters. Select a ticker and click 'Refresh Latest News' above to fetch latest market data.")

def render_developer_page():
    st.header("🛠️ Developer Diagnostics")
    st.markdown("Monitor real-time system connections, vector memory configurations, schema migrations, and LLM orchestration.")
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["🏥 System Health", "🧬 Vector Space & Router", "📜 Schema & Audit Logs"])
    
    with tab1:
        st.subheader("Infrastructure Connections")
        
        # Check CockroachDB Connection & Version
        db_status = "Disconnected"
        db_version = "Unknown"
        db_err = None
        try:
            conn = database.get_db_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                db_version = cur.fetchone()[0]
            database.release_db_connection(conn)
            db_status = "Connected"
        except Exception as e:
            db_err = str(e)
        
        # Check Supabase Storage Connection
        supabase_status = "Disconnected"
        supabase_err = None
        try:
            from supabase import create_client
            supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
            supabase.storage.list_buckets()
            supabase_status = "Connected"
        except Exception as e:
            supabase_status = "Policy Restricted / Offline"
            supabase_err = str(e)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div style="background-color: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 20px; margin-bottom: 15px;">
                    <h4 style="margin: 0; color: #FFFFFF;">CockroachDB Serverless</h4>
                    <p style="margin: 10px 0 0 0; font-size: 24px; font-weight: bold; color: {'#10B981' if db_status == 'Connected' else '#EF4444'};">
                        {'🟢 Connected' if db_status == 'Connected' else '🔴 Disconnected'}
                    </p>
                    <p style="margin: 5px 0 0 0; font-size: 12px; color: #9CA3AF; font-family: monospace; white-space: pre-wrap;">{db_version}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            if db_err:
                st.error(f"DB Connection Error: {db_err}")

        with col2:
            st.markdown(
                f"""
                <div style="background-color: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 20px; margin-bottom: 15px;">
                    <h4 style="margin: 0; color: #FFFFFF;">Supabase Object Storage</h4>
                    <p style="margin: 10px 0 0 0; font-size: 24px; font-weight: bold; color: {'#10B981' if supabase_status == 'Connected' else '#F59E0B'};">
                        {'🟢 Connected' if supabase_status == 'Connected' else '🟡 Restricted / Offline'}
                    </p>
                    <p style="margin: 5px 0 0 0; font-size: 12px; color: #9CA3AF;">Access check via bucket enumeration</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            if supabase_err:
                st.warning(f"Supabase Warning/Error: {supabase_err}")
        
        st.markdown("---")
        st.subheader("Generative AI & Models")
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown(
                f"""
                <div style="background-color: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 20px; margin-bottom: 15px;">
                    <h5 style="margin: 0; color: #9CA3AF;">Embedding Model</h5>
                    <p style="margin: 10px 0 0 0; font-size: 20px; font-weight: bold; color: #00A8A8; font-family: monospace;">
                        {config.EMBEDDING_MODEL}
                    </p>
                    <p style="margin: 5px 0 0 0; font-size: 12px; color: #9CA3AF;">Dimensions: 768</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m_col2:
            st.markdown(
                f"""
                <div style="background-color: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 20px; margin-bottom: 15px;">
                    <h5 style="margin: 0; color: #9CA3AF;">Generative Model</h5>
                    <p style="margin: 10px 0 0 0; font-size: 20px; font-weight: bold; color: #00A8A8; font-family: monospace;">
                        {config.GENERATIVE_MODEL}
                    </p>
                    <p style="margin: 5px 0 0 0; font-size: 12px; color: #9CA3AF;">Provider: Gemini API</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    with tab2:
        st.subheader("AI Chatbot Router Decisions")
        if "last_router_output" in st.session_state:
            st.markdown(
                f"""
                <div style="background-color: #111827; border-left: 4px solid #00A8A8; padding: 15px; border-radius: 4px; margin-bottom: 15px;">
                    <h5 style="margin: 0; color: #00A8A8;">Last Session Routing Decision</h5>
                    <p style="margin: 10px 0 5px 0;"><strong>Router Explanation:</strong> {st.session_state.last_router_output["explanation"]}</p>
                    <p style="margin: 5px 0 0 0;"><strong>Resolved Actions:</strong> <code>{json.dumps(st.session_state.last_router_output["actions"])}</code></p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("No chatbot session routed yet in this Streamlit session. Send a message to the AI Chatbot to see routing decisions.")
        
        st.markdown("---")
        st.subheader("HNSW Cosine Similarity Query Executed")
        st.markdown("Inspect how query vectors match stored database records. Uses cosine distance (`<=>` operator) on HNSW indexes.")
        sql_example = """
        SELECT strategy_id, strategy_text, (embedding <=> %s::VECTOR) AS distance
        FROM user_strategies
        ORDER BY distance ASC
        LIMIT %s;
        """
        st.code(sql_example, language="sql")

    with tab3:
        st.subheader("Schema Migrations Applied")
        try:
            conn = database.get_db_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT version, applied_at FROM schema_version ORDER BY version ASC;")
                mig_rows = cur.fetchall()
            database.release_db_connection(conn)
            
            # Render migrations nicely
            mig_df = pd.DataFrame(mig_rows, columns=["Migration Version", "Applied At"])
            st.dataframe(mig_df, use_container_width=True)
        except Exception as e:
            st.error("Unable to fetch migration logs.")
        
        st.markdown("---")
        st.subheader("Database Research Audit Logs (Row-Level TTL active)")
        try:
            logs = database.get_research_logs()
            if logs:
                for idx, log in enumerate(logs[:10]):
                    with st.expander(f"Log {idx+1}: {log['prompt_query'][:50]}... ({str(log['created_at'])[:19]})"):
                        st.markdown(f"**Query**: `{log['prompt_query']}`")
                        st.write(f"- Distance score of top strategy: `{log['vector_distance']}`")
                        st.write(f"- News count: `{log['session_metadata'].get('news_sources_count', 'N/A')}`")
                        st.write(f"- Latency: `{log['session_metadata'].get('execution_latency_sec', 'N/A')}s` | Model: `{log['generative_model']}`")
                        st.markdown("**Generated Summary:**")
                        st.info(log['generated_summary'])
                        
                        b_col1, b_col2 = st.columns(2)
                        with b_col1:
                            st.markdown("**Bull Perspective:**")
                            st.success(log['bull_perspective'] or "N/A")
                        with b_col2:
                            st.markdown("**Bear Perspective:**")
                            st.error(log['bear_perspective'] or "N/A")
            else:
                st.write("No audit logs saved yet.")
        except Exception as e:
            st.write(f"Error fetching audit trail: {e}")

# ==========================================
# PAGE ROUTING CONTROL
# ==========================================

if selected_page == "💼 Portfolio Positions":
    render_portfolio_page()
elif selected_page == "💬 AI Research Chatbot":
    render_chatbot_page()
elif selected_page == "📰 Market News Portal":
    render_news_page()
elif selected_page == "🛠️ Developer Diagnostics":
    render_developer_page()


