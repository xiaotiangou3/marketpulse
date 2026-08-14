import io
import json
import time
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import services.database as database
import services.alpaca_service as alpaca_service
import services
import config
from typing import Optional


# Set up page configurations
st.set_page_config(
    page_title="MarketPulse AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for tabs, header, and clean spacing
st.markdown("""
<style>
    /* Header styling */
    .marketpulse-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.25rem 0 1rem 0;
        border-bottom: 1px solid rgba(250, 250, 250, 0.1);
        margin-bottom: 1.25rem;
    }
    .marketpulse-title {
        font-size: 1.85rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .marketpulse-badge {
        font-size: 0.8rem;
        font-weight: 500;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.25);
    }
    
    /* Native tab styling */
    div[data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 4px;
        margin-bottom: 1rem;
    }
    div[data-baseweb="tab"] {
        padding: 8px 18px;
        border-radius: 8px 8px 0 0;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.2s ease-in-out;
    }
    div[data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.04);
        color: #38bdf8;
    }
    div[data-baseweb="tab"][aria-selected="true"] {
        font-weight: 600;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #38bdf8;
        height: 3px;
        border-radius: 3px 3px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# Run database migrations on startup
try:
    database.run_migrations()
except Exception as e:
    st.error(f"Failed to execute database migrations: {e}")

# Initialize session state variables
if "active_ingestion_jobs" not in st.session_state:
    st.session_state.active_ingestion_jobs = {}
if "pending_portfolio_overwrite" not in st.session_state:
    st.session_state.pending_portfolio_overwrite = None
if "active_session_file_holdings" not in st.session_state:
    st.session_state.active_session_file_holdings = None
if "pending_toasts" not in st.session_state:
    st.session_state.pending_toasts = []

# Render and clear pending toasts
for msg in st.session_state.pending_toasts:
    st.toast(msg)
st.session_state.pending_toasts = []

# Header
st.markdown("""
<div class="marketpulse-header">
    <div class="marketpulse-title">MarketPulse AI</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# FORMATTING HELPERS
# ==========================================

def format_currency(val: float, show_sign: bool = False) -> str:
    """Formats a float as currency. Handles negative signs correctly: -$1.00 instead of $-1.00."""
    try:
        val = float(val)
    except (ValueError, TypeError):
        return "$0.00"
    if val < 0:
        return f"-${abs(val):,.2f}"
    elif show_sign and val > 0:
        return f"+${val:,.2f}"
    return f"${val:,.2f}"

def format_delta(val: float, pct: Optional[float] = None) -> str:
    """Formats a delta with sign before dollar: -$1.00 (-0.50%) or +$1.00 (+0.50%)."""
    try:
        val = float(val)
    except (ValueError, TypeError):
        val = 0.0
    sign = "-" if val < 0 else ("+" if val > 0 else "")
    pct_part = f" ({pct:+.2f}%)" if pct is not None else ""
    return f"{sign}${abs(val):,.2f}{pct_part}"

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
    holdings = metrics.get("holdings_details", [])
    
    if "portfolio_timeframe" not in st.session_state:
        st.session_state["portfolio_timeframe"] = "1D"
        
    st.markdown("---")
    
    # 1. Full-width Hero Card Section
    with st.container(border=True):
        hero_left, hero_right = st.columns([3, 2])
        
        with hero_right:
            selected_tf = st.radio(
                "Timeframe",
                options=["1D", "1W", "1M", "1Y", "ALL"],
                index=["1D", "1W", "1M", "1Y", "ALL"].index(st.session_state.get("portfolio_timeframe", "1D")),
                horizontal=True,
                key="portfolio_timeframe_radio"
            )
            st.session_state["portfolio_timeframe"] = selected_tf
            
        # Fetch dynamic timeframe history for holdings
        history_result = services.fetch_portfolio_history(holdings, timeframe=selected_tf)
        hist_df = history_result.get("df", pd.DataFrame())
        
        # Calculate adaptive timeframe change
        if selected_tf == "1D" and metrics["total_value"] > 0:
            tf_change_val = metrics["daily_change"]
            tf_change_pct = metrics["daily_change_pct"]
        elif not hist_df.empty and history_result.get("start_value", 0.0) > 0:
            tf_change_val = history_result["change_value"]
            tf_change_pct = history_result["change_pct"]
        else:
            tf_change_val = metrics.get("daily_change", 0.0)
            tf_change_pct = metrics.get("daily_change_pct", 0.0)
            
        with hero_left:
            st.caption("Total Portfolio Value")
            st.markdown(f"<h1 style='margin: 0; padding: 0; font-size: 2.3rem; font-weight: 800;'>{format_currency(metrics['total_value'])}</h1>", unsafe_allow_html=True)
            
            delta_color = "#ef4444" if tf_change_val < 0 else ("#10b981" if tf_change_val > 0 else "#94a3b8")
            st.markdown(
                f"<div style='font-size: 1.1rem; font-weight: 600; color: {delta_color}; margin-top: 4px;'>"
                f"{format_delta(tf_change_val, tf_change_pct)} "
                f"<span style='font-size: 0.85rem; color: #94a3b8; font-weight: 400;'>({selected_tf})</span>"
                f"</div>",
                unsafe_allow_html=True
            )
            
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        
        # Secondary Summary Metrics
        # sub_m1, sub_m2, sub_m3 = st.columns(3)
        # sub_m1.metric("Total Cost Basis", format_currency(metrics["total_cost"]))
        # sub_m2.metric(
        #     "Total Unrealized P&L",
        #     format_currency(metrics["total_gain_loss"], show_sign=True),
        #     delta=f"{metrics['total_gain_loss_pct']:+.2f}%"
        # )
        # sub_m3.metric("Tracked Positions", f"{len(holdings)} Assets")
        
        st.markdown("---")
        
        # Historical Portfolio Performance Chart (Directly below portfolio value in Hero Card)
        st.markdown(f"##### 📈 Historical Portfolio Performance ({selected_tf})")
        if not hist_df.empty and len(hist_df) > 1:
            st.line_chart(hist_df, color="#38bdf8", use_container_width=True)
        elif holdings:
            # Fallback to database snapshots if real-time series is building
            try:
                snapshots = database.get_portfolio_snapshots(limit=100)
                if snapshots:
                    snap_df = pd.DataFrame(snapshots)
                    snap_df["total_value"] = snap_df["total_value"].astype(float)
                    snap_df = snap_df[["recorded_at", "total_value"]].rename(columns={
                        "recorded_at": "Timestamp",
                        "total_value": "Portfolio Value ($)"
                    }).set_index("Timestamp")
                    st.line_chart(snap_df, color="#38bdf8", use_container_width=True)
                else:
                    st.info(f"Loading {selected_tf} performance chart data...")
            except Exception as e:
                st.caption(f"Historical valuation trends unavailable: {e}")
        else:
            st.info("Your portfolio is currently empty. Add positions below to track historical performance.")
            
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("➕ Add Asset Position")
        with st.form("add_asset_form", clear_on_submit=True):
            ticker_input = st.text_input("Asset Ticker", placeholder="e.g. MSFT, AAPL, BTC").upper().strip()
            shares_input = st.number_input("Shares", min_value=0.0001, step=1.0, format="%.4f")
            cost_input = st.number_input("Average Cost Basis ($)", min_value=0.01, step=10.0, format="%.2f")
            submit_asset = st.form_submit_button("Save Asset Position", use_container_width=True, type="primary")
            
            if submit_asset:
                if not ticker_input:
                    st.error("Please provide a valid ticker symbol.")
                elif shares_input <= 0 or cost_input <= 0:
                    st.error("Shares and cost basis must be positive values.")
                else:
                    try:
                        canonical_ticker = services.canonicalize_ticker(ticker_input)
                        services.add_stock_holding(canonical_ticker, shares_input, cost_input)
                        # Immediately log a pricing snapshot on manual update using multi-tier real-time price fetcher
                        try:
                            last_p, daily_ch, _ = services.fetch_realtime_price(canonical_ticker, fallback_price=cost_input)
                            database.save_stock_price(canonical_ticker, last_p, daily_ch)
                        except Exception:
                            database.save_stock_price(canonical_ticker, cost_input, 0.0)
                        
                        disp_name = services.display_ticker(canonical_ticker)
                        if disp_name != canonical_ticker:
                            st.success(f"Saved {disp_name} position (as {canonical_ticker}) to CockroachDB.")
                        else:
                            st.success(f"Saved {canonical_ticker} position to CockroachDB.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding asset: {e}")
                        
    with col2:
        st.subheader("📊 Active Portfolio Holdings")
        if metrics["holdings_details"]:
            total_cost = metrics["total_cost"]
            header_cols = st.columns([1.5, 1.2, 1.5, 1.5, 1.5, 1.2])
            header_cols[0].markdown("**Ticker**")
            header_cols[1].markdown("**Shares**")
            header_cols[2].markdown("**Cost Basis**")
            header_cols[3].markdown("**Current Price**")
            header_cols[4].markdown("**Market Value**")
            header_cols[5].markdown("**Action**")
            
            for h in metrics["holdings_details"]:
                row_cols = st.columns([1.5, 1.2, 1.5, 1.5, 1.5, 1.2])
                disp = services.display_ticker(h['ticker'])
                row_cols[0].markdown(f"**{disp}**")
                row_cols[1].write(f"{h['shares']:.4f}" if h['shares'] < 1 else f"{h['shares']:.2f}")
                row_cols[2].write(format_currency(h['cost_basis']))
                row_cols[3].write(format_currency(h['current_price']))
                row_cols[4].write(format_currency(h['position_value']))
                
                if row_cols[5].button("Delete", key=f"del_{h['ticker']}"):
                    try:
                        all_holdings = database.get_holdings()
                        h_id = next((x["holding_id"] for x in all_holdings if x["ticker"] == h["ticker"]), None)
                        if h_id:
                            services.remove_stock_holding(h_id)
                            st.toast(f"Removed {disp} position.")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error removing position: {e}")
                        
            st.markdown(f"**Total Cost Basis Value: {format_currency(total_cost)}**")
            
            # Allocation Comparison Bar Chart
            st.markdown("---")
            st.markdown("##### 📊 Asset Allocation: Cost Basis vs. Current Market Value")
            df = pd.DataFrame(metrics["holdings_details"])
            df["display_ticker"] = df["ticker"].apply(services.display_ticker)
            df["position_cost"] = df["position_cost"].astype(float)
            df["position_value"] = df["position_value"].astype(float)

            chart_df = df[["display_ticker", "position_cost", "position_value"]].rename(columns={
                "display_ticker": "Ticker",
                "position_cost": "Cost Basis Value ($)",
                "position_value": "Current Market Value ($)"
            }).set_index("Ticker")
            st.bar_chart(chart_df)
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

def inject_slash_command_palette():
    palette_html = """
    <script>
    (function() {
        try {
            const parentDoc = window.parent.document;
            const parentWin = window.parent;
            if (!parentDoc || !parentWin) return;

            // 1. Inject styles into parent document head if not present
            let styles = parentDoc.getElementById("mp-slash-palette-styles");
            if (!styles) {
                styles = parentDoc.createElement("style");
                styles.id = "mp-slash-palette-styles";
                styles.innerHTML = `
                    .mp-palette-popup {
                        position: fixed !important;
                        background: #0f172a !important;
                        border: 2px solid #38bdf8 !important;
                        border-radius: 12px !important;
                        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.95), 0 0 20px rgba(56, 189, 248, 0.3) !important;
                        padding: 8px !important;
                        z-index: 2147483647 !important;
                        font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
                        max-height: 290px !important;
                        overflow-y: auto !important;
                        box-sizing: border-box !important;
                        pointer-events: auto !important;
                        width: 450px !important;
                    }
                    .mp-palette-header {
                        padding: 6px 10px 8px !important;
                        font-size: 11px !important;
                        font-weight: 700 !important;
                        text-transform: uppercase !important;
                        letter-spacing: 0.05em !important;
                        color: #94a3b8 !important;
                        border-bottom: 1px solid #1e293b !important;
                        margin-bottom: 6px !important;
                        display: flex !important;
                        justify-content: space-between !important;
                    }
                    .mp-palette-item {
                        display: flex !important;
                        align-items: center !important;
                        justify-content: space-between !important;
                        padding: 10px 12px !important;
                        border-radius: 8px !important;
                        cursor: pointer !important;
                        margin-bottom: 4px !important;
                        border: 1px solid transparent !important;
                        transition: background 0.1s ease, border-color 0.1s ease !important;
                        background: transparent !important;
                        color: #cbd5e1 !important;
                    }
                    .mp-palette-item.active {
                        background: #1e293b !important;
                        border-color: #38bdf8 !important;
                        color: #ffffff !important;
                    }
                `;
                parentDoc.head.appendChild(styles);
            }

            const commands = [
                {
                    cmd: "/debate",
                    template: "/debate ",
                    icon: "⚔️",
                    badge: "stock / topic",
                    title: "Bull vs. Bear Debate",
                    desc: "Debate upside catalysts vs downside risks based on news"
                },
                {
                    cmd: "/stress",
                    template: "/stress ",
                    icon: "🌪️",
                    badge: "macro scenario",
                    title: "Macro Stress Test",
                    desc: "Simulate portfolio impact under macro / economic shocks"
                },
                {
                    cmd: "/backtest",
                    template: "/backtest ",
                    icon: "📈",
                    badge: "ticker / strategy",
                    title: "Strategy Backtest",
                    desc: "Simulate quantitative strategies (RSI, MACD, SMA, etc.)"
                },
                {
                    cmd: "/trade",
                    template: "/trade ",
                    icon: "🧪",
                    badge: "paper trade",
                    title: "Paper Trade",
                    desc: "Execute paper order via Alpaca Sandbox API"
                },
                {
                    cmd: "/performance",
                    template: "/performance",
                    icon: "📊",
                    badge: "portfolio",
                    title: "Performance & Returns",
                    desc: "Calculate total portfolio valuation, P&L, and allocations"
                },
                {
                    cmd: "/help",
                    template: "/help",
                    icon: "💡",
                    badge: "guide",
                    title: "Commands Reference",
                    desc: "Display reference manual and example commands"
                }
            ];

            let activeIndex = 0;
            let visibleCommands = [];
            let currentTextarea = null;

            function getOrCreatePalette() {
                let palette = parentDoc.getElementById("mp-slash-palette");
                if (!palette) {
                    palette = parentDoc.createElement("div");
                    palette.id = "mp-slash-palette";
                    palette.className = "mp-palette-popup";
                    palette.style.display = "none";

                    const header = parentDoc.createElement("div");
                    header.className = "mp-palette-header";
                    header.innerHTML = "<span>⚡ MarketPulse Commands</span><span>↑↓ Navigate • Tab/Click Select • Esc Close</span>";
                    palette.appendChild(header);

                    const listContainer = parentDoc.createElement("div");
                    listContainer.id = "mp-palette-list";
                    palette.appendChild(listContainer);

                    parentDoc.body.appendChild(palette);
                } else if (palette.parentElement !== parentDoc.body) {
                    parentDoc.body.appendChild(palette);
                }
                return palette;
            }

            function setNativeValue(element, value) {
                const valueSetter = Object.getOwnPropertyDescriptor(element, 'value')?.set;
                const prototype = Object.getPrototypeOf(element);
                const prototypeValueSetter = Object.getOwnPropertyDescriptor(prototype, 'value')?.set;
                
                if (prototypeValueSetter && valueSetter !== prototypeValueSetter) {
                    prototypeValueSetter.call(element, value);
                } else if (valueSetter) {
                    valueSetter.call(element, value);
                } else {
                    element.value = value;
                }
                element.dispatchEvent(new Event('input', { bubbles: true }));
                element.dispatchEvent(new Event('change', { bubbles: true }));
            }

            function selectCommand(textarea, cmdObj) {
                const text = textarea.value || "";
                const cursorPos = textarea.selectionStart ?? text.length;
                const beforeCursor = text.slice(0, cursorPos);
                const afterCursor = text.slice(cursorPos);
                const lastSlash = beforeCursor.lastIndexOf("/");
                
                let newText = "";
                let newCursorPos = 0;
                if (lastSlash !== -1) {
                    const prefix = text.slice(0, lastSlash);
                    newText = prefix + cmdObj.template + afterCursor;
                    newCursorPos = (prefix + cmdObj.template).length;
                } else {
                    newText = cmdObj.template;
                    newCursorPos = cmdObj.template.length;
                }
                
                setNativeValue(textarea, newText);
                hidePalette();
                setTimeout(() => {
                    textarea.focus();
                    textarea.setSelectionRange(newCursorPos, newCursorPos);
                }, 50);
            }

            function renderItems(textarea, shouldScroll = false) {
                const palette = getOrCreatePalette();
                const list = palette.querySelector("#mp-palette-list");
                if (!list) return;
                list.innerHTML = "";

                visibleCommands.forEach((cmd, idx) => {
                    const item = parentDoc.createElement("div");
                    const isActive = idx === activeIndex;
                    item.className = "mp-palette-item" + (isActive ? " active" : "");

                    item.innerHTML = `
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span style="font-size: 16px;">${cmd.icon}</span>
                            <div>
                                <div style="display: flex; align-items: center; gap: 6px;">
                                    <span style="font-weight: 700; font-size: 13px; color: ${isActive ? "#38bdf8" : "#94a3b8"};">${cmd.cmd}</span>
                                    <span style="font-size: 11px; background: #334155; color: #cbd5e1; padding: 1px 6px; border-radius: 4px; font-family: monospace;">${cmd.badge}</span>
                                </div>
                                <div style="font-size: 12px; color: ${isActive ? "#f1f5f9" : "#64748b"}; margin-top: 1px;">${cmd.desc}</div>
                            </div>
                        </div>
                    `;

                    item.addEventListener("mousemove", () => {
                        if (activeIndex !== idx) {
                            activeIndex = idx;
                            renderItems(textarea, false);
                        }
                    });

                    item.addEventListener("mousedown", (e) => {
                        e.preventDefault();
                        selectCommand(textarea, cmd);
                    });

                    list.appendChild(item);

                    if (isActive && shouldScroll) {
                        setTimeout(() => {
                            item.scrollIntoView({ block: "nearest", behavior: "smooth" });
                        }, 10);
                    }
                });
            }

            function positionPalette(textarea) {
                const palette = getOrCreatePalette();
                const container = textarea.closest('div[data-testid="stChatInput"]') || textarea.parentElement;
                const targetRect = container ? container.getBoundingClientRect() : textarea.getBoundingClientRect();
                
                palette.style.position = "fixed";
                palette.style.left = targetRect.left + "px";
                palette.style.width = Math.max(targetRect.width, 320) + "px";
                palette.style.bottom = (parentWin.innerHeight - targetRect.top + 8) + "px";
                palette.style.top = "auto";
                palette.style.zIndex = "2147483647";
            }

            function showPalette(textarea) {
                currentTextarea = textarea;
                const palette = getOrCreatePalette();
                positionPalette(textarea);
                renderItems(textarea, true);
                palette.style.display = "block";
            }

            function hidePalette() {
                const palette = parentDoc.getElementById("mp-slash-palette");
                if (palette) {
                    palette.style.display = "none";
                }
            }

            function handleInputOrFocus(e) {
                const target = e.target;
                if (!target || target.tagName !== "TEXTAREA") return;
                
                const isChatInput = target.closest('div[data-testid="stChatInput"]') || 
                                   target.getAttribute("data-testid") === "stChatInputTextArea" ||
                                   target.placeholder?.toLowerCase().includes("ask") ||
                                   target.placeholder?.toLowerCase().includes("/");
                if (!isChatInput) return;

                const val = target.value || "";
                const cursorPos = target.selectionStart ?? val.length;
                const textBeforeCursor = val.slice(0, cursorPos);
                const lastSlashIndex = textBeforeCursor.lastIndexOf("/");

                if (lastSlashIndex !== -1) {
                    const charBeforeSlash = lastSlashIndex > 0 ? textBeforeCursor[lastSlashIndex - 1] : " ";
                    const isWordStart = /\s/.test(charBeforeSlash) || lastSlashIndex === 0;
                    const searchWord = textBeforeCursor.slice(lastSlashIndex);

                    if (isWordStart && !/\s/.test(searchWord)) {
                        const search = searchWord.trim().toLowerCase();
                        visibleCommands = commands.filter(c => c.cmd.toLowerCase().startsWith(search) || search === "/" || c.cmd.toLowerCase().includes(search));
                        if (visibleCommands.length > 0) {
                            if (activeIndex >= visibleCommands.length) activeIndex = 0;
                            showPalette(target);
                            return;
                        }
                    }
                }
                hidePalette();
            }

            function handleKeydown(e) {
                const palette = parentDoc.getElementById("mp-slash-palette");
                const isOpen = palette && palette.style.display === "block";

                if (isOpen && currentTextarea) {
                    if (e.key === "ArrowDown") {
                        e.preventDefault();
                        e.stopPropagation();
                        activeIndex = (activeIndex + 1) % visibleCommands.length;
                        renderItems(currentTextarea, true);
                    } else if (e.key === "ArrowUp") {
                        e.preventDefault();
                        e.stopPropagation();
                        activeIndex = (activeIndex - 1 + visibleCommands.length) % visibleCommands.length;
                        renderItems(currentTextarea, true);
                    } else if (e.key === "Tab" || (e.key === "Enter" && !e.shiftKey && visibleCommands.length > 0)) {
                        const text = currentTextarea.value || "";
                        const cursorPos = currentTextarea.selectionStart ?? text.length;
                        const textBeforeCursor = text.slice(0, cursorPos);
                        const lastSlashIndex = textBeforeCursor.lastIndexOf("/");
                        const searchWord = lastSlashIndex !== -1 ? textBeforeCursor.slice(lastSlashIndex) : "";
                        
                        if (lastSlashIndex !== -1 && !/\s/.test(searchWord)) {
                            e.preventDefault();
                            e.stopPropagation();
                            selectCommand(currentTextarea, visibleCommands[activeIndex]);
                        }
                    } else if (e.key === "Escape") {
                        e.preventDefault();
                        e.stopPropagation();
                        hidePalette();
                    }
                }
            }



            // Global capture-phase listeners on parent document
            parentDoc.removeEventListener("input", handleInputOrFocus, true);
            parentDoc.removeEventListener("focusin", handleInputOrFocus, true);
            parentDoc.removeEventListener("click", handleInputOrFocus, true);
            parentDoc.removeEventListener("keydown", handleKeydown, true);

            parentDoc.addEventListener("input", handleInputOrFocus, true);
            parentDoc.addEventListener("focusin", handleInputOrFocus, true);
            parentDoc.addEventListener("click", handleInputOrFocus, true);
            parentDoc.addEventListener("keydown", handleKeydown, true);

            parentWin.addEventListener("resize", () => {
                if (currentTextarea) positionPalette(currentTextarea);
            });
            parentDoc.addEventListener("scroll", () => {
                if (currentTextarea) positionPalette(currentTextarea);
            }, true);

            // Reposition & check on interval
            setInterval(() => {
                if (currentTextarea) {
                    if (!currentTextarea.isConnected) {
                        hidePalette();
                        currentTextarea = null;
                    } else {
                        const palette = parentDoc.getElementById("mp-slash-palette");
                        if (palette && palette.style.display === "block") {
                            positionPalette(currentTextarea);
                        }
                    }
                }
            }, 300);

        } catch (e) {
            console.error("Slash command palette error:", e);
        }
    })();
    </script>
    """
    components.html(palette_html, height=0, width=0)




def render_backtest_card(bt_data: dict):
    """Renders an inline quantitative strategy validation card with KPI metrics and cumulative performance chart."""
    if not bt_data or bt_data.get("error"):
        return
        
    with st.container(border=True):
        outperformed = bt_data.get("outperformed", False)
        ticker = bt_data.get("ticker", "ASSET")
        period = bt_data.get("period", "1y")
        strat_name = bt_data.get("strategy_name", "Quantitative Strategy")
        cond_summary = bt_data.get("condition_summary", "Technical trading rule simulation")
        strat_type = bt_data.get("strategy_type", "sma_cross").lower()
        
        # Determine icon based on strategy type
        icon_map = {
            "rsi": "⚡",
            "macd": "📊",
            "ema_cross": "📈",
            "bollinger": "🎯",
            "breakout": "🚀",
            "sma_cross": "📊"
        }
        icon = icon_map.get(strat_type, "📈")
        
        header_col, badge_col = st.columns([2, 1])
        with header_col:
            st.markdown(f"#### {icon} **{strat_name}: {ticker}**")
            st.caption(f"**Rule Logic**: {cond_summary} | **Timeframe**: {period} *(1-Day Shift / Zero Lookahead Bias)*")
        with badge_col:
            if outperformed:
                st.markdown(
                    """
                    <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10B981; border-radius: 8px; padding: 6px 12px; text-align: center; color: #10B981; font-weight: 600; font-size: 0.85rem;">
                        🟢 Outperformed Benchmark
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div style="background: rgba(245, 158, 11, 0.15); border: 1px solid #F59E0B; border-radius: 8px; padding: 6px 12px; text-align: center; color: #F59E0B; font-weight: 600; font-size: 0.85rem;">
                        ⚠️ Underperformed Benchmark
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
        st.markdown("---")
        
        # 4 KPI Metrics
        m1, m2, m3, m4 = st.columns(4)
        strat_ret = bt_data.get("Strategy_Return_Pct", 0.0)
        bh_ret = bt_data.get("Buy_Hold_Return_Pct", 0.0)
        win_rate = bt_data.get("Win_Rate_Pct", 0.0)
        max_dd = bt_data.get("Max_Drawdown_Pct", 0.0)
        
        m1.metric("Strategy Return", f"{strat_ret:+.2f}%", delta=f"{strat_ret - bh_ret:+.2f}% vs. B&H")
        m2.metric("Buy & Hold Benchmark", f"{bh_ret:+.2f}%")
        m3.metric("Win Rate", f"{win_rate:.2f}%")
        m4.metric("Max Drawdown", f"{max_dd:.2f}%")
        
        # Cumulative returns comparison line chart
        chart_data = bt_data.get("chart_data")
        if chart_data:
            try:
                chart_df = pd.DataFrame(chart_data).set_index("Date")
                st.markdown("##### 📈 Cumulative Performance Comparison (%)")
                st.line_chart(chart_df, color=["#38BDF8", "#94A3B8"])
            except Exception:
                pass

def render_trade_receipt_card(trade_data: dict):
    """Renders an inline paper trade execution summary card with order metadata."""
    if not trade_data or trade_data.get("error"):
        return
        
    with st.container(border=True):
        symbol = trade_data.get("symbol", "ASSET")
        side = str(trade_data.get("side", "BUY")).upper()
        qty = float(trade_data.get("qty", 0.0))
        status = str(trade_data.get("status", "ACCEPTED")).upper()
        order_id = trade_data.get("order_id", "N/A")
        price = trade_data.get("execution_price")
        timestamp = trade_data.get("timestamp", "Just now")
        tif = trade_data.get("time_in_force", "GTC")
        sbx_name = trade_data.get("sandbox_name")
        
        is_buy = (side == "BUY")
        badge_bg = "rgba(16, 185, 129, 0.15)" if is_buy else "rgba(239, 68, 68, 0.15)"
        badge_border = "#10B981" if is_buy else "#EF4444"
        badge_color = "#10B981" if is_buy else "#EF4444"
        icon = "🟢" if is_buy else "🔴"
        
        header_col, badge_col = st.columns([2, 1])
        with header_col:
            sbx_badge = f"<span style='background-color: #0369a1; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; margin-left: 8px;'>🧪 {sbx_name}</span>" if sbx_name else ""
            st.markdown(f"#### 🧪 **Alpaca Paper Order: {symbol}** {sbx_badge}", unsafe_allow_html=True)
            st.caption(f"**Order ID**: `{order_id}` | **Time-in-Force**: `{tif}` | **Timestamp**: {timestamp}")
        with badge_col:
            st.markdown(
                f"""
                <div style="background: {badge_bg}; border: 1px solid {badge_border}; border-radius: 8px; padding: 6px 12px; text-align: center; color: {badge_color}; font-weight: 600; font-size: 0.85rem;">
                    {icon} {side} {qty:g} SHARES
                </div>
                """,
                unsafe_allow_html=True
            )

            
        st.markdown("---")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Order Side", side)
        m2.metric("Quantity", f"{qty:g} shares")
        m3.metric("Execution Price", f"${price:,.2f}" if price else "Market Price")
        m4.metric("Status", status)
        
        st.caption("💡 *Track live positions, cash balance, and virtual equity in the 🧪 **Paper Trading Sandbox** tab.*")

@st.fragment(run_every=5)
def poll_ingestion_jobs_status():
    if not st.session_state.get("active_ingestion_jobs"):
        return
        
    st.markdown("### 📥 Ingestion Progress")
    completed_jobs = []
    failed_jobs = []
    
    for job_id, job_name in list(st.session_state.active_ingestion_jobs.items()):
        try:
            job = database.get_ingestion_job(job_id)
            if not job:
                continue
                
            status = job["status"]
            progress = job["progress_pct"]
            error_msg = job["error_message"]
            
            if status == "completed":
                if job["file_type"] == "csv":
                    try:
                        metadata = json.loads(error_msg) if error_msg else {}
                        if metadata.get("overwrite_intent"):
                            st.session_state.pending_portfolio_overwrite = {
                                "job_id": job_id,
                                "file_name": job_name,
                                "holdings": metadata["holdings"]
                            }
                        else:
                            st.session_state.active_session_file_holdings = metadata["holdings"]
                    except Exception:
                        pass
                completed_jobs.append(job_id)
                st.toast(f"✅ Ingestion complete: {job_name}")
            elif status == "failed":
                failed_jobs.append(job_id)
                st.error(f"❌ Ingestion failed for {job_name}: {error_msg}")
            else:
                st.progress(progress / 100.0, text=f"**{job_name}**: {status.capitalize()} ({progress}%)")
        except Exception as e:
            print(f"Error polling job {job_id}: {e}")
            
    for jid in completed_jobs + failed_jobs:
        if jid in st.session_state.active_ingestion_jobs:
            del st.session_state.active_ingestion_jobs[jid]
            
    if completed_jobs or failed_jobs:
        st.rerun()

def render_chatbot_page():
    col_header, col_new = st.columns([3, 1])
    with col_header:
        st.header("💬 AI Research Assistant Chatbot")
        st.markdown("Interact with MarketPulse AI using natural language or slash commands (e.g. `/debate`, `/stress`, `/performance`, `/help`).")
    
    with col_new:
        st.write("") # Spacer to align buttons slightly lower
        if st.button("➕ New Chat", use_container_width=True):
            try:
                database.clear_chat_history()
                st.session_state.chat_cleared = False
                greeting = (
                    "Hello! I am MarketPulse AI. How can I help you research your portfolio or manage your investment strategies today?\n\n"
                    "💡 **Try typing `/` in the chat bar or asking:**\n"
                    "- *'/debate talk about the bull and bear catalysts for NVDA based on news'*\n"
                    "- *'/stress analyze what happens if interest rates rise 50bps'*\n"
                    "- *'/performance'*\n"
                    "- *'Add a strategy to limit tech to 30%'*\n"
                    "- *'/help' for full commands guide*"
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
                    if msg.get("backtest_data"):
                        render_backtest_card(msg["backtest_data"])
                    if msg.get("trade_data"):
                        render_trade_receipt_card(msg["trade_data"])
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
                                with chat_container:
                                    with st.chat_message("assistant"):
                                        rem_status = st.status("🔄 Resuming portfolio actions...", expanded=True)
                                        def handle_rem_status(label: str, detail: str = None, state: str = "running"):
                                            rem_status.update(label=label, state=state)
                                            if detail:
                                                with rem_status:
                                                    st.write(detail)
                                        try:
                                            res_remaining = services.run_remaining_actions(remaining, pending["original_prompt"], status_callback=handle_rem_status)
                                            rem_status.update(label="✅ Continuation actions complete", state="complete", expanded=False)
                                            
                                            if res_remaining.get("backtest_data"):
                                                render_backtest_card(res_remaining["backtest_data"])
                                            if res_remaining.get("trade_data"):
                                                render_trade_receipt_card(res_remaining["trade_data"])
                                                
                                            final_content = res_remaining["response"]
                                            def stream_words(text: str):
                                                words = text.split(" ")
                                                for i, word in enumerate(words):
                                                    yield word + (" " if i < len(words) - 1 else "")
                                                    time.sleep(0.008)
                                            st.write_stream(stream_words(final_content))
                                            st.session_state.chat_history.append({
                                                "role": "assistant",
                                                "content": final_content,
                                                "backtest_data": res_remaining.get("backtest_data"),
                                                "trade_data": res_remaining.get("trade_data")
                                            })

                                            try:
                                                database.save_chat_message(role="assistant", content=final_content)
                                            except Exception as e:
                                                print(f"Error persisting remaining actions response: {e}")
                                        except Exception as e:
                                            rem_status.update(label="❌ Continuation actions failed", state="error", expanded=True)
                                            st.error(f"Failed to complete actions: {e}")
                                        
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

        # Show pending portfolio overwrite confirmation form if active
        if st.session_state.get("pending_portfolio_overwrite"):
            pending_port = st.session_state.pending_portfolio_overwrite
            st.markdown(f"🤖 **MarketPulse AI**: I parsed the portfolio CSV **{pending_port['file_name']}**. Do you want to overwrite your active holdings with these new items?")
            
            # Render a summary table of the new holdings
            holdings_df = pd.DataFrame(pending_port["holdings"])
            st.dataframe(holdings_df, hide_index=True)
            
            col_pbtn1, col_pbtn2 = st.columns(2)
            with col_pbtn1:
                if st.button("✅ Overwrite Holdings", key="confirm_port_overwrite", use_container_width=True):
                    try:
                        # Overwrite holdings in DB
                        # First delete all holdings
                        conn = database.get_db_connection()
                        try:
                            with conn.cursor() as cur:
                                cur.execute("DELETE FROM user_holdings;")
                                conn.commit()
                        finally:
                            database.release_db_connection(conn)
                            
                        # Insert new holdings
                        for h in pending_port["holdings"]:
                            database.add_holding(h["ticker"], h["shares"], h["cost_basis"])
                            
                        confirm_msg = f"✅ **Portfolio Holdings Overwritten** with {len(pending_port['holdings'])} assets from `{pending_port['file_name']}`."
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": confirm_msg
                        })
                        try:
                            database.save_chat_message(role="assistant", content=confirm_msg)
                        except Exception as e:
                            print(f"Error persisting holdings overwrite msg: {e}")
                            
                        del st.session_state.pending_portfolio_overwrite
                        st.toast("Portfolio holdings successfully updated!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to update holdings: {e}")
            with col_pbtn2:
                if st.button("❌ Keep Current & Analyze Uploaded Only", key="cancel_port_overwrite", use_container_width=True):
                    cancel_msg = f"ℹ️ **Using uploaded CSV data for analysis context only** without overwriting your portfolio."
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": cancel_msg
                    })
                    try:
                        database.save_chat_message(role="assistant", content=cancel_msg)
                    except Exception as e:
                        print(f"Error persisting cancel holdings overwrite msg: {e}")
                        
                    st.session_state.active_session_file_holdings = pending_port["holdings"]
                    del st.session_state.pending_portfolio_overwrite
                    st.toast("Holding overwrite cancelled. Uploaded assets will be used for current chat context.")
                    st.rerun()

        # Render ingestion progress tracker fragment
        poll_ingestion_jobs_status()

        # Chat Input - disabled if strategy or portfolio pending user decision
        is_pending = bool(st.session_state.get("pending_strategy")) or bool(st.session_state.get("pending_portfolio_overwrite"))
        
        inject_slash_command_palette()
        # Requires Streamlit 1.37.0+
        user_input_data = st.chat_input(
            "Ask MarketPulse AI or type '/' for commands...", 
            disabled=is_pending,
            accept_file="multiple",
            file_type=["pdf", "csv"]
        )

        if user_input_data:
            user_prompt = ""
            uploaded_files_list = []
            file_context_texts = []
            
            if isinstance(user_input_data, dict):
                user_prompt = str(user_input_data.get("text") or "").strip()
                st_files = user_input_data.get("files", []) or []
            elif hasattr(user_input_data, "text"):
                user_prompt = str(getattr(user_input_data, "text", "") or "").strip()
                st_files = getattr(user_input_data, "files", []) or []
            elif isinstance(user_input_data, str):
                user_prompt = user_input_data.strip()
                st_files = []
            else:
                try:
                    user_prompt = str(getattr(user_input_data, "text", "")).strip()
                except Exception:
                    user_prompt = ""
                st_files = getattr(user_input_data, "files", []) or []
                
            # Filter and restrict uploaded files to PDF or CSV only
            valid_files = []
            for f in st_files:
                if not f.name.lower().endswith(('.pdf', '.csv')):
                    st.session_state.pending_toasts.append(f"❌ Unsupported file format: `{f.name}`. Only PDF and CSV files are allowed.")
                else:
                    valid_files.append(f)
            
            # If no prompt and no valid files are present, skip execution
            if not user_prompt and not valid_files:
                st.rerun()
                
            st_files = valid_files
                
            # Track jobs we kick off in this interaction
            current_active_jobs = {}
            for f in st_files:
                f_bytes = f.getvalue()
                file_type = "csv" if f.name.lower().endswith(".csv") else "pdf"
                job_id = database.create_ingestion_job(f.name, file_type)
                
                # Extract ticker if any ticker is implied in user prompt
                implied_ticker = None
                words = user_prompt.upper().split()
                for w in words:
                    w_clean = "".join([c for c in w if c.isalnum()])
                    if len(w_clean) >= 1 and len(w_clean) <= 5 and w_clean.isalpha():
                        implied_ticker = w_clean
                        break
                
                services.start_ingestion_job(job_id, f.name, f_bytes, user_prompt, implied_ticker)
                st.session_state.active_ingestion_jobs[job_id] = f.name
                current_active_jobs[job_id] = f.name
                
            # Block and update status in Streamlit while jobs are active
            if current_active_jobs:
                status_placeholder = st.empty()
                import time
                while any(jid in st.session_state.active_ingestion_jobs for jid in current_active_jobs):
                    time.sleep(1.0)
                    active_desc = []
                    for jid, jname in list(current_active_jobs.items()):
                        job = database.get_ingestion_job(jid)
                        if job:
                            status = job["status"]
                            progress = job["progress_pct"]
                            if status in ("completed", "failed"):
                                if jid in st.session_state.active_ingestion_jobs:
                                    if status == "completed":
                                        if job["file_type"] == "csv":
                                            try:
                                                metadata = json.loads(job["error_message"]) if job["error_message"] else {}
                                                if metadata.get("overwrite_intent"):
                                                    st.session_state.pending_portfolio_overwrite = {
                                                        "job_id": jid,
                                                        "file_name": jname,
                                                        "holdings": metadata["holdings"]
                                                    }
                                                else:
                                                    st.session_state.active_session_file_holdings = metadata["holdings"]
                                            except Exception:
                                                pass
                                        else:
                                            file_context_texts.append(f"[Document '{jname}' processed and indexed in database. Use semantic search/RAG for queries regarding it.]")
                                        st.toast(f"✅ Indexed {jname}!")
                                    else:
                                        st.error(f"❌ Failed to index {jname}: {job['error_message']}")
                                    del st.session_state.active_ingestion_jobs[jid]
                            else:
                                active_desc.append(f"🔄 **{jname}**: {status.capitalize()} ({progress}%)")
                    
                    if active_desc:
                        status_placeholder.markdown("\n".join(active_desc))
                    else:
                        status_placeholder.empty()
                status_placeholder.empty()

            # Fetch and append all indexed documents in database to the context
            try:
                all_docs = database.get_all_documents()
                if all_docs:
                    docs_list_str = "### INDEXED DOCUMENTS IN DATABASE ###\n"
                    for d in all_docs:
                        docs_list_str += f"- Name: {d['name']} (Type: {d['file_type']})\n"
                    file_context_texts.append(docs_list_str)
            except Exception as e:
                print(f"Error appending indexed documents to context: {e}")

            # Append active session file holdings context if present (e.g. portfolio uploaded for analysis)
            if st.session_state.get("active_session_file_holdings"):
                holdings_list = st.session_state.active_session_file_holdings
                uploaded_filename = st_files[0].name if st_files else "portfolio.csv"
                context_str = f"### CURRENTLY UPLOADED FILE ###\nFile Name: {uploaded_filename}\nUploaded Portfolio CSV Holdings Context:\n"
                for h in holdings_list:
                    context_str += f"- Ticker: {h['ticker']}, Shares: {h['shares']}, Cost Basis: ${h['cost_basis']}\n"
                
                est_tokens = len(context_str) // 4
                if est_tokens < config.DIRECT_CONTEXT_TOKEN_THRESHOLD:
                    file_context_texts.append(context_str)
                else:
                    file_context_texts.append(f"[Uploaded portfolio CSV '{uploaded_filename}' contains too many rows ({len(holdings_list)}). Relying on database search.]")
                st.session_state.active_session_file_holdings = None # Consume
                
            file_context = "\n\n".join(file_context_texts) if file_context_texts else None
            
            if st_files:
                file_badges = ", ".join([f"`{f.name}`" for f in st_files])
                if user_prompt:
                    display_prompt = f"{user_prompt}\n\n📎 **Attached:** {file_badges}"
                else:
                    display_prompt = f"📎 **Attached:** {file_badges}"
            else:
                display_prompt = user_prompt
            
            try:
                database.save_chat_message(role="user", content=display_prompt)
            except Exception as e:
                print(f"Error persisting user message: {e}")
            st.session_state.chat_history.append({"role": "user", "content": display_prompt})
            
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(display_prompt)
                with st.chat_message("assistant"):
                    status_box = st.status("🧠 MarketPulse AI is initializing...", expanded=True)
                    def handle_status(label: str, detail: str = None, state: str = "running"):
                        status_box.update(label=label, state=state)
                        if detail:
                            with status_box:
                                st.write(detail)
                    try:
                        res = services.run_chatbot_session(
                            user_prompt, 
                            uploaded_files=uploaded_files_list, 
                            file_context=file_context,
                            status_callback=handle_status
                        )
                        status_box.update(label="✅ Analysis & execution complete", state="complete", expanded=False)
                        
                        # Render inline quantitative backtest validation card if present
                        if res.get("backtest_data"):
                            render_backtest_card(res["backtest_data"])
                        if res.get("trade_data"):
                            render_trade_receipt_card(res["trade_data"])
                            
                        def stream_words(text: str):
                            words = text.split(" ")
                            for i, word in enumerate(words):
                                yield word + (" " if i < len(words) - 1 else "")
                                time.sleep(0.008)
                                
                        st.write_stream(stream_words(res["response"]))
                        
                        try:
                            database.save_chat_message(role="assistant", content=res["response"])
                        except Exception as e:
                            print(f"Error persisting assistant message: {e}")
                        
                        # Append assistant response
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": res["response"],
                            "backtest_data": res.get("backtest_data"),
                            "trade_data": res.get("trade_data")
                        })

                        st.session_state.last_router_output = res["router"]
                        if res.get("pending_strategy"):
                            st.session_state.pending_strategy = res["pending_strategy"]
                        st.rerun()
                    except Exception as e:
                        status_box.update(label="❌ Error processing request", state="error", expanded=True)
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
        
        custom_ticker_raw = st.text_input(
            "Or search specific custom ticker:",
            placeholder="e.g. NVDA, TSLA, BTC, ADA, XRP"
        ).upper().strip()
        custom_ticker = services.canonicalize_ticker(custom_ticker_raw) if custom_ticker_raw else ""
        
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
# MULTI-STRATEGY PAPER TRADING SANDBOX PAGE
# ==========================================

@st.dialog("➕ Create New Strategy Sandbox")
def show_create_sandbox_dialog(num_sandboxes: int):
    if num_sandboxes >= 10:
        st.warning("⚠️ **Maximum limit of 10 strategy sandboxes reached.** Please delete an existing sandbox first.")
        return
    st.markdown(f"Configure an isolated sub-ledger to test a new trading strategy. (`{num_sandboxes}/10 Sandboxes Active`)")
    with st.form("create_sandbox_dialog_form"):
        sbx_name = st.text_input("Sandbox Name", placeholder="e.g. NVDA RSI Reversal, Tech Momentum, Macro Defense")
        sbx_desc = st.text_area("Strategy Description (Optional)", placeholder="e.g. Buys when RSI drops below 30, sells above 70")
        sbx_capital = st.number_input("Initial Virtual Capital ($)", min_value=1000.0, max_value=10000000.0, value=100000.0, step=5000.0)
        
        strat_opts = [
            "General / Discretionary", 
            "RSI Mean Reversion (rsi)", 
            "MACD Crossover (macd)", 
            "SMA Crossover (sma_cross)", 
            "EMA Crossover (ema_cross)", 
            "Bollinger Bands (bollinger)", 
            "Price Breakout (breakout)"
        ]
        sel_strat = st.selectbox("Strategy Rule Binding", options=strat_opts)
        strat_code = sel_strat.split("(")[-1].replace(")", "").strip() if "(" in sel_strat else "general"
        
        btn_submit = st.form_submit_button("🚀 Launch Sandbox", type="primary", use_container_width=True)
        if btn_submit:
            if not sbx_name.strip():
                st.error("Please enter a valid sandbox name.")
            else:
                try:
                    new_id = database.create_sandbox(
                        name=sbx_name.strip(),
                        description=sbx_desc.strip() if sbx_desc else None,
                        initial_capital=sbx_capital,
                        strategy_type=strat_code
                    )
                    st.toast(f"✅ Sandbox '{sbx_name}' created successfully!")
                    st.session_state.selected_sandbox_id = str(new_id)
                    time.sleep(0.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to create sandbox: {e}")

def render_paper_trading_page():
    sandboxes = database.get_sandboxes() if alpaca_service.is_alpaca_configured() else []
    num_sandboxes = len(sandboxes)

    col_t, col_b1, col_b2 = st.columns([2.4, 1.2, 1.2])
    with col_t:
        st.header("🧪 Paper Trading Sandbox")
        st.markdown("Run, monitor, and benchmark up to **10 independent strategy sandboxes** with virtual sub-ledgers.")
    with col_b1:
        st.write("")
        if st.button("➕ Create New Sandbox", use_container_width=True, type="primary"):
            show_create_sandbox_dialog(num_sandboxes)
    with col_b2:
        st.write("")
        if st.button("🔄 Refresh Sandbox Data", use_container_width=True):
            st.toast("Refreshed Paper Trading metrics.")
            st.rerun()

    if not alpaca_service.is_alpaca_configured():
        st.warning(
            """
            ### ⚠️ Alpaca Paper Trading Credentials Not Configured
            
            To enable paper trading orders and portfolio tracking:
            1. Log into or sign up for free at [Alpaca Markets](https://app.alpaca.markets/paper/dashboard/overview).
            2. Switch to **Paper Trading** in the top navigation.
            3. Generate your **API Key** and **Secret Key**.
            4. Add `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in your `.env` file.
            """
        )
        return

    # 1. Zero-Sandbox Empty State
    if num_sandboxes == 0:
        st.markdown(
            """
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px dashed #38bdf8; border-radius: 12px; padding: 40px; text-align: center; margin: 20px 0;">
                <h3 style="color: #38bdf8; margin-top: 0;">No Active Strategy Sandboxes</h3>
                <p style="color: #94a3b8; font-size: 1rem; max-width: 600px; margin: 0 auto 25px auto;">
                    You have no paper trading sandboxes created yet. Create a dedicated sandbox to test, isolate, and benchmark individual trading strategies with virtual capital.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # 2. Multi-Sandbox View Setup
    valid_ids = [str(s["sandbox_id"]) for s in sandboxes]
    if "selected_sandbox_id" not in st.session_state or st.session_state.selected_sandbox_id not in valid_ids:
        st.session_state.selected_sandbox_id = valid_ids[0]

    # Sub-tabs: Active Dashboard vs Leaderboard
    sub_tab_dashboard, sub_tab_leaderboard = st.tabs([
        "📊 Active Sandbox Dashboard",
        "🏆 Strategy Leaderboard"
    ])

    with sub_tab_dashboard:
        # Selector & Actions Bar inside Active Dashboard
        sbx_dict = {str(s["sandbox_id"]): s for s in sandboxes}
        curr_idx = valid_ids.index(st.session_state.selected_sandbox_id) if st.session_state.selected_sandbox_id in valid_ids else 0

        top_c1, top_c2, top_c3 = st.columns([3, 1, 1])
        with top_c1:
            selected_id = st.selectbox(
                f"Select Active Strategy Sandbox ({num_sandboxes}/10)",
                options=valid_ids,
                format_func=lambda sid: f"🧪 {sbx_dict[sid]['name']} (${float(sbx_dict[sid]['cash_balance']):,.0f} Cash | {sbx_dict[sid].get('strategy_type') or 'General'})",
                index=curr_idx
            )
            st.session_state.selected_sandbox_id = selected_id
            selected_sbx = sbx_dict[selected_id]

        with top_c2:
            st.write("")
            if st.button("🔄 Reset", use_container_width=True, help="Reset cash to initial capital and clear open positions"):
                database.reset_sandbox(selected_id)
                st.toast(f"✅ Reset sandbox '{selected_sbx['name']}' to ${float(selected_sbx['initial_capital']):,.2f}")
                time.sleep(0.5)
                st.rerun()

        with top_c3:
            st.write("")
            if st.button("🗑️ Delete", use_container_width=True, help="Delete this strategy sandbox"):
                database.delete_sandbox(selected_id)
                st.toast(f"🗑️ Deleted sandbox '{selected_sbx['name']}'")
                st.session_state.pop("selected_sandbox_id", None)
                time.sleep(0.5)
                st.rerun()

        # Compute live metrics for active sandbox
        metrics = database.calculate_sandbox_metrics(selected_id)
        if not metrics:
            st.error("Failed to load metrics for selected sandbox.")
            return

        trade_logs = database.get_paper_trade_logs(limit=50, sandbox_id=selected_id)

        # 1. Top KPI Metrics Bar
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric(
            label="Virtual Portfolio Equity",
            value=format_currency(metrics['equity']),
            delta=format_delta(metrics['total_pl'], metrics['total_return_pct'])
        )
        kpi2.metric(
            label="Available Cash",
            value=format_currency(metrics['cash'])
        )
        kpi3.metric(
            label="Positions Value",
            value=format_currency(metrics['positions_value'])
        )
        kpi4.metric(
            label="Bound Strategy",
            value=f"{metrics.get('strategy_type', 'General').upper()}"
        )

        st.markdown("---")

        col_left, col_right = st.columns([2, 1])

        with col_left:
            # Component 2: Active Open Positions Table with In-Row Liquidation
            st.subheader(f"📊 Active Positions ({selected_sbx['name']})")
            positions = metrics.get("positions", [])
            if positions:
                # Table column headers
                h_cols = st.columns([1.1, 0.8, 0.9, 1.1, 1.1, 1.2, 1.2, 1.1, 1.3])
                headers = ["Symbol", "Side", "Shares", "Avg Price", "Current", "Market Val", "P&L ($)", "Return", "Action"]
                for hc, hname in zip(h_cols, headers):
                    hc.markdown(f"**{hname}**")
                
                st.markdown("<hr style='margin: 4px 0 10px 0; border: 0; border-top: 1px solid rgba(148, 163, 184, 0.2);'>", unsafe_allow_html=True)
                
                for idx, p in enumerate(positions):
                    r_cols = st.columns([1.1, 0.8, 0.9, 1.1, 1.1, 1.2, 1.2, 1.1, 1.3])
                    r_cols[0].write(p["symbol"])
                    r_cols[1].write(p["side"].upper())
                    r_cols[2].write(f"{p['qty']:g}")
                    r_cols[3].write(format_currency(p['avg_entry_price']))
                    r_cols[4].write(format_currency(p['current_price']))
                    r_cols[5].write(format_currency(p['market_value']))
                    
                    pl_val = p['unrealized_pl']
                    pl_color = "#10b981" if pl_val >= 0 else "#ef4444"
                    r_cols[6].markdown(f"<span style='color:{pl_color}; font-weight:600;'>{format_currency(pl_val, show_sign=True)}</span>", unsafe_allow_html=True)
                    
                    ret_val = p['unrealized_plpc']
                    r_cols[7].markdown(f"<span style='color:{pl_color}; font-weight:600;'>{ret_val:+.2f}%</span>", unsafe_allow_html=True)
                    
                    with r_cols[8]:
                        if st.button("⚡ Liquidate", key=f"liq_btn_{selected_id}_{p['symbol']}_{idx}", use_container_width=True, type="secondary"):
                            try:
                                close_res = alpaca_service.close_sandbox_position(selected_id, p["symbol"])
                                st.toast(f"✅ Closed position for {p['symbol']} ({close_res['status'].upper()})")
                                time.sleep(0.5)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to close position: {e}")
            else:
                st.info(f"No open positions in **{selected_sbx['name']}**. Use the order form on the right or mention this sandbox in chat to execute trades.")

            st.markdown("---")

            # Component 4: Sandbox Execution Audit Log
            st.subheader(f"📜 Trade Audit Log ({selected_sbx['name']})")
            if trade_logs:
                log_records = []
                for l in trade_logs:
                    created_str = l['created_at'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(l['created_at'], 'strftime') else str(l['created_at'])[:19]
                    price_val = l.get('execution_price')
                    price_str = f"${float(price_val):,.2f}" if price_val is not None else "Market"
                    log_records.append({
                        "Time (UTC)": created_str,
                        "Symbol": l["symbol"],
                        "Side": l["side"].upper(),
                        "Qty": f"{float(l['qty']):g}",
                        "Price": price_str,
                        "Status": l["status"].upper(),
                        "Type": l["order_type"].upper(),
                        "TIF": l["time_in_force"].upper(),
                        "Order ID": str(l["order_id"])[:14] + "..."
                    })
                logs_df = pd.DataFrame(log_records)
                st.dataframe(logs_df, use_container_width=True, hide_index=True)
            else:
                st.caption(f"No executed paper trades recorded for '{selected_sbx['name']}' in CockroachDB audit log yet.")

        with col_right:
            # Component 3: Manual Trade Order Form
            with st.container(border=True):
                st.subheader("⚡ Place Paper Order")
                st.caption(f"Routing order directly into **{selected_sbx['name']}** sub-ledger.")

                with st.form("manual_paper_trade_form", clear_on_submit=False):
                    order_ticker = st.text_input("Ticker Symbol", value="NVDA", max_chars=10).upper().strip()
                    order_qty = st.number_input("Share Quantity", min_value=0.1, max_value=100000.0, value=10.0, step=1.0)
                    order_side = st.radio("Order Side", options=["BUY", "SELL"], horizontal=True)
                    st.text_input("Target Sandbox", value=selected_sbx["name"], disabled=True)
                    st.text_input("Order Type", value="Market Order (GTC)", disabled=True)

                    submitted = st.form_submit_button("🚀 Execute Trade", use_container_width=True, type="primary")
                    if submitted:
                        if not order_ticker:
                            st.error("Please enter a valid ticker symbol.")
                        elif order_qty <= 0:
                            st.error("Quantity must be positive.")
                        else:
                            try:
                                trade_res = alpaca_service.submit_paper_order(
                                    symbol=order_ticker,
                                    qty=order_qty,
                                    side=order_side.lower(),
                                    sandbox_id=selected_id,
                                    order_type="market",
                                    time_in_force="gtc"
                                )
                                st.toast(f"✅ Executed {order_side} {order_qty:g} {order_ticker} in {selected_sbx['name']}!")
                                st.success(f"Order `{trade_res['order_id']}` submitted ({trade_res['status'].upper()})")
                                time.sleep(0.5)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to execute paper order: {e}")

    with sub_tab_leaderboard:
        st.subheader("🏆 Multi-Strategy Sandbox Leaderboard")
        st.markdown("Side-by-side performance ranking across all active strategy sandboxes.")

        leaderboard = database.get_all_sandboxes_leaderboard()
        if leaderboard:
            l_records = []
            for item in leaderboard:
                l_records.append({
                    "Rank": item["rank"],
                    "Sandbox Name": item["name"],
                    "Strategy": (item.get("strategy_type") or "General").upper(),
                    "Initial Capital": format_currency(item['initial_capital']),
                    "Current Equity": format_currency(item['equity']),
                    "Total P&L ($)": format_currency(item['total_pl'], show_sign=True),
                    "Total Return (%)": f"{item['total_return_pct']:+.2f}%",
                    "Open Positions": item["positions_count"]
                })
            ldf = pd.DataFrame(l_records)
            st.dataframe(ldf, use_container_width=True, hide_index=True)

            st.markdown("#### 📊 Comparative Returns (% Return by Strategy Sandbox)")
            chart_df = pd.DataFrame({
                "Strategy Sandbox": [item["name"] for item in leaderboard],
                "Total Return (%)": [item["total_return_pct"] for item in leaderboard]
            }).set_index("Strategy Sandbox")
            st.bar_chart(chart_df, color="#38bdf8")
        else:
            st.info("No sandboxes available to benchmark.")



# ==========================================
# PAGE ROUTING (NATIVE TABS)
# ==========================================

tab_portfolio, tab_chat, tab_news, tab_paper_trading, tab_diagnostics = st.tabs([
    "💼 Portfolio",
    "💬 Research Chat",
    "📰 Market News",
    "🧪 Paper Trading Sandbox",
    "🛠️ Diagnostics"
])

with tab_portfolio:
    render_portfolio_page()

with tab_chat:
    render_chatbot_page()

with tab_news:
    render_news_page()

with tab_paper_trading:
    render_paper_trading_page()

with tab_diagnostics:
    render_developer_page()



