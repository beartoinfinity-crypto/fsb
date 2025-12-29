##python -m streamlit run app.py
## TSLA, MSFT, NVDA, GOOG, AAPL, AMZN,AVGO, CRWD
import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import numpy as np
import logging
from datetime import timedelta
import plotly.graph_objects as go
import json
import hashlib
import os

# --- USER AUTHENTICATION SYSTEM ---
USER_FILE = "users.json"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            return json.load(f)
    # Default: create admin user on first run
    default_users = {"banana": hash_password("Thisisultra!@#")}
    save_users(default_users)
    return default_users

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

def init_auth():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.is_admin = False
        st.session_state.show_user_management = False  # Track management page

def login_page():
    st.title("🔐 Login to AI Market Intelligence Pro+")

    tab1, tab2 = st.tabs(["Login", "Admin: Manage Users"])

    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            users = load_users()
            if username in users and users[username] == hash_password(password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.is_admin = (username == "banana")
                st.session_state.show_user_management = False  # Start on analysis
                st.success(f"Welcome back, {username}!")
                st.rerun()
            else:
                st.error("Invalid username or password")

    with tab2:
        if st.session_state.get("is_admin", False):
            st.success("Admin Access Granted")
            # ... (same add/delete user code as before)
            new_user = st.text_input("New Username")
            new_pass = st.text_input("New Password", type="password")
            if st.button("Add User"):
                if new_user and new_pass:
                    users = load_users()
                    if new_user in users:
                        st.warning("User already exists")
                    else:
                        users[new_user] = hash_password(new_pass)
                        save_users(users)
                        st.success(f"User '{new_user}' added!")
                        st.rerun()

            st.write("### Current Users")
            users = load_users()
            for user in list(users.keys()):
                if user != "admin":
                    col1, col2 = st.columns([3,1])
                    col1.write(user)
                    if col2.button("Delete", key=f"del_login_{user}"):
                        del users[user]
                        save_users(users)
                        st.success(f"Deleted {user}")
                        st.rerun()
        else:
            st.warning("You must log in as **admin** to manage users.")

    if st.button("Logout", type="secondary"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.is_admin = False
        st.session_state.show_user_management = False
        st.rerun()

# --- USER MANAGEMENT PAGE (separate function for reuse) ---
def user_management_page():
    st.title("👤 Admin: Manage Users")
    st.info("Add or delete non-admin users below.")

    new_user = st.text_input("New Username")
    new_pass = st.text_input("New Password", type="password")
    if st.button("Add User"):
        if new_user and new_pass:
            users = load_users()
            if new_user in users:
                st.warning("User already exists")
            else:
                users[new_user] = hash_password(new_pass)
                save_users(users)
                st.success(f"User '{new_user}' added!")
                st.rerun()

    st.write("### Current Users")
    users = load_users()
    for user in list(users.keys()):
        if user != "admin":
            col1, col2 = st.columns([3,1])
            col1.write(user)
            if col2.button("Delete", key=f"del_mgmt_{user}"):
                del users[user]
                save_users(users)
                st.success(f"Deleted {user}")
                st.rerun()

    if st.button("← Back to Analysis"):
        st.session_state.show_user_management = False
        st.rerun()

# --- 1. SILENCE LOGS ---
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('prophet').setLevel(logging.ERROR)

st.set_page_config(page_title="AI Market Intelligence Pro+", layout="wide")

# --- 2. TECHNICAL INDICATORS ---

def calculate_kd(s_high, s_low, s_close, n=9, m=3):
    low_n = s_low.rolling(window=n).min()
    high_n = s_high.rolling(window=n).max()
    k_value = 100 * ((s_close - low_n) / (high_n - low_n + 1e-9))
    d_value = k_value.rolling(window=m).mean()
    return k_value.iloc[-1], d_value.iloc[-1]

def calculate_vpt(s_close, s_vol):
    vpt = (s_vol * (s_close.diff() / s_close.shift(1))).cumsum()
    vpt_ema = vpt.ewm(span=20).mean()
    return vpt.iloc[-1], vpt_ema.iloc[-1]

def calculate_obv(s_close, s_vol):
    direction = np.sign(s_close.diff())
    obv = (direction * s_vol).cumsum()
    obv_rising = obv.iloc[-1] > obv.iloc[-2] if len(obv) > 1 else False
    return obv.iloc[-1], obv_rising

def calculate_rsi(series, windows=[3, 5, 7, 14, 30]):
    results = {}
    for w in windows:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=w).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=w).mean()
        rs = gain / (loss + 1e-9)
        results[w] = (100 - (100 / (1 + rs))).iloc[-1]
    return results

def get_rsi_label(val):
    if val >= 80: return "!! PARABOLIC !!"
    if val >= 70: return "OVERBOUGHT"
    if val <= 20: return "!! EXTREME OVERSOLD !!"
    if val <= 30: return "OVERSOLD"
    return "NEUTRAL"

def analyze_volume_signals(s_series, v_series):
    v3 = v_series.tail(3).mean()
    v10 = v_series.tail(10).mean()
    v30 = v_series.tail(30).mean()
    rvol = v3 / v30 if v30 > 0 else 0
    price_change = (s_series.iloc[-1] - s_series.iloc[-4]) / (s_series.iloc[-4] + 1e-9)
    pv_conf = "BULLISH" if price_change > 0 and v3 > v10 else "BEARISH" if price_change < 0 and v3 > v10 else "DULL"
    return {
        'rvol': rvol,
        'trend': "SURGING" if v3 > v10 else "DRYING UP",
        'signal': "INSTITUTIONAL" if rvol > 1.5 else "RETAIL",
        'pv_conf': pv_conf
    }

def calculate_macd(s_close, fast=12, slow=26, signal=9):
    ema_fast = s_close.ewm(span=fast).mean()
    ema_slow = s_close.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]

def calculate_bollinger(s_close, window=20, stds=2):
    sma = s_close.rolling(window).mean()
    std = s_close.rolling(window).std()
    upper = sma + (std * stds)
    lower = sma - (std * stds)
    bandwidth = (upper - lower) / sma
    percent_b = (s_close - lower) / (upper - lower + 1e-9)
    return sma.iloc[-1], upper, lower, bandwidth.iloc[-1], percent_b.iloc[-1]

def calculate_cmf(df, window=20):
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-9)
    mfv = mfm * df['Volume']
    cmf = mfv.rolling(window).sum() / df['Volume'].rolling(window).sum()
    return cmf.iloc[-1] if not cmf.empty else 0

def find_support_resistance(price_series, lookback=365, min_distance=20, tolerance_pct=2.0):
    prices = price_series.tail(lookback)
    if len(prices) < 50:
        return {'support': [], 'resistance': []}
    
    levels = []
    for i in range(1, len(prices)-1):
        price = prices.iloc[i]
        left = prices.iloc[max(0, i-min_distance):i]
        right = prices.iloc[i+1:min(len(prices), i+min_distance+1)]
        
        if (price <= left.min() * (1 + tolerance_pct/100)) and (price <= right.min() * (1 + tolerance_pct/100)):
            levels.append(('support', price))
        elif (price >= left.max() * (1 - tolerance_pct/100)) and (price >= right.max() * (1 - tolerance_pct/100)):
            levels.append(('resistance', price))
    
    supports = sorted(set([round(p, 2) for typ, p in levels if typ == 'support']), reverse=True)[:5]
    resistances = sorted(set([round(p, 2) for typ, p in levels if typ == 'resistance']), reverse=True)[:5]
    
    return {'support': supports, 'resistance': resistances}

def calculate_relative_strength(stock_close, market_close):
    stock_ret = stock_close.pct_change().add(1).cumprod()
    market_ret = market_close.pct_change().add(1).cumprod()
    rs = stock_ret / market_ret
    rs_rating = (rs.iloc[-1] / rs.mean()) * 100
    return rs.iloc[-1], rs_rating

def detect_follow_through(df, window=20):
    """
    Checks the last 20 days for a valid Follow-Through Day (FTD).
    Criteria: 
    1. A 'Day 1' rally attempt (off a recent low).
    2. A 'Day 4-10' spike: Price > +1.5% AND Volume > previous day.
    """
    recent = df.tail(window).copy()
    if len(recent) < 10:
        return None

    # Step 1: Find the 'Rally Attempt' (Day 1)
    # This is the lowest point in the window where the price stopped falling
    lowest_idx = recent['Close'].idxmin()
    day1_pos = recent.index.get_loc(lowest_idx)
    
    # We need at least 4 days of data after the low to have an FTD
    if day1_pos > len(recent) - 4:
        return {"status": "Waiting", "msg": "Rally attempt in progress (too early for FTD)"}

    # Step 2: Look for the spike between Day 4 and Day 10 of the rally
    search_range = recent.iloc[day1_pos + 3 : day1_pos + 10] # Days 4 through 10
    
    for i in range(len(search_range)):
        current_day = search_range.iloc[i]
        prev_day = recent.iloc[recent.index.get_loc(search_range.index[i]) - 1]
        
        price_gain = (current_day['Close'] / prev_day['Close']) - 1
        vol_increase = current_day['Volume'] > prev_day['Volume']
        
        # O'Neil Criteria: >1.5% gain on higher volume
        if price_gain >= 0.015 and vol_increase:
            # Check if we've undercut the Day 1 low since then (Failure check)
            undercut = recent.iloc[day1_pos : recent.index.get_loc(search_range.index[i])]['Close'].min() < recent.loc[lowest_idx, 'Close']
            
            if not undercut:
                return {
                    "status": "VALID", 
                    "date": search_range.index[i].date(),
                    "gain": f"{price_gain*100:.2f}%",
                    "msg": f"Confirmed on {search_range.index[i].date()}"
                }

    return {"status": "None", "msg": "No valid FTD detected in this window."}

# --- 3. DATA FETCHING WITH CACHE ---
@st.cache_data(ttl=3600, show_spinner="Fetching latest market data...")
def fetch_data(tickers, period="5y"):
    data = yf.download(tickers, period=period, auto_adjust=True)
    if data.empty:
        st.error("Failed to fetch data. Check tickers or connection.")
        st.stop()
    return data

# --- MAIN APP ---
def main_app():
# Sidebar: login status + logout + manage users button (for admin)
    st.sidebar.success(f"Logged in as: **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.is_admin = False
        st.session_state.show_user_management = False
        st.rerun()

    if st.session_state.is_admin:
        if st.sidebar.button("👤 Manage Users"):
            st.session_state.show_user_management = True
            st.rerun()

    # If in user management mode, show that page
    if st.session_state.get("show_user_management", False):
        user_management_page()
        return  # Stop here - don't show analysis

    # === NORMAL ANALYSIS PAGE ===
    st.sidebar.title("🚀 AI Market Intelligence Pro+")

    # Add the same Manage Users button at the top of the main page (for admin)
    if st.session_state.is_admin:
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            if st.button("👤 Manage Users"):
                st.session_state.show_user_management = True
                st.rerun()

    user_input = st.sidebar.text_input("Tickers (comma-separated)", "MSFT, NVDA").upper()
    backtest_date = st.sidebar.date_input("Backtest Date (Historical Mode)", value=None, help="Leave empty for live data")
    forecast_days = st.sidebar.slider("Forecast Horizon (days)", 30, 180, 90, 30)
    show_chart = st.sidebar.checkbox("Show Interactive Chart", value=True)
    show_sr = st.sidebar.checkbox("Detect Support/Resistance Levels", value=True)

    market_indices = st.sidebar.multiselect(
        "Market Regressors",
        options=['^GSPC (S&P 500)', '^DJI (Dow Jones)', '^IXIC (Nasdaq)', '^RUT (Russell 2000)'],
        default=['^GSPC (S&P 500)', '^DJI (Dow Jones)', '^IXIC (Nasdaq)']
    )
    market_tickers = [opt.split()[0] for opt in market_indices]

    stocks = [s.strip() for s in user_input.split(",") if s.strip()]

    if not stocks:
        stocks = [ "TSLA", "MSFT", "NVDA", "GOOG", "AAPL", "AMZN", "AVGO", "CRWD"] 
        st.info(f"💡 No input detected. Running analysis for default leaders: {', '.join(stocks)}")

    if st.sidebar.button("🔥 Run Full Analysis"):
        tickers_to_fetch = stocks + market_tickers
        raw_data = fetch_data(tickers_to_fetch)

        close_data = raw_data['Close'] if isinstance(raw_data['Close'], pd.DataFrame) else pd.DataFrame({tickers_to_fetch[0]: raw_data['Close']})
        high_data = raw_data['High'] if isinstance(raw_data['High'], pd.DataFrame) else pd.DataFrame({tickers_to_fetch[0]: raw_data['High']})
        low_data = raw_data['Low'] if isinstance(raw_data['Low'], pd.DataFrame) else pd.DataFrame({tickers_to_fetch[0]: raw_data['Low']})
        volume_data = raw_data['Volume'] if isinstance(raw_data['Volume'], pd.DataFrame) else pd.DataFrame({tickers_to_fetch[0]: raw_data['Volume']})

        primary_market = market_tickers[0] if market_tickers else '^GSPC'

        comparison_df = pd.DataFrame({s: close_data[s].pct_change().add(1).cumprod() for s in stocks})
        comparison_df[primary_market] = close_data[primary_market].pct_change().add(1).cumprod()

        for s in stocks:
            try:
                df = pd.DataFrame({
                    'Close': close_data[s],
                    'High': high_data[s],
                    'Low': low_data[s],
                    'Volume': volume_data[s]
                }).ffill().dropna()

                if df.empty:
                    st.error(f"No data for {s}")
                    continue

                if backtest_date:
                    bt_date = pd.to_datetime(backtest_date)
                    df = df.loc[:bt_date]
                    if df.empty:
                        st.error(f"No data for {s} up to {backtest_date}")
                        continue
                    st.warning(f"🕰️ BACKTEST MODE: Analyzing {s} as of {df.index[-1].date()}")

                price = df['Close'].iloc[-1]
                last_date = df.index[-1]

                # Indicators
                rsis = calculate_rsi(df['Close'])
                vol_data = analyze_volume_signals(df['Close'], df['Volume'])
                vpt, vpt_ema = calculate_vpt(df['Close'], df['Volume'])
                obv_val, obv_rising = calculate_obv(df['Close'], df['Volume'])  # NEW: OBV
                k_val, d_val = calculate_kd(df['High'], df['Low'], df['Close'])
                macd_line, macd_signal, macd_hist = calculate_macd(df['Close'])
                bb_mid, bb_upper, bb_lower, bb_bw, bb_percent = calculate_bollinger(df['Close'])
                cmf_val = calculate_cmf(df)
                smas = {w: df['Close'].rolling(w).mean().iloc[-1] for w in [20, 50, 200]}
                recent_std = df['Close'].tail(30).std()

                sr_levels = find_support_resistance(df['Close']) if show_sr else {'support': [], 'resistance': []}
                rs_ratio, rs_rating = calculate_relative_strength(df['Close'], close_data[primary_market])

                # --- PROPHET FORECAST (kept from your version) ---
                regressor_df = pd.DataFrame(index=df.index)
                for ticker in market_tickers:
                    reg_name = ticker.replace('^', '')
                    regressor_df[reg_name] = close_data[ticker].ffill().reindex(df.index)

                s_df = df['Close'].reset_index()
                s_df.columns = ['ds', 'y']
                s_df['ds'] = pd.to_datetime(s_df['ds']).dt.tz_localize(None)
                s_df = s_df.join(regressor_df.reset_index(drop=True))

                model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False,
                                growth='linear', changepoint_prior_scale=0.05)
                for reg in regressor_df.columns:
                    model.add_regressor(reg)
                model.fit(s_df)

                future_dates = model.make_future_dataframe(periods=forecast_days, freq='B')

                future_regressors = pd.DataFrame({'ds': future_dates['ds']})
                for ticker in market_tickers:
                    reg_name = ticker.replace('^', '')
                    m_model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False,
                                    growth='linear', changepoint_prior_scale=0.05)
                    m_df = close_data[ticker].ffill().loc[:df.index[-1]].reset_index()
                    m_df.columns = ['ds', 'y']
                    m_df['ds'] = pd.to_datetime(m_df['ds']).dt.tz_localize(None)
                    m_model.fit(m_df)
                    m_forecast = m_model.predict(future_dates)
                    future_regressors[reg_name] = m_forecast['yhat']

                future = future_dates.merge(future_regressors, on='ds')

                forecast = model.predict(future)
                future_horizon = forecast.tail(forecast_days)

                ftd_data = detect_follow_through(df)

                # Interactive Chart (unchanged)
                if show_chart:
                    fig = go.Figure()
                    hist_len = min(400, len(df))
                    recent_df = df.tail(hist_len)
                    
                    fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df['Close'], name="Price", line=dict(width=3)))
                    fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df['Close'].rolling(50).mean(), name="50 SMA", line=dict(color='orange')))
                    fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df['Close'].rolling(200).mean(), name="200 SMA", line=dict(color='purple')))
                    fig.add_trace(go.Scatter(x=recent_df.index, y=bb_upper.tail(hist_len), name="BB Upper", line=dict(dash='dot', color='gray')))
                    fig.add_trace(go.Scatter(x=recent_df.index, y=bb_lower.tail(hist_len), name="BB Lower", line=dict(dash='dot', color='gray'), fill='tonexty', fillcolor='rgba(200,200,200,0.2)'))
                    
                    for level in sr_levels['support']:
                        fig.add_hline(y=level, line_dash="dash", line_color="green", annotation_text=f"Support ${level:.2f}")
                    for level in sr_levels['resistance']:
                        fig.add_hline(y=level, line_dash="dash", line_color="red", annotation_text=f"Resistance ${level:.2f}")

                    fig.add_trace(go.Scatter(x=future_horizon['ds'], y=future_horizon['yhat'], name="Forecast", line=dict(color='lime', dash='dot')))
                    fig.add_trace(go.Scatter(x=future_horizon['ds'], y=future_horizon['yhat_upper'], line=dict(color='lightgreen', dash='dot'), showlegend=False))
                    fig.add_trace(go.Scatter(x=future_horizon['ds'], y=future_horizon['yhat_lower'], line=dict(color='lightgreen', dash='dot'), fill='tonexty', fillcolor='rgba(0,255,0,0.1)'))

                    fig.update_layout(title=f"{s} - Price, Indicators & {forecast_days}D Forecast", height=600)
                    st.plotly_chart(fig, width="stretch")

                # Output sections
                st.header(f"📊 {s} Analysis ({last_date.date()})")

                c1, c2, c3 = st.columns([2, 2, 1])
                with c1:
                    st.subheader("🎯 Short-Term Targets")
                    rows = [[last_date.date(), "Current", f"${price:.2f}", "Base"]]
                    for days in [3, 7, 14, 30]:
                        target_date = last_date + timedelta(days=days)
                        pred_row = forecast[forecast['ds'] >= target_date].iloc[0]
                        pct = (pred_row['yhat'] / price - 1) * 100
                        signal = "🚀 Strong Up" if pct > 5 else "⬆️ Bullish" if pct > 0 else "⬇️ Bearish" if pct < -2 else "➡️ Flat"
                        rows.append([pred_row['ds'].date(), f"{days}D", f"${pred_row['yhat']:.2f}", f"{signal} ({pct:+.1f}%)"])
                    st.table(pd.DataFrame(rows, columns=["Date", "Horizon", "Target", "Signal"]))

                    st.subheader(f"📈 {forecast_days}D Outlook")
                    st.success(f"**Target Range:** ${future_horizon['yhat_lower'].min():.2f} → ${future_horizon['yhat'].mean():.2f} → ${future_horizon['yhat_upper'].max():.2f}")

                with c2:
                    st.subheader("🛡️ Technical Dashboard")
                    t1, t2 = st.columns(2)
                    with t1:
                        st.write("**Momentum**")
                        for w, v in rsis.items():
                            st.write(f"RSI {w}D: {v:.1f} → {get_rsi_label(v)}")
                        st.write(f"MACD Hist: {macd_hist:+.3f} → {'🟢 Bullish' if macd_hist > 0 else '🔴 Bearish'}")
                        st.write(f"%B (BB): {bb_percent:.2f} → {'Overbought' if bb_percent > 0.8 else 'Oversold' if bb_percent < 0.2 else 'Neutral'}")

                    with t2:
                        st.write("**Volume & Flow**")
                        st.write(f"VPT: {'Accumulating 🟢' if vpt > vpt_ema else 'Distributing 🔴'}")
                        st.write(f"OBV: {'RISING 🟢' if obv_rising else 'FALLING 🔴'}")  # NEW: Show OBV direction
                        st.write(f"CMF(20): {cmf_val:+.3f} → {'Strong Buying 🟢' if cmf_val > 0.05 else 'Strong Selling 🔴' if cmf_val < -0.05 else 'Neutral'}")
                        st.write(f"RVOL: {vol_data['rvol']:.2f}x → {vol_data['signal']}")

                with c3:
                    st.subheader("💪 Pro Metrics")
                    st.metric("Relative Strength", f"{rs_ratio:.2f}x", f"{rs_rating:.0f} Rating")
                    ftd_color = "normal" if ftd_data["status"] == "VALID" else "off"
                    st.metric("Follow-Through", ftd_data["status"], ftd_data["msg"], delta_color=ftd_color)
                    st.metric("Price vs 200SMA", "Above" if price > smas[200] else "Below", f"{(price / smas[200] - 1)*100:+.1f}%")

                st.divider()
                st.subheader("🤖 AI Recommendation Engine")

                # ENHANCED SCORING WITH 3 NEW STRATEGIES
                score = 0
                # --- ADVANCED RSI STACKING SCORING ---

                # A. The Baseline (14-Day)
                if rsis[14] < 30: score += 4
                elif rsis[14] > 70: score -= 4

                # B. Long-Term Context (30-Day)
                # If the 30-day RSI is above 50, we are in a "Structural Bull Trend"
                if rsis[30] > 50: 
                    score += 2  # Reward staying in a strong long-term trend
                elif rsis[30] < 40:
                    score -= 2  # Penalty for long-term weakness

                # C. Short-Term "Rubber Band" (3-Day & 5-Day)
                # We look for EXTREMES here to identify exhaustion
                if rsis[3] > 95 and rsis[5] > 90:
                    score -= 3  # "Too Hot" - The rubber band is stretched; expect a 1-2 day pullback
                elif rsis[3] < 10 and rsis[5] < 15:
                    score += 3  # "Spring Loaded" - Short term panic, usually a great entry

                # D. Momentum Ignition (The "Crossover")
                # If fast RSIs are higher than slow RSIs, momentum is accelerating
                if rsis[5] > rsis[30] + 20:
                    score += 2  # Acceleration signal

                if bb_percent < 0.2: score += 3
                if bb_percent > 0.8: score -= 3
                if vpt > vpt_ema: score += 3
                if cmf_val > 0.05: score += 4
                if cmf_val < -0.05: score -= 4
                if macd_hist > 0: score += 3
                if price > smas[200]: score += 4
                if rs_ratio > 1.1: score += 3
                if ftd_data["status"] == "VALID": 
                    score += 6  # High weight because FTDs are rare and powerful
                elif ftd_data["status"] == "Waiting":
                    score += 1  # Minor bonus for being in a rally attempt

                # 1. Add OBV direction
                if obv_rising: score += 2

                # 2. Weight RVOL more + bonus for price-volume confirmation
                if vol_data['rvol'] > 1.5: score += 3      # Increased weight
                if vol_data['rvol'] > 2.0: score += 2      # Extra for very high RVOL
                if vol_data['pv_conf'] == "BULLISH": score += 2
                if vol_data['pv_conf'] == "BEARISH": score -= 2

                # 3. Penalty for divergence (price up but volume flow down, or vice versa)
                price_trend_up = df['Close'].pct_change(20).iloc[-1] > 0
                price_trend_down = df['Close'].pct_change(20).iloc[-1] < 0
                volume_flow_bullish = (cmf_val > 0 or vpt > vpt_ema)
                volume_flow_bearish = (cmf_val < 0 or vpt < vpt_ema)

                if price_trend_up and volume_flow_bearish: score -= 3   # Hidden distribution
                if price_trend_down and volume_flow_bullish: score -= 3 # Hidden accumulation (bearish signal)

                # Updated thresholds to reflect higher max score
                rec_col1, rec_col2 = st.columns(2)
                with rec_col1:
                    if score >= 15:
                        st.success(f"🔥 **STRONG BUY** – Very High Conviction (Score: {score})")
                    elif score >= 10:
                        st.success(f"✅ **BUY** – Bullish Setup (Score: {score})")
                    elif score <= -10:
                        st.error(f"⚠️ **STRONG SELL** – High Risk (Score: {score})")
                    elif score <= -5:
                        st.error(f"🔻 **SELL / REDUCE** (Score: {score})")
                    else:
                        st.warning(f"⚖️ **HOLD / WAIT** – Mixed Signals (Score: {score})")

                with rec_col2:
                    st.write("**Key Levels**")
                    st.write(f"• Near Support: ${max(price - recent_std*2, bb_lower.iloc[-1]):.2f}")
                    st.write(f"• 200D Floor: ${smas[200]:.2f}")
                    if sr_levels['support']:
                        st.write(f"• Major Support: ${min(sr_levels['support']):.2f}")
                    if sr_levels['resistance']:
                        st.write(f"• Next Resistance: ${min(sr_levels['resistance']):.2f}")

                st.divider()

            except Exception as e:
                st.error(f"Error processing {s}: {str(e)}")

        if len(stocks) > 1:
            st.header("📊 Multi-Stock Performance Comparison")
            comp_fig = go.Figure()
            for col in comparison_df.columns:
                label = col if col != primary_market else "Market (Primary)"
                comp_fig.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df[col], name=label))
            comp_fig.update_layout(height=500, title="Normalized Total Return")
            st.plotly_chart(comp_fig, width="stretch")




init_auth()

if not st.session_state.authenticated:
    login_page()
else:
    main_app()