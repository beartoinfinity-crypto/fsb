
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
    default_users = {"banana": hash_password("140484")}
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
    recent = df.tail(window).copy()
    if len(recent) < 10: return None
    lowest_idx = recent['Close'].idxmin()
    day1_pos = recent.index.get_loc(lowest_idx)
    if day1_pos > len(recent) - 4:
        return {"status": "Waiting", "msg": "Rally attempt in progress (too early for FTD)"}

    search_range = recent.iloc[day1_pos + 3 : day1_pos + 10] 
    for i in range(len(search_range)):
        current_day = search_range.iloc[i]
        prev_day = recent.iloc[recent.index.get_loc(search_range.index[i]) - 1]
        price_gain = (current_day['Close'] / prev_day['Close']) - 1
        vol_increase = current_day['Volume'] > prev_day['Volume']
        if price_gain >= 0.015 and vol_increase:
            undercut = recent.iloc[day1_pos : recent.index.get_loc(search_range.index[i])]['Close'].min() < recent.loc[lowest_idx, 'Close']
            if not undercut:
                return {"status": "VALID", "date": search_range.index[i].date(), "gain": f"{price_gain*100:.2f}%", "msg": f"Confirmed on {search_range.index[i].date()}"}
    return {"status": "None", "msg": "No valid FTD detected in this window."}

# --- EXIT STRATEGY FUNCTION (NEW) ---
def get_exit_strategy(price, s_series):
    recent_std = s_series.tail(30).std()
    take_profit = price + (recent_std * 2.5)
    stop_loss = price - (recent_std * 1.5)
    return take_profit, stop_loss

# --- 3. DATA FETCHING WITH CACHE ---
@st.cache_data(ttl=3600)
def fetch_data(tickers, period="5y"):
    return yf.download(tickers, period=period, auto_adjust=True)

# --- MAIN APP ---
def main_app():
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

    if st.session_state.get("show_user_management", False):
        user_management_page()
        return 

    st.sidebar.title("🚀 AI Market Intelligence Pro+")

    if st.session_state.is_admin:
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            if st.button("👤 Manage Users", key="main_mgmt"):
                st.session_state.show_user_management = True
                st.rerun()

    user_input = st.sidebar.text_input("Tickers (comma-separated)", "MSFT, NVDA").upper()
    backtest_date = st.sidebar.date_input("Backtest Date (Historical Mode)", value=None)
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
        stocks = ["TSLA", "MSFT", "NVDA", "GOOG", "AAPL", "AMZN", "AVGO", "CRWD"]
        st.info(f"💡 Running analysis for leaders: {', '.join(stocks)}")

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
                df = pd.DataFrame({'Close': close_data[s], 'High': high_data[s], 'Low': low_data[s], 'Volume': volume_data[s]}).ffill().dropna()
                if backtest_date:
                    df = df.loc[:pd.to_datetime(backtest_date)]
                    st.warning(f"🕰️ BACKTEST MODE: Analyzing {s} as of {df.index[-1].date()}")

                price = df['Close'].iloc[-1]
                last_date = df.index[-1]
                rsis = calculate_rsi(df['Close'])
                vol_data = analyze_volume_signals(df['Close'], df['Volume'])
                vpt, vpt_ema = calculate_vpt(df['Close'], df['Volume'])
                obv_val, obv_rising = calculate_obv(df['Close'], df['Volume'])
                k_val, d_val = calculate_kd(df['High'], df['Low'], df['Close'])
                macd_line, macd_signal, macd_hist = calculate_macd(df['Close'])
                bb_mid, bb_upper, bb_lower, bb_bw, bb_percent = calculate_bollinger(df['Close'])
                cmf_val = calculate_cmf(df)
                smas = {w: df['Close'].rolling(w).mean().iloc[-1] for w in [20, 50, 200]}
                recent_std = df['Close'].tail(30).std()
                sr_levels = find_support_resistance(df['Close']) if show_sr else {'support': [], 'resistance': []}
                rs_ratio, rs_rating = calculate_relative_strength(df['Close'], close_data[primary_market])
                ftd_data = detect_follow_through(df)

                # NEW: Exit Strategy Calculation
                tp_val, sl_val = get_exit_strategy(price, df['Close'])

                # --- PROPHET FORECAST ---
                regressor_df = pd.DataFrame(index=df.index)
                for ticker in market_tickers:
                    reg_name = ticker.replace('^', '')
                    regressor_df[reg_name] = close_data[ticker].ffill().reindex(df.index)

                # 2. Prepare the main training dataframe
                s_df = df['Close'].reset_index()
                s_df.columns = ['ds', 'y']  # Now it has exactly 2 columns
                s_df['ds'] = pd.to_datetime(s_df['ds']).dt.tz_localize(None)
                s_df = pd.concat([s_df, regressor_df.reset_index(drop=True)], axis=1)
                s_df = s_df.ffill().bfill()

                model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False,
                                growth='linear', changepoint_prior_scale=0.05)
                for reg in regressor_df.columns:
                    model.add_regressor(reg)
                model.fit(s_df)
                
                future_dates = model.make_future_dataframe(periods=forecast_days, freq='B')
                future_regressors = pd.DataFrame({'ds': future_dates['ds']})
                
                for ticker in market_tickers:
                    reg_name = ticker.replace('^', '')
                    
                    # 1. Properly format the regressor training data
                    m_train = close_data[ticker].ffill().loc[:df.index[-1]].reset_index()
                    m_train.columns = ['ds', 'y'] # Force exact naming
                    m_train['ds'] = pd.to_datetime(m_train['ds']).dt.tz_localize(None)
                    
                    # 2. Fit and predict the regressor's future path
                    m_model = Prophet().fit(m_train)
                    m_forecast = m_model.predict(future_dates)
                    future_regressors[reg_name] = m_forecast['yhat']
                
                # 3. Combine and generate final stock forecast
                forecast = model.predict(future_dates.merge(future_regressors, on='ds'))

                future_horizon = forecast.tail(forecast_days)

                if show_chart:
                    fig = go.Figure()
                    recent_df = df.tail(400)
                    
                    # 1. Bollinger Band Prediction (The Shaded Cloud)
                    fig.add_trace(go.Scatter(
                        x=future_horizon['ds'], y=future_horizon['yhat_upper'],
                        mode='lines', line=dict(width=0), showlegend=False, name='Upper Bound'
                    ))

                    # 1. Calculate Historical Bollinger Bands for the chart
                    # Using the same window (20) and std (2) as your calculation function
                    recent_df['MA20'] = recent_df['Close'].rolling(window=20).mean()
                    recent_df['STD20'] = recent_df['Close'].rolling(window=20).std()
                    recent_df['Upper'] = recent_df['MA20'] + (recent_df['STD20'] * 2)
                    recent_df['Lower'] = recent_df['MA20'] - (recent_df['STD20'] * 2)

                    # 2. Plot Historical BB (Past Data)
                    fig.add_trace(go.Scatter(
                        x=recent_df.index.tolist() + recent_df.index[::-1].tolist(),
                        y=recent_df['Upper'].tolist() + recent_df['Lower'][::-1].tolist(),
                        fill='toself',
                        fillcolor='rgba(128, 128, 128, 0.1)',
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        name='Historical BB'
                    ))

                    # 3. Plot Forecast BB Cloud (Future Data)
                    fig.add_trace(go.Scatter(
                        x=future_horizon['ds'].tolist() + future_horizon['ds'][::-1].tolist(),
                        y=future_horizon['yhat_upper'].tolist() + future_horizon['yhat_lower'][::-1].tolist(),
                        fill='toself',
                        fillcolor='rgba(0, 176, 246, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Forecast BB Cloud'
                    ))

                    # 2. Historical Price & 200 SMA
                    fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df['Close'], name="Price", line=dict(width=3)))
                    fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df['Close'].rolling(200).mean(), 
                                             name="200 SMA", line=dict(color='purple')))
                    
                    # 3. AI Price Forecast Line
                    fig.add_trace(go.Scatter(x=future_horizon['ds'], y=future_horizon['yhat'], 
                                             name="AI Price Forecast", line=dict(color='lime', dash='dot')))

                    # 4. Support/Resistance (10-Jan Version Logic)
                    if show_sr:
                        # Draw Support Levels (Green)
                        if sr_levels['support']:
                            fig.add_hline(y=sr_levels['support'][0], line_dash="dash", line_color="green")
                        if sr_levels['resistance']:
                            fig.add_hline(y=sr_levels['resistance'][0], line_dash="dash", line_color="red")

                    # Display the chart with a unique key
                    st.plotly_chart(fig, use_container_width=True, key=f"sr_chart_{s}")

                st.header(f"📊 {s} Analysis ({df.index[-1].date()})")
                c1, c2, c3 = st.columns([2, 2, 1])
                with c1:
                    st.subheader("🎯 Short-Term Targets")
                    rows = [[last_date.date(), "Current", f"${price:.2f}", "Base"]]
                    for days in [3, 7, 14, 30]:
                        pred = forecast[forecast['ds'] >= (last_date + timedelta(days=days))].iloc[0]
                        pct = (pred['yhat']/price - 1)*100
                        rows.append([pred['ds'].date(), f"{days}D", f"${pred['yhat']:.2f}", f"{pct:+.1f}%"])
                    st.table(pd.DataFrame(rows, columns=["Date", "Horizon", "Target", "Signal"]))

                with c2:
                    st.subheader("🛡️ Technical Dashboard")
                    ##st.write(f"RSI 14D: {rsis[14]:.1f} | KD K: {k_val:.1f}")
                    st.write(f"**RSI (14D) is {rsis[14]:.1f}:** " + 
                             ("Overbought. High risk of price correction." if rsis[14] > 70 else 
                              "Oversold. Potential value entry zone." if rsis[14] < 30 else "Neutral momentum."))
                    
                    st.write(f"**KD Index (K={k_val:.1f}):** " + 
                             ("Price is at top of recent range." if k_val > 80 else "Price is at bottom of range." if k_val < 20 else "Stable range."))

                    st.write(f"VPT: {'Accumulating 🟢' if vpt > vpt_ema else 'Distributing 🔴'}")
                    st.write(f"OBV: {'RISING 🟢' if obv_rising else 'FALLING 🔴'}")
                    st.write(f"CMF: {cmf_val:.2f} | VPT: {'🟢' if vpt > vpt_ema else '🔴'} " + 
                            (f"Strong accumulation (Money Flow IN)." if cmf_val > 0 else f"Heavy distribution (Money Flow OUT)."))

                    st.write(f"CMF(20): {cmf_val:+.3f} → {'Strong Buying 🟢' if cmf_val > 0.05 else 'Strong Selling 🔴' if cmf_val < -0.05 else 'Neutral'}")

                    st.write(f"MACD Hist: {macd_hist:+.3f} → {'🟢 Bullish' if macd_hist > 0 else '🔴 Bearish'}")
                    st.write(f"%B (BB): {bb_percent:.2f} → {'Overbought' if bb_percent > 0.8 else 'Oversold' if bb_percent < 0.2 else 'Neutral'}")
                    
                    # --- RVOL DISPLAY
                    rvol_val = vol_data['rvol']
                    rvol_status = "🔥 High Volume" if rvol_val > 1.5 else "❄️ Low Volume" if rvol_val < 0.7 else "Steady"
                    st.metric("Relative Volume (RVOL)", f"{rvol_val:.2f}x", rvol_status)

                    future_price_30d = future_horizon.iloc[min(len(future_horizon)-1, 30)]['yhat']
                    predicted_rsi_change = "Rising 📈" if future_price_30d > price else "Falling 📉"
                    
                    st.write(f"**Predicted RSI Trend (30D):** {predicted_rsi_change}")

                    st.write("**RSI Momentum Stack:**")
                    
                    rsi_display_cols = st.columns(len(rsis)) 
                    for i, (period, val) in enumerate(rsis.items()):
                        status = get_rsi_label(val)
                        label_color = "red" if ("PARABOLIC" in status or "OVERBOUGHT" in status) else "green" if "OVERSOLD" in status else "gray"
                        
                        rsi_display_cols[i].caption(f"{period}D")
                        rsi_display_cols[i].markdown(f":{label_color}[{val:.1f}]")
                        rsi_display_cols[i].caption(f":{label_color}[{status}]")
                                       
                    st.write(f"**Primary Signal:** {get_rsi_label(rsis[14])}")

                with c3:
                    st.subheader("💪 Pro Metrics")
                    st.metric("Relative Strength", f"{rs_ratio:.2f}x", f"{rs_rating:.0f} Rating")

                    ftd_color_val = "normal" if ftd_data["status"] == "VALID" else "inverse"
                    st.metric("Follow-Through Status", ftd_data["status"], ftd_data["msg"], delta_color=ftd_color_val)

                    st.metric("Price vs 200SMA", "Above" if price > smas[200] else "Below", f"{(price / smas[200] - 1)*100:+.1f}%")
                    
                    # NEW: EXIT STRATEGY UI
                    st.divider()
                    st.subheader("🚪 Exit Strategy")
                    st.success(f"**Take Profit:** ${tp_val:.2f}")
                    st.error(f"**Stop Loss:** ${sl_val:.2f}")
                    #tp, sl = get_exit_strategy(price, df['Close'])

                st.divider()
                st.subheader("🤖 AI Recommendation Engine")
                score = 0
                # 1. RSI Logic (Balanced)
                if rsis[14] < 30: score += 4
                elif rsis[14] > 70: score -= 4
                # 2. OBV/VPT Logic (Balanced)
                if obv_rising: score += 2
                else: score -= 2  # Subtract points if OBV is falling
                # 3. Trend Logic (Balanced)
                if price > smas[200]: score += 4
                else: score -= 4  # Subtract points if below 200 SMA (Bearish)
                # 4. FTD Logic
                if ftd_data["status"] == "VALID": score += 6
                # 5. CMF Logic (Added)
                if cmf_val > 0: score += 2
                else: score -= 2
                
                rec_col1, rec_col2 = st.columns(2)
                with rec_col1:
                    if score >= 12: 
                        st.success(f"🔥 **STRONG BUY** (Score: {score})")
                    elif score >= 6: 
                        st.success(f"✅ **BUY** (Score: {score})")
                    elif score <= -6: 
                        st.error(f"⚠️ **STRONG SELL** (Score: {score})")
                    elif score <= -2: 
                        st.error(f"🔻 **SELL / REDUCE** (Score: {score})")
                    else: 
                        st.warning(f"⚖️ **HOLD / WAIT** (Score: {score})")
                with rec_col2:
                    st.write("**Key Levels**")
                    st.write(f"• Near Support: ${max(price - recent_std*2, bb_lower.iloc[-1]):.2f}")
                    st.write(f"• 200D Floor: ${smas[200]:.2f}")

            except Exception as e: st.error(f"Error: {str(e)}")

        if len(stocks) > 1:
            st.header("📊 Multi-Stock Performance")
            comp_fig = go.Figure()
            for col in comparison_df.columns: comp_fig.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df[col], name=col))
            st.plotly_chart(comp_fig, use_container_width=True, key="multi_stock_comparison")

init_auth()
if not st.session_state.authenticated: login_page()
else: main_app()

