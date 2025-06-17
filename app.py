import streamlit as st
import pandas as pd
import requests
import datetime
from prophet import Prophet
import yfinance as yf
from io import BytesIO

st.set_page_config(page_title="Altcoin AI Predictor", layout="wide")
st.title("🧠 Altcoin AI Price Predictor (1–24 Hour Forecasts)")
st.markdown("🔮 Enter altcoin symbol to get hourly price forecasts in beautiful tables.")

coin_input = st.text_input("💡 Enter Altcoin Symbol (e.g., BTC, PEPE)", "BTC").upper()

@st.cache_data
def load_coin_mapping():
    url = "https://api.coingecko.com/api/v3/coins/list"
    response = requests.get(url)
    data = response.json()
    return {coin['symbol'].upper(): coin['id'] for coin in data}

coin_map = load_coin_mapping()
time_slots = [1] + list(range(2, 25, 2))

if st.button("🔍 Predict Price Movement"):
    try:
        st.info("📥 Fetching market data...")
        df = yf.download(f"{coin_input}-USD", period="90d", interval="1h")

        if not df.empty:
            df.reset_index(inplace=True)
            df['Datetime'] = df['Datetime'].dt.tz_localize(None)
            df = df[['Datetime', 'Close']]
            df.columns = ['ds', 'y']
            st.success("✅ Data loaded from Yahoo Finance.")
        else:
            st.warning("⚠ Yahoo Finance failed. Trying CoinGecko...")
            coin_id = coin_map.get(coin_input)
            if coin_id is None:
                st.error("❌ Coin not found on CoinGecko.")
                st.stop()

            end = int(datetime.datetime.now().timestamp())
            start = end - 90 * 24 * 60 * 60
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
            params = {'vs_currency': 'usd', 'from': start, 'to': end}
            res = requests.get(url, params=params)
            data = res.json()
            prices = data['prices']
            df = pd.DataFrame(prices, columns=['ds', 'y'])
            df['ds'] = pd.to_datetime(df['ds'], unit='ms')
            st.success("✅ Data loaded from CoinGecko.")

        # Prophet Model
        st.info("🔮 Predicting future prices...")
        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=24, freq='H')
        forecast = m.predict(future)

        now = df['ds'].iloc[-1]
        future_data = forecast[forecast['ds'] > now].reset_index(drop=True)
        current_price = round(df['y'].iloc[-1], 5)

        all_tables = []

        st.header(f"📈 {coin_input} Forecast Tables (1–24 Hours)")

        for hour in time_slots:
            try:
                row = future_data.iloc[hour - 1]
                predicted = round(row['yhat'], 5)
                lower = round(row['yhat_lower'], 5)
                upper = round(row['yhat_upper'], 5)
                change = round(predicted - current_price, 5)
                pct_change = round((change / current_price) * 100, 2)
                trend = "🔺 Uptrend" if pct_change > 0 else "🔻 Downtrend"

                table = pd.DataFrame({
                    "Hour Ahead": [hour],
                    "Predicted Time": [row['ds'].strftime('%Y-%m-%d %H:%M')],
                    "Current Price (USD)": [current_price],
                    "Predicted Price (USD)": [predicted],
                    "Lower Prediction (Min Expected Price)": [lower],
                    "Upper Prediction (Max Expected Price)": [upper],
                    "Change (USD)": [change],
                    "Change (%)": [f"{pct_change}%"],
                    "Trend": [trend]
                })

                st.markdown(f"### ⏰ Forecast After {hour} Hour(s)")
                st.dataframe(table, use_container_width=True)
                all_tables.append(table)

            except:
                st.warning(f"⚠ Not enough data for {hour} hours.")

        # Excel download button
        if all_tables:
            full_df = pd.concat(all_tables, ignore_index=True)

            def convert_excel(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='Forecasts')
                return output.getvalue()

            excel_data = convert_excel(full_df)

            st.download_button(
                label="📥 Download All Forecasts (Excel)",
                data=excel_data,
                file_name=f"{coin_input}_forecast.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"❌ Error occurred: {e}")
