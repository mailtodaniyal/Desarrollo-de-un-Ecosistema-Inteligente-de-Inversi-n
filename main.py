import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

st.set_page_config(page_title="Forex Smart Ecosystem", layout="wide")

@st.cache_data
def load_data():
    url = "https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol=EUR&to_symbol=USD&interval=60min&apikey=demo&datatype=csv"
    df = pd.read_csv(url)
    df.columns = [col.strip().lower() for col in df.columns]
    if 'time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['time'])
    else:
        st.error("The API did not return expected data format.")
        st.stop()
    df = df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'})
    df = df.sort_values('timestamp')
    return df

def calculate_indicators(df):
    df['RSI'] = 100 - (100 / (1 + df['close'].pct_change().rolling(14).mean()))
    df['SMA_20'] = df['close'].rolling(20).mean()
    df['Bollinger_Upper'] = df['SMA_20'] + 2 * df['close'].rolling(20).std()
    df['Bollinger_Lower'] = df['SMA_20'] - 2 * df['close'].rolling(20).std()
    return df

def train_model(df):
    df = df.dropna()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    features = df[['close', 'RSI', 'SMA_20']]
    X = features[:-1]
    y = df['target'][:-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    joblib.dump(clf, 'forex_model.pkl')
    return acc

def predict(df):
    clf = joblib.load('forex_model.pkl')
    latest = df.dropna().iloc[-1:][['close', 'RSI', 'SMA_20']]
    return clf.predict(latest)[0]

def risk_management(df):
    drawdown = (df['close'] / df['close'].cummax()) - 1
    sharpe_ratio = (df['close'].pct_change().mean() / df['close'].pct_change().std()) * np.sqrt(252)
    return drawdown.min(), sharpe_ratio

def plot_dashboard(df):
    st.line_chart(df.set_index('timestamp')[['close', 'SMA_20', 'Bollinger_Upper', 'Bollinger_Lower']])
    st.area_chart(df.set_index('timestamp')[['RSI']])

def show_metrics(df):
    dd, sr = risk_management(df)
    st.metric("Max Drawdown", f"{dd:.2%}")
    st.metric("Sharpe Ratio", f"{sr:.2f}")

def broker_order_example():
    st.code("response = requests.post('https://api-fxtrade.oanda.com/v3/orders', headers=headers, json=payload)")

df = load_data()
df = calculate_indicators(df)

st.title("Ecosistema Inteligente de Inversión en Forex")

tabs = st.tabs(["📈 Dashboard", "🧠 IA & Backtest", "⚙️ Risk Mgmt", "🔗 Broker API"])

with tabs[0]:
    st.subheader("Indicadores Técnicos")
    plot_dashboard(df)
    show_metrics(df)

with tabs[1]:
    st.subheader("Modelo de IA para predicción de movimiento")
    accuracy = train_model(df)
    st.success(f"Precisión del modelo: {accuracy:.2%}")
    prediction = predict(df)
    st.info(f"Predicción: {'Subirá' if prediction == 1 else 'Bajará'}")

with tabs[2]:
    st.subheader("Gestión de Riesgo Centralizada")
    dd, sr = risk_management(df)
    st.write(f"Max Drawdown: {dd:.2%}")
    st.write(f"Sharpe Ratio: {sr:.2f}")
    st.write("Límites de exposición, Stop Loss global, y VaR simulados")

with tabs[3]:
    st.subheader("Ejemplo de Integración con Broker API")
    broker_order_example()
    st.write("Integración con MT4/MT5, FIX API, Binance u otros brokers a través de REST/WS")
