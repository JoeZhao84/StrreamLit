import streamlit as st
import numpy as np
import pandas as pd


@st.cache_data
def load_data():
    '''
    dataframe example:
    Date Stock Price
    2025-01-01 00:00:00	AAPL	100.5
    2025-01-02 00:00:00	AAPL	100.36
    2025-01-03 00:00:00	AAPL	101.01
    '''
    np.random.seed(42)

    dates = pd.date_range(start="2025-01-01", periods=30, freq="D")
    stocks = ["AAPL", "MSFT", "GOOG"]

    data = []
    for stock in stocks:
        price = 100 + np.cumsum(np.random.randn(len(dates)))  # random walk
        for d, p in zip(dates, price):
            data.append({"Date": d, "Stock": stock, "Price": round(float(p), 2)})

    df = pd.DataFrame(data)
    return df
