import yfinance as yf
import time

watch_us = ["AAPL","DXYZ","QQQM","VOO","SPY"]

def get_signal(symbol):
    df = yf.download(symbol, period="20d", interval="1d", auto_adjust=True, progress=False)
    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma10"] = df["Close"].rolling(10).mean()
    latest = df.iloc[-1]
    # 用 .item() 提取数值，解决报错
    return "✅可入场" if latest["ma5"].item() > latest["ma10"].item() else "❌观望"

while True:
    print("\n===== 美股实时信号 =====")
    for s in watch_us:
        res = get_signal(s)
        print(f"{s:8s} → {res}")
    time.sleep(3600)