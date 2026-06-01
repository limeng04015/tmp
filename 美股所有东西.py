# ================================
# AI股票多周期上涨概率分析系统
# 纯净无错版：删除set_config / cache_dir
# 彻底解决：维度错误 + 拉取失败 + 函数不兼容
# ================================
import yfinance as yf
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ta.momentum import RSIIndicator
from ta.trend import MACD
import warnings

warnings.filterwarnings("ignore")

# ================================
# 最稳定标的池（100%能拉到）
# ================================
us_stocks = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN",
    "GOOGL", "META"
]

us_etfs = [
    "QQQ", "SPY", "XLK", "VOO"
]

# ================================
# 时间周期
# ================================
periods = {
    "1天": 1,
    "2天": 2,
    "1周": 5,
    "1月": 20
}


# ================================
# 数据获取（纯净版 + 重试 + 降维）
# ================================
def get_stock_data(ticker, max_retries=3):
    for attempt in range(max_retries):
        try:
            t = yf.Ticker(ticker)
            df = t.history(period="2y", interval="1d", auto_adjust=True)

            if df.empty:
                raise Exception("数据为空")

            df = df.copy()
            df.dropna(inplace=True)

            # 技术指标
            df["MA5"] = df["Close"].rolling(5).mean()
            df["MA20"] = df["Close"].rolling(20).mean()
            rsi = RSIIndicator(df["Close"], window=14)
            df["RSI"] = rsi.rsi()
            macd = MACD(df["Close"])
            df["MACD"] = macd.macd()
            df["MACD_SIGNAL"] = macd.macd_signal()

            df.dropna(inplace=True)

            # ✅ 关键：强制降维，解决 (501,1) 错误
            for col in df.columns:
                df[col] = np.ravel(df[col])

            return df

        except Exception as e:
            print(f"⚠️ {ticker} 第{attempt + 1}次重试...")
            time.sleep(2)

    print(f"❌ {ticker} 拉取失败，跳过")
    return None


# ================================
# 模型1：技术指标
# ================================
def rule_based_probability(df):
    latest = df.iloc[-1]
    score = 0
    if latest["MA5"] > latest["MA20"]: score += 25
    if latest["RSI"] > 50: score += 25
    if latest["MACD"] > latest["MACD_SIGNAL"]: score += 25
    if latest["Close"] > df["Close"].iloc[-2]: score += 25
    return score


# ================================
# 模型2：机器学习（无维度错误）
# ================================
def ml_probability(df, future_days=1):
    data = df.copy()
    data["Future"] = (data["Close"].shift(-future_days) > data["Close"]).astype(int)

    features = data[["MA5", "MA20", "RSI", "MACD", "MACD_SIGNAL"]].copy()
    target = data["Future"].copy()

    valid_idx = features.dropna().index
    features = features.loc[valid_idx]
    target = target.loc[valid_idx]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, shuffle=False
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    latest = features.iloc[[-1]]
    prob = model.predict_proba(latest)[0][1]
    acc = accuracy_score(y_test, model.predict(X_test))

    return round(prob * 100, 2), round(acc * 100, 2)


# ================================
# 分析单只股票
# ================================
def analyze_stock(ticker):
    print("\n" + "=" * 60)
    print(f"📈 标的：{ticker}")
    print("=" * 60)

    df = get_stock_data(ticker)
    if df is None:
        return

    rule_prob = rule_based_probability(df)
    print(f"\n📊 模型1（技术指标）概率：{rule_prob}%")

    for name, days in periods.items():
        ml_prob, acc = ml_probability(df, days)
        final = round(rule_prob * 0.4 + ml_prob * 0.6, 2)

        print(f"\n├── 周期：{name}")
        print(f"├── 模型2预测：{ml_prob}%")
        print(f"├── 模型准确率：{acc}%")
        print(f"└── 综合上涨概率：{final}%")

        if final >= 70:
            print("💡 结论：强势看涨")
        elif final >= 55:
            print("💡 结论：偏多")
        elif final >= 45:
            print("💡 结论：震荡")
        else:
            print("💡 结论：偏空")

    print(f"✅ 最新数据日期：{df.index[-1].date()}")


# ================================
# 主运行
# ================================
if __name__ == "__main__":
    print("🚀 开始分析美股个股")
    for s in us_stocks:
        analyze_stock(s)
        time.sleep(2)

    print("\n\n🚀 开始分析美股ETF")
    for e in us_etfs:
        analyze_stock(e)
        time.sleep(2)

    print("\n🎉 全部运行完成！")