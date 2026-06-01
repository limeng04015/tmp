# ================================
# AI上涨概率分析系统 · 自动推荐TOP版
# 功能：批量分析观察池标的 → 按未来一周上涨概率排序 → 输出推荐榜单
# 数据来源：雅虎财经（延迟15‑20分钟，手机热点稳定）
# ================================
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ta.momentum import RSIIndicator
from ta.trend import MACD
import warnings

warnings.filterwarnings("ignore")

# ====================== 你的观察标的池（可自行增删） ======================
watch_pool = [
    # ETF
    "510300.SS",  # 沪深300ETF
    "159338.SZ",  # 中证A500ETF
    "515180.SS",  # 红利ETF
    "159819.SZ",  # 人工智能ETF
    "562500.SS",  # 机器人ETF
    "512760.SS",  # 芯片ETF
    # A股龙头
    "601868.SS",  # 中国能建
    "000725.SZ",  # 京东方A
    # 美股
    "AAPL",
    "QQQ"
]

# 周期：未来1天、未来一周
period_map = {"未来1天": 1, "未来一周": 5}


# ====================== 快速获取行情 ======================
def get_data(ticker):
    try:
        df = yf.Ticker(ticker).history(period="2y", interval="1d", auto_adjust=True)
        if df.empty:
            return None
        # 计算指标
        df["MA5"] = df["Close"].rolling(5).mean()
        df["MA20"] = df["Close"].rolling(20).mean()
        df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()
        macd = MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_SIGNAL"] = macd.macd_signal()
        df.dropna(inplace=True)
        return df
    except:
        return None


# ====================== 技术打分 ======================
def rule_score(df):
    last = df.iloc[-1]
    s = 0
    if last["MA5"] > last["MA20"]: s += 25
    if last["RSI"] > 50: s += 25
    if last["MACD"] > last["MACD_SIGNAL"]: s += 25
    if last["Close"] > df["Close"].iloc[-2]: s += 25
    return s


# ====================== 机器学习预测 ======================
def ml_pred(df, future_days):
    df["Future"] = (df["Close"].shift(-future_days) > df["Close"]).astype(int)
    feat = df[["MA5", "MA20", "RSI", "MACD", "MACD_SIGNAL"]].dropna()
    tgt = df["Future"].loc[feat.index]
    X_train, X_test, y_train, y_test = train_test_split(feat, tgt, test_size=0.2, shuffle=False)
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    prob = model.predict_proba(feat.iloc[[-1]])[0][1]
    acc = accuracy_score(y_test, model.predict(X_test))
    return round(prob * 100, 2), round(acc * 100, 2)


# ====================== 批量分析并收集结果 ======================
def run_pool():
    result_list = []
    for tick in watch_pool:
        df = get_data(tick)
        if df is None:
            continue
        base = rule_score(df)
        row = {"标的": tick}
        for name, days in period_map.items():
            ml, acc = ml_pred(df, days)
            total = round(base * 0.4 + ml * 0.6, 2)
            row[f"{name}概率"] = total
            row[f"{name}准确率"] = acc
        result_list.append(row)
    return pd.DataFrame(result_list)


# ====================== 主程序：输出TOP推荐 ======================
if __name__ == "__main__":
    df_result = run_pool()
    # 按未来一周上涨概率排序，取前8只推荐
    df_sort = df_result.sort_values("未来一周概率", ascending=False).head(8)

    print("=" * 60)
    print("🤖 AI选股推荐榜单（未来一周上涨概率TOP8）")
    print("=" * 60)
    print(df_sort[["标的", "未来一周概率", "未来一周准确率"]].to_string(index=False))
    print("\n💡 说明：概率越高，AI判断上涨可能性越大；数据延迟约15‑20分钟")