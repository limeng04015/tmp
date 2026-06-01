import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# ETF池（可自行扩展）
# =========================================================
FUND_POOL = {
    "515000.SS": "科技ETF",
    "159801.SZ": "芯片ETF",
    "512980.SS": "传媒ETF",
    "515250.SS": "智能制造ETF",
    "515180.SS": "新能源ETF",
    "159991.SZ": "人工智能ETF",
    "159928.SZ": "消费ETF",
    "512400.SS": "有色ETF",
    "515220.SS": "煤炭ETF",
    "518880.SS": "黄金ETF",
    "513100.SS": "纳指ETF"
}

# =========================================================
# 参数
# =========================================================
LOOKBACK_MOM = 20
LOOKBACK_HEAT = 120
LOOKBACK_VOL = 60
PREDICT_DAYS = 15
TOP_N = 3

# 因子权重
WEIGHTS = {
    "momentum": 0.30,
    "sharpe": 0.25,
    "drawdown": 0.20,
    "heat": 0.15,
    "crowd": 0.10
}

# =========================================================
# 市场环境过滤 沪深300趋势过滤
# =========================================================
MARKET_BENCH = "510300.SS"

# =========================================================
# 工具函数
# =========================================================
def safe_arr(series):
    return series.dropna().values.astype(float)

# ---------------------------------------------------------
# 动量因子（增加异常捕获，彻底解决横盘回归报错）
# ---------------------------------------------------------
def calc_momentum(arr):
    if len(arr) < LOOKBACK_MOM:
        return 0.0, 0.0
    arr20 = arr[-LOOKBACK_MOM:]
    if len(np.unique(arr20)) <= 1:
        return 0.0, 0.0
    x = np.arange(len(arr20))
    y = np.log(arr20)
    try:
        slope = linregress(x, y).slope
    except Exception:
        slope = 0.0
    ret20 = arr20[-1] / arr20[0] - 1
    return slope, ret20

# ---------------------------------------------------------
# 热度分位
# ---------------------------------------------------------
def calc_heat(arr):
    if len(arr) < LOOKBACK_HEAT:
        return 0.5
    win = arr[-LOOKBACK_HEAT:]
    low = np.min(win)
    high = np.max(win)
    if high - low < 1e-8:
        return 0.5
    return (arr[-1] - low) / (high - low)

# ---------------------------------------------------------
# 风险指标（修复广播错误）
# ---------------------------------------------------------
def calc_risk(arr):
    if len(arr) < LOOKBACK_VOL:
        return 0.0, 0.0, 0.0
    rets = []
    for i in range(1, len(arr)):
        if arr[i-1] < 1e-8:
            rets.append(0.0)
        else:
            rets.append((arr[i] - arr[i-1]) / arr[i-1])
    rets = np.array(rets)

    vol = np.std(rets) * np.sqrt(252)
    cummax = np.maximum.accumulate(arr)
    dd = np.min(arr / cummax - 1)

    rf = 0.02
    if vol > 1e-8:
        sharpe = (np.mean(rets) * 252 - rf) / vol
    else:
        sharpe = 0.0
    return vol, dd, sharpe

# ---------------------------------------------------------
# 市场趋势过滤
# ---------------------------------------------------------
def market_filter():
    try:
        df = yf.download(MARKET_BENCH, period="300d", progress=False)
        arr = safe_arr(df["Close"])
        if len(arr) < 120:
            return False
        ma20 = np.mean(arr[-20:])
        ma60 = np.mean(arr[-60:])
        ma120 = np.mean(arr[-120:])
        return ma20 > ma60 > ma120
    except:
        return False

# ---------------------------------------------------------
# 历史未来收益统计（非未来函数）
# ---------------------------------------------------------
def historical_prediction(arr):
    if len(arr) < 220:
        return 0.0, 0.0
    future_returns = []
    for i in range(LOOKBACK_HEAT, len(arr) - PREDICT_DAYS):
        hist = arr[:i]
        slope, _ = calc_momentum(hist)
        heat = calc_heat(hist)
        if slope > 0 and heat > 0.6:
            fut = arr[i + PREDICT_DAYS] / arr[i] - 1
            future_returns.append(fut)
    if len(future_returns) == 0:
        return 0.0, 0.0
    win_rate = np.mean(np.array(future_returns) > 0)
    avg_ret = np.mean(future_returns)
    return win_rate, avg_ret

# ---------------------------------------------------------
# Rank标准化
# ---------------------------------------------------------
def rank_normalize(series):
    return pd.Series(series).rank(pct=True)

# =========================================================
# 主程序
# =========================================================
def main():
    print("=" * 80)
    print("机构级 ETF 多因子轮动系统")
    print("=" * 80)

    market_ok = market_filter()
    if not market_ok:
        print("\n当前市场环境偏弱")
        print("建议：")
        print("1. 降低仓位")
        print("2. 保守防守")
        print("3. 黄金ETF优先")
        print("4. 可空仓等待")
        print()
    else:
        print("\n当前市场环境：多头趋势")
        print("允许开启ETF轮动")
        print()

    raw = []
    for code, name in FUND_POOL.items():
        try:
            df = yf.download(code, period="300d", progress=False)
            arr = safe_arr(df["Close"])
            if len(arr) < 120:
                continue
            slope, ret20 = calc_momentum(arr)
            heat = calc_heat(arr)
            vol, dd, sharpe = calc_risk(arr)
            win_rate, future_ret = historical_prediction(arr)
            raw.append({
                "代码": code,
                "名称": name,
                "slope": slope,
                "ret20": ret20,
                "heat": heat,
                "vol": vol,
                "dd": dd,
                "sharpe": sharpe,
                "win_rate": win_rate,
                "future_ret": future_ret
            })
        except Exception as e:
            print(f"{code} 数据异常: {e}")

    if len(raw) == 0:
        print("无有效数据")
        return

    df = pd.DataFrame(raw)

    df["mom_rank"] = rank_normalize(df["ret20"])
    df["heat_rank"] = rank_normalize(df["heat"])
    df["sharpe_rank"] = rank_normalize(df["sharpe"])
    df["dd_rank"] = rank_normalize(-df["dd"])
    df["crowd_rank"] = rank_normalize(df["ret20"])

    df["score"] = (
        WEIGHTS["momentum"] * df["mom_rank"]
        + WEIGHTS["sharpe"] * df["sharpe_rank"]
        + WEIGHTS["drawdown"] * df["dd_rank"]
        + WEIGHTS["heat"] * df["heat_rank"]
        + WEIGHTS["crowd"] * (1 - df["crowd_rank"])
    )

    def suggest_position(score):
        if score >= 0.8:
            return "100%"
        elif score >= 0.65:
            return "70%"
        elif score >= 0.5:
            return "40%"
        else:
            return "观察"

    df["建议仓位"] = df["score"].apply(suggest_position)
    df = df.sort_values("score", ascending=False)

    output_cols = [
        "代码", "名称", "score", "ret20", "heat", "sharpe", "dd", "win_rate", "future_ret", "建议仓位"
    ]
    print("\n")
    print("=" * 120)
    print("ETF轮动综合排名")
    print("=" * 120)
    print(df[output_cols].round(4).to_string(index=False))

    print("\n")
    print("=" * 80)
    print(f"Top {TOP_N} ETF")
    print("=" * 80)
    top_df = df.head(TOP_N)
    print(top_df[["代码", "名称", "score", "建议仓位"]].to_string(index=False))

    print("\n")
    print("=" * 80)
    print("策略建议")
    print("=" * 80)
    if market_ok:
        print("当前适合进行ETF轮动")
        print("优先配置 Top Rank ETF")
        print("避免追涨拥挤ETF")
        print("建议5~15交易日轮动")
    else:
        print("当前市场弱势")
        print("建议防守")
        print("降低仓位")
        print("黄金ETF优先")

if __name__ == "__main__":
    main()