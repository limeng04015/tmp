import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# 基金池：A股行业ETF + 全球热门国家/地区ETF（覆盖主流海外市场）
# =========================================================
FUND_POOL = {
    # A股场内ETF
    "515000.SS": "半导体ETF",
    "159801.SZ": "AI算力ETF",
    "512980.SS": "数字经济ETF",
    "515250.SS": "机器人ETF",
    "515180.SS": "红利ETF",
    "159991.SZ": "医药ETF",
    "159928.SZ": "消费ETF",
    "512400.SS": "有色ETF",
    "515220.SS": "煤炭ETF",
    "518880.SS": "黄金ETF",
    "513100.SS": "纳指ETF(国内)",
    # 海外热门国家/地区ETF（美股上市，全球主流宽基）
    "SPY": "美国标普500ETF",
    "QQQ": "美国纳斯达克100ETF",
    "EWJ": "日本东证ETF",
    "EWG": "德国DAXETF",
    "EWU": "英国富时ETF",
    "EWL": "瑞士ETF",
    "EWA": "澳大利亚ETF",
    "EPI": "印度ETF",
    "EWT": "中国台湾ETF",
    "EWH": "中国香港ETF",
    "ILF": "巴西ETF"
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
# 动量因子（新版兼容+异常兜底，彻底解决回归报错）
# ---------------------------------------------------------
def calc_momentum(arr):
    try:
        arr = np.array(arr).astype(float).flatten()
        arr = arr[~np.isnan(arr)]
        if len(arr) < LOOKBACK_MOM:
            return 0.0, 0.0

        arr20 = arr[-LOOKBACK_MOM:]
        # 标准差判断横盘，比唯一值判断更稳定
        if np.std(arr20) < 1e-8:
            ret20 = arr20[-1] / arr20[0] - 1
            return 0.0, ret20

        x = np.arange(len(arr20)).astype(float)
        y = np.log(arr20)
        slope = linregress(x, y).slope
        ret20 = arr20[-1] / arr20[0] - 1

        return float(slope), float(ret20)
    except Exception as e:
        print(f"动量计算异常: {e}")
        return 0.0, 0.0

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
        # 新版yfinance兼容参数
        df = yf.download(
            MARKET_BENCH,
            period="300d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )
        close_data = df["Close"]
        # 兼容DataFrame/Series两种返回结构
        if isinstance(close_data, pd.DataFrame):
            close_data = close_data.iloc[:, 0]
        arr = safe_arr(close_data)
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
    print("机构级 全球ETF多因子轮动系统（A股+海外国家ETF）")
    print("=" * 80)

    market_ok = market_filter()
    if not market_ok:
        print("\n当前A股市场环境偏弱")
        print("建议：")
        print("1. 降低A股仓位，优先配置美股/黄金等避险标的")
        print("2. 关注日本、印度等独立行情国家ETF")
        print("3. 可空仓等待国内市场企稳")
        print()
    else:
        print("\n当前A股市场环境：多头趋势")
        print("允许开启全球ETF轮动，跨市场分散配置")
        print()

    raw = []
    for code, name in FUND_POOL.items():
        try:
            # 新版yfinance兼容拉取参数
            df = yf.download(
                code,
                period="300d",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False
            )
            close_data = df["Close"]
            # 核心兼容逻辑：处理DataFrame结构
            if isinstance(close_data, pd.DataFrame):
                close_data = close_data.iloc[:, 0]
            arr = safe_arr(close_data)

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
    print("全球ETF轮动综合排名（A股+海外国家ETF）")
    print("=" * 120)
    print(df[output_cols].round(4).to_string(index=False))

    print("\n")
    print("=" * 80)
    print(f"Top {TOP_N} ETF（跨市场最优标的）")
    print("=" * 80)
    top_df = df.head(TOP_N)
    print(top_df[["代码", "名称", "score", "建议仓位"]].to_string(index=False))

    print("\n")
    print("=" * 80)
    print("跨市场轮动策略建议")
    print("=" * 80)
    if market_ok:
        print("国内多头环境，可搭配海外高景气国家ETF分散风险")
        print("优先配置 Top Rank 标的，兼顾A股弹性与海外对冲")
        print("建议5~15交易日轮动，关注汇率与海外宏观事件")
    else:
        print("国内弱势环境，优先布局独立行情的海外国家ETF")
        print("黄金、美股科技、印度、日本ETF为优先防守选择")

if __name__ == "__main__":
    main()