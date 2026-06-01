import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

# ====================== 双轨标的池 ======================
CONSERVATIVE_POOL = {
    "NVDA": "美股-英伟达-AI算力GPU",
    "SMCI": "美股-超威电脑-AI服务器硬件",
    "AVGO": "美股-博通-高速数据中心芯片",
    "ASML": "欧股-阿斯麦-光刻机龙头",
    "SSNLF": "韩股-三星电子-AI存储芯片",
    "SONY": "日股-索尼-AI视觉传感器",
    "MSFT": "美股-微软-Azure大模型生态",
    "GOOGL": "美股-谷歌-Gemini云",
    "ISRG": "美股-直觉外科-医疗AI手术机器人",
    "INTC": "美股-英特尔-先进制程修复"
}

AGGRESSIVE_POOL = {
    **CONSERVATIVE_POOL,
    "0700.HK": "港股-腾讯控股-AI应用生态",
    "9988.HK": "港股-阿里巴巴-云服务",
    "JD": "美股ADR-京东-产业AI",
    "BABA": "美股ADR-阿里-AI云",
    "D05.SI": "新交所-星展科技-金融AI"
}

START_DATE = "2021-01-01"
END_DATE = "2026-05-29"
RISK_FREE = 0.04


# ====================== 核心工具函数（增强容错版） ======================
def macro_liquidity_score():
    return 0.62


def dynamic_industry_stage(ticker):
    try:
        tk = yf.Ticker(ticker)
        fin = tk.financials
        capex = fin.loc["Capital Expenditure"].dropna()
        capex_growth = capex.pct_change().mean() if len(capex) > 1 else 0
        if ticker in ["NVDA", "SMCI", "ASML", "AVGO", "SSNLF"]:
            tam = 0.83
        elif ticker in ["MSFT", "GOOGL", "ISRG"]:
            tam = 0.71
        else:
            tam = 0.58
        score = capex_growth * 0.4 + tam * 0.6
        return max(min(score, 0.9), 0.3)
    except:
        return 0.55


def revenue_2nd_derivative(ticker):
    try:
        tk = yf.Ticker(ticker)
        fin = tk.financials
        rev = fin.loc["Total Revenue"].dropna()
        rev_g = rev.pct_change().dropna()
        if len(rev_g) < 2:
            return 0.08
        accel = rev_g.diff().mean()
        return max(min(accel, 0.32), -0.15)
    except:
        return 0.05


def safe_get(info, key, default=0):
    val = info.get(key, default)
    if val is None or not isinstance(val, (int, float)):
        return default
    return val


def quality_growth_score(ticker):
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        accel = revenue_2nd_derivative(ticker)
        rnd = min(safe_get(info, "researchDevelopmentToRevenue", 0), 0.3)
        roic = min(safe_get(info, "returnOnEquity", 0), 0.26)
        fcf = min(safe_get(info, "freeCashflowYield", 0), 0.16)
        score = accel * 0.36 + rnd * 0.24 + roic * 0.21 + fcf * 0.19
        return max(score, 0.22)
    except:
        return 0.26


def expectation_gap_score(ticker):
    try:
        tk = yf.Ticker(ticker)
        act = safe_get(tk.info, "revenueGrowth", 0)
        est = safe_get(tk.info, "earningsGrowth", 0.15)
        gap = act - est
        return max(min(gap / 0.4 + 0.5, 0.75), 0.35)
    except:
        return 0.5


def valuation_safety(ticker):
    try:
        peg = safe_get(yf.Ticker(ticker).info, "pegRatio", 3)
        peg_score = max(0, 1 - (peg - 2.5) / 2)
        crowded = {"NVDA": 0.21, "SMCI": 0.19, "ASML": 0.13, "AVGO": 0.15, "SSNLF": 0.08, "MSFT": 0.06, "GOOGL": 0.06,
                   "ISRG": 0.04}
        return max(peg_score - crowded.get(ticker, 0), 0.12)
    except:
        return 0.23


def em_risk_penalty(ticker):
    em_list = ["0700.HK", "9988.HK", "JD", "BABA", "D05.SI"]
    return 0.12 if ticker in em_list else 0


def quarter_kelly(quality, vol):
    win_rate = 0.52 + quality * 0.13
    pl_ratio = 1.7 if vol < 0.3 else 1.3
    full_k = (win_rate * pl_ratio - (1 - win_rate)) / pl_ratio
    return round(max(full_k * 0.25, 0), 4)


def forward_5y_return(ticker, quality, val_score, is_aggressive=False):
    base = 0.08 + quality * 0.22
    adj = val_score * 0.06
    if is_aggressive:
        adj -= em_risk_penalty(ticker)
    return base + adj


def risk_budget_port(forward_ret, cov_matrix):
    n = len(forward_ret)
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    init = np.array([1 / n] * n)

    def neg_risk(w):
        ret = np.sum(forward_ret * w)
        vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        return -(ret - RISK_FREE) / vol

    res = minimize(neg_risk, init, method="SLSQP", bounds=bounds, constraints=cons)
    return res


# 新增：AI Beta 相关性聚类预警
def correlation_warning(cov, top_tickers):
    sub_cov = cov.loc[top_tickers, top_tickers]
    corr = sub_cov.corr()
    high_corr = {}
    for t in top_tickers:
        cnt = sum(corr[t].drop(t) > 0.75)
        high_corr[t] = cnt
    return high_corr


# ====================== 统一执行函数（修复列名兼容问题） ======================
def run_allocation(pool_dict, is_aggressive=False):
    tickers = list(pool_dict.keys())
    raw_data = yf.download(tickers, start=START_DATE, end=END_DATE)
    # 兼容两种列名：Adj Close / Adj. Close
    if "Adj. Close" in raw_data.columns:
        price = raw_data["Adj. Close"].dropna()
    elif "Adj Close" in raw_data.columns:
        price = raw_data["Adj Close"].dropna()
    else:
        price = raw_data["Close"].dropna()

    ret = price.pct_change().dropna()
    cov = ret.cov() * 252
    macro = macro_liquidity_score()
    res_list = []
    for ticker in tickers:
        name = pool_dict[ticker]
        vol = np.sqrt(cov.loc[ticker, ticker])
        ind = dynamic_industry_stage(ticker)
        accel = revenue_2nd_derivative(ticker)
        qual = quality_growth_score(ticker)
        eg = expectation_gap_score(ticker)
        val = valuation_safety(ticker)
        total = macro * 0.1 + ind * 0.28 + qual * 0.32 + eg * 0.2 + val * 0.1
        if is_aggressive:
            total -= em_risk_penalty(ticker)
        fwd = forward_5y_return(ticker, qual, val, is_aggressive)
        kelly = quarter_kelly(qual, vol)
        res_list.append({
            "代码": ticker,
            "标的全称": name,
            "动态产业阶段分": round(ind, 4),
            "增长加速度分": round(accel, 4),
            "护城河质量分": round(qual, 4),
            "预期差得分": round(eg, 4),
            "估值安全分": round(val, 4),
            "5年综合预测得分": round(total, 4),
            "前瞻5年CAGR": round(fwd, 4),
            "1/4凯利推荐仓位": kelly
        })
    df = pd.DataFrame(res_list).sort_values("5年综合预测得分", ascending=False).reset_index(drop=True)
    top6 = df.head(6)
    top6_ticks = top6["代码"].tolist()
    top6_fwd = [forward_5y_return(t, quality_growth_score(t), valuation_safety(t), is_aggressive) for t in top6_ticks]
    top6_cov = cov.loc[top6_ticks, top6_ticks]
    opt_w = risk_budget_port(top6_fwd, top6_cov).x
    df.loc[:5, "年度风险预算权重"] = np.round(opt_w, 3)
    corr_alert = correlation_warning(cov, top6_ticks)
    return df, corr_alert


# ====================== 执行并输出 ======================
df_conservative, corr_cons = run_allocation(CONSERVATIVE_POOL, False)
df_aggressive, corr_agg = run_allocation(AGGRESSIVE_POOL, True)

cols = ["代码", "标的全称", "动态产业阶段分", "增长加速度分", "护城河质量分", "预期差得分", "估值安全分",
        "5年综合预测得分", "前瞻5年CAGR", "1/4凯利推荐仓位", "年度风险预算权重"]

print("=" * 120)
print("【保守成熟市场配置表（机构优选，实盘推荐）】")
print("=" * 120)
print(df_conservative[cols].to_string(index=False))
print("\n⚠️ AI高相关性预警（同赛道标的，建议搭配非科技资产对冲）：", corr_cons)

print("\n\n" + "=" * 120)
print("【全市场激进配置表（含港股/ADR，风险补偿后）】")
print("=" * 120)
print(df_aggressive[cols].to_string(index=False))
print("\n⚠️ AI高相关性预警：", corr_agg)