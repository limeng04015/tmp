import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

# ====================== 全局配置 ======================
FUND_POOL = {
    "016824": "永赢科技智选混合发起A",
    "016825": "永赢科技智选混合发起C",
    "009992": "信澳业绩驱动混合A",
    "009993": "信澳业绩驱动混合C",
    "011217": "财通成长优选混合A",
    "011218": "财通成长优选混合C",
    "007300": "诺安成长混合",
    "008887": "华夏国证半导体芯片ETF联接A",
    "008888": "华夏国证半导体芯片ETF联接C",
    "012768": "广发创新升级混合"
}
START_DATE = "2023-01-01"
END_DATE = "2026-05-29"
RISK_FREE = 0.025

# ====================== 核心工具函数（修复M简写问题） ======================
def industry_boom_score(fund_code):
    tech_funds = list(FUND_POOL.keys())
    return 0.82 if fund_code in tech_funds else 0.55

# 修复点：将 resample("M") 改为 resample("ME")
def return_2nd_derivative(return_series):
    if len(return_series) < 24:
        return 0.06
    month_ret = return_series.resample("ME").mean()
    mom_12m = month_ret.rolling(12).mean()
    accel = mom_12m.diff().mean()
    return max(min(accel, 0.22), -0.12)

def manager_quality_score(ann_ret, sharpe, ret_std):
    ret_score = min(ann_ret, 0.45) * 0.35
    sharpe_score = min(sharpe, 2.8) * 0.4
    stability_score = (1 - min(ret_std, 0.25)) * 0.25
    return max(ret_score + sharpe_score + stability_score, 0.2)

def crowding_penalty(fund_code):
    high_crowd = ["016824","016825","009992","009993"]
    mid_crowd = ["011217","011218"]
    if fund_code in high_crowd:
        return 0.18
    elif fund_code in mid_crowd:
        return 0.10
    return 0.04

def risk_safety_score(max_dd, fund_size):
    dd_score = max(0, 1 - max_dd / 0.5)
    if 5 <= fund_size <= 20:
        size_score = 0.85
    elif 2 <= fund_size <5 or 20 < fund_size <=50:
        size_score = 0.65
    else:
        size_score = 0.35
    return (dd_score * 0.6 + size_score * 0.4)

def fund_kelly(quality_score, vol):
    win_rate = 0.53 + quality_score * 0.14
    pl_ratio = 1.6 if vol < 0.28 else 1.25
    full_kelly = (win_rate * pl_ratio - (1 - win_rate)) / pl_ratio
    return round(max(full_kelly * 0.25, 0), 4)

def risk_budget_port(forward_ret, cov_matrix):
    n = len(forward_ret)
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    init = np.array([1/n] * n)
    def neg_risk_adjusted(w):
        ret = np.sum(forward_ret * w)
        vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        return -(ret - RISK_FREE) / vol
    res = minimize(neg_risk_adjusted, init, method="SLSQP", bounds=bounds, constraints=cons)
    return res

# ====================== 净值数据生成 ======================
np.random.seed(666)
fund_codes = list(FUND_POOL.keys())
n_days = (pd.to_datetime(END_DATE) - pd.to_datetime(START_DATE)).days

daily_returns = np.random.normal(0.0012, 0.018, size=(n_days, len(fund_codes)))
nav_data = pd.DataFrame(
    np.cumprod(1 + daily_returns, axis=0),
    index=pd.date_range(START_DATE, END_DATE, periods=n_days),
    columns=fund_codes
)

ret_df = nav_data.pct_change().dropna()
ann_ret = ret_df.mean() * 252
ret_std = ret_df.std() * np.sqrt(252)
sharpe = (ann_ret - RISK_FREE) / ret_std

max_dd_dict = {code: round(np.random.uniform(0.22, 0.48), 4) for code in fund_codes}
size_dict = {code: round(np.random.uniform(3, 45), 2) for code in fund_codes}

# ====================== 多因子打分 ======================
result_list = []
for code in fund_codes:
    name = FUND_POOL[code]
    ar = ann_ret[code]
    std = ret_std[code]
    shp = sharpe[code]
    max_dd = max_dd_dict[code]
    fund_size = size_dict[code]

    boom_score = industry_boom_score(code)
    accel_score = return_2nd_derivative(ret_df[code])
    qual_score = manager_quality_score(ar, shp, std)
    safety_score = risk_safety_score(max_dd, fund_size)
    crowd_pen = crowding_penalty(code)

    total_score = (boom_score * 0.3 + qual_score * 0.32 + accel_score * 0.18 + safety_score * 0.15) - crowd_pen * 0.05
    total_score = max(total_score, 0.2)
    forward_cagr = 0.07 + qual_score * 0.21 + boom_score * 0.12 - crowd_pen
    kelly_pos = fund_kelly(qual_score, std)

    result_list.append({
        "基金代码": code,
        "基金名称": name,
        "赛道景气分": round(boom_score, 4),
        "增长加速度分": round(accel_score, 4),
        "经理质量分": round(qual_score, 4),
        "风险安全分": round(safety_score, 4),
        "拥挤度惩罚": round(crowd_pen, 4),
        "2年综合预测得分": round(total_score, 4),
        "前瞻2年CAGR": round(forward_cagr, 4),
        "1/4凯利推荐仓位": kelly_pos
    })

df = pd.DataFrame(result_list).sort_values("2年综合预测得分", ascending=False).reset_index(drop=True)
top6_funds = df.head(6)
top6_codes = top6_funds["基金代码"].tolist()
top6_fwd = [df.loc[df["基金代码"] == c, "前瞻2年CAGR"].values[0] for c in top6_codes]
cov_mat = ret_df[top6_codes].cov() * 252
opt_weights = risk_budget_port(top6_fwd, cov_mat).x
df.loc[:5, "年度风险预算配置权重"] = np.round(opt_weights, 3)

# ====================== 输出结果 ======================
print("="*110)
print("2026-05-29 A股场外基金2年维度量化推荐表（科技成长赛道）")
print("="*110)
cols = ["基金代码","基金名称","赛道景气分","增长加速度分","经理质量分","风险安全分","拥挤度惩罚","2年综合预测得分","前瞻2年CAGR","1/4凯利推荐仓位","年度风险预算配置权重"]
print(df[cols].to_string(index=False))