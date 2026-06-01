import pandas as pd
import numpy as np
import baostock as bs
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from datetime import datetime
import time
import warnings
warnings.filterwarnings("ignore")

# ===================== 1. 登录 baostock =====================
lg = bs.login()
print(f"baostock 登录状态: {lg.error_msg}")

# ===================== 2. 标的清单 =====================
symbol_list = [
    ("广发远见智选混合A", "016873", "fund"),
    ("机器人ETF华夏", "562500", "etf"),
    ("联瑞新材", "688300", "stock"),
    ("人工智能ETF华富", "515980", "etf"),
    ("半导体ETF国联安", "512480", "etf"),
    ("光库科技", "300620", "stock"),
    ("索辰科技", "688507", "stock"),
    ("易方达全球成长精选混合", "012922", "fund"),
    ("标普500ETF博时", "513500", "etf"),
    ("中证1000ETF华夏", "159845", "etf"),
    ("通信ETF华夏", "515050", "etf"),
    ("养殖ETF汇添富", "159172", "etf"),
    ("畜牧养殖ETF招商", "516670", "etf"),
    ("养殖ETF国泰", "159865", "etf"),
    ("人工智能ETF易方达", "159819", "etf"),
    ("证券ETF国泰", "512880", "etf"),
    ("科创50ETF华夏", "588000", "etf"),
    ("南玻A", "000012", "stock"),
    ("龙腾光电", "688055", "stock"),
    ("新天科技", "300259", "stock"),
    ("晶科能源", "688223", "stock"),
    ("易方达中盘成长混合", "005875", "fund"),
    ("大成成长进取混合A", "010371", "fund"),
    ("富国创新科技混合A", "002692", "fund"),
    ("红利ETF易方达", "515180", "etf"),
    ("海航控股", "600221", "stock"),
    ("浙商银行", "601916", "stock"),
    ("京东方A", "000725", "stock"),
    ("中国能建", "601868", "stock"),
    ("港股科技ETF景顺", "513980", "etf"),
    ("长江电力", "600900", "stock"),
    ("伟星股份", "002003", "stock")
]

# ===================== 3. 模型参数（放宽数据门槛） =====================
LOOK_AHEAD_DAYS = 5
TRAIN_RATIO = 0.8
RANDOM_STATE = 42
FEATURE_COLS = ["MA5", "MA10", "MA20", "RSI", "MACD", "Volatility", "Volume_Ratio"]
MIN_DATA_LEN = 60  # 降低最低数据量要求，适配短数据标的

# ===================== 4. 修复编码+脏值清洗的数据拉取函数 =====================
def get_bs_data(name, code, s_type):
    try:
        # 修复编码：5开头ETF/6开头沪市 → sh；0/3/688深市 → sz
        if code.startswith(("5", "6")):
            bs_code = f"sh.{code}"
        else:
            bs_code = f"sz.{code}"

        if s_type in ["etf", "stock"]:
            # 拉长起始时间，获取更多历史数据
            rs = bs.query_history_k_data_plus(
                code=bs_code,
                fields="date,close,volume",
                start_date="2022-01-01",
                end_date=datetime.now().strftime("%Y-%m-%d"),
                frequency="d",
                adjustflag="3"
            )
            df = rs.get_data()
            df.rename(columns={"date":"Date","close":"Close","volume":"Volume"}, inplace=True)
            # 脏值清洗
            df["Close"] = df["Close"].replace("", np.nan).astype(float)
            df["Volume"] = df["Volume"].replace("", np.nan).astype(float)
        elif s_type == "fund":
            import akshare as ak
            df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
            df.rename(columns={"净值日期":"Date","单位净值":"Close"}, inplace=True)
            df["Volume"] = 0
            df["Close"] = df["Close"].astype(float)

        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df = df.sort_index()
        return df.dropna()
    except Exception as e:
        print(f"【{name} {code}】数据拉取失败：{str(e)}")
        return pd.DataFrame()

# ===================== 5. 技术指标计算 =====================
def calc_tech_features(df):
    df["pct_change"] = df["Close"].pct_change()
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta>0, 0)
    loss = -delta.where(delta<0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100/(1+rs))

    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2

    df["Volatility"] = df["pct_change"].rolling(10).std()

    if df["Volume"].sum() > 0:
        df["Volume_MA5"] = df["Volume"].rolling(5).mean()
        df["Volume_Ratio"] = df["Volume"] / df["Volume_MA5"]
    else:
        df["Volume_Ratio"] = 0

    df["future_5d_return"] = df["Close"].shift(-LOOK_AHEAD_DAYS)/df["Close"] - 1
    return df.dropna()

# ===================== 6. 随机森林预测（适配低数据量） =====================
def run_quant_model(df):
    if len(df) < MIN_DATA_LEN:
        return {"pred_return": np.nan, "test_r2": np.nan, "trend": "数据不足"}
    X = df[FEATURE_COLS]
    y = df["future_5d_return"]
    split_idx = int(len(df)*TRAIN_RATIO)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 低数据量下调小模型复杂度，避免过拟合
    model = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    test_r2 = r2_score(y_test, model.predict(X_test_scaled))
    latest_scaled = scaler.transform(X.iloc[-1:])
    pred_return = model.predict(latest_scaled)[0]

    if pred_return > 0.01:
        trend = "偏多（强趋势）"
    elif pred_return > 0:
        trend = "偏多（弱趋势）"
    elif pred_return > -0.01:
        trend = "震荡"
    else:
        trend = "偏空"
    return {"pred_return": pred_return, "test_r2": test_r2, "trend": trend}

# ===================== 7. 批量运行 =====================
if __name__ == "__main__":
    report_list = []
    update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"===== 量化分析更新时间：{update_time} =====")
    for name, code, s_type in symbol_list:
        df = get_bs_data(name, code, s_type)
        time.sleep(0.3)
        if len(df) == 0:
            report_list.append({
                "标的名称": name, "代码": code, "类型": s_type,
                "最新价": "无数据", "当日涨跌幅": "无数据",
                "预测5日收益": "无", "模型R²": "无", "趋势判断": "数据异常"
            })
            continue
        df = calc_tech_features(df)
        res = run_quant_model(df)
        if len(df) == 0:
            report_list.append({
                "标的名称": name, "代码": code, "类型": s_type,
                "最新价": round(df["Close"].iloc[-1],4) if len(df)>0 else "无",
                "当日涨跌幅": "无", "预测5日收益": "无", "模型R²": "无", "趋势判断": "指标计算后数据不足"
            })
            continue
        latest = df.iloc[-1]
        now_price = round(latest["Close"], 4)
        daily_pct = round(latest["pct_change"]*100, 2)
        report_list.append({
            "标的名称": name,
            "代码": code,
            "类型": s_type,
            "最新价": now_price,
            "当日涨跌幅(%)": daily_pct,
            "预测5日收益(%)": round(res["pred_return"]*100, 2) if not np.isnan(res["pred_return"]) else "无",
            "模型R²(拟合度)": round(res["test_r2"], 4) if not np.isnan(res["test_r2"]) else "无",
            "趋势判断": res["trend"]
        })
    report_df = pd.DataFrame(report_list)
    print("\n===== 全标的量化分析总表 =====")
    print(report_df.to_string(index=False))
    report_df.to_excel(f"自选股量化报告_{update_time.replace(':','-')}.xlsx", index=False)
    print(f"\n✅ 分析报告已保存为Excel文件，更新时间：{update_time}")
    bs.logout()