# ================================
# 每日自动多周期量化选股系统｜每日09:00自动运行｜日志保存｜无推送
# 周期：日内(1天) / 周内(5天) / 月内(20天) / 年内(250天)
# 决策：✅可买入 / ⚠️搏一搏 / ❌坚决不买
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
from datetime import datetime
import time

warnings.filterwarnings("ignore")

# ====================== 配置区 ======================
# 每日9点整定时运行
RUN_HOUR = 9
RUN_MINUTE = 0

# 候选股票池（自选 + 热门ETF）
candidate_pool = [
    {"code":"159172.SZ","name":"养殖ETF汇添富"},
    {"code":"516670.SS","name":"畜牧养殖ETF招商"},
    {"code":"159865.SZ","name":"养殖ETF国泰"},
    {"code":"159819.SZ","name":"人工智能ETF易方达"},
    {"code":"512880.SS","name":"证券ETF国泰"},
    {"code":"588000.SS","name":"科创50ETF华夏"},
    {"code":"562500.SS","name":"机器人ETF华夏"},
    {"code":"515980.SS","name":"人工智能ETF华富"},
    {"code":"512480.SS","name":"半导体ETF国联安"},
    {"code":"512760.SS","name":"芯片ETF"},
    {"code":"513500.SS","name":"标普500ETF博时"},
    {"code":"159845.SZ","name":"中证1000ETF华夏"},
    {"code":"515050.SS","name":"通信ETF华夏"},
    {"code":"515180.SS","name":"红利ETF易方达"},
    {"code":"513980.SS","name":"港股科技ETF景顺"},
    {"code":"000012.SZ","name":"南玻A"},
    {"code":"688055.SS","name":"龙腾光电"},
    {"code":"300259.SZ","name":"新天科技"},
    {"code":"688223.SS","name":"晶科能源"},
    {"code":"688300.SS","name":"联瑞新材"},
    {"code":"300620.SZ","name":"光库科技"},
    {"code":"688507.SS","name":"索辰科技"},
    {"code":"600221.SS","name":"海航控股"},
    {"code":"601916.SS","name":"浙商银行"},
    {"code":"000725.SZ","name":"京东方A"},
    {"code":"601868.SS","name":"中国能建"},
    {"code":"600900.SS","name":"长江电力"},
    {"code":"002003.SZ","name":"伟星股份"},
    {"code":"513130.SS","name":"纳指ETF"},
    {"code":"515000.SS","name":"科创ETF"},
    {"code":"159801.SZ","name":"医药ETF"},
    {"code":"515700.SS","name":"新能源ETF"},
    {"code":"512690.SS","name":"国防ETF"},
    {"code":"512000.SS","name":"券商ETF"},
]

period_map = {
    "日内(1天)": 1,
    "周内(5天)": 5,
    "月内(20天)": 20,
    "年内(250天)": 250
}

# ====================== 工具函数 ======================
def get_data(ticker):
    try:
        df = yf.Ticker(ticker).history(period="3y", interval="1d", auto_adjust=True)
        if df.shape[0] < 300:
            return None
        df["MA5"] = df["Close"].rolling(5).mean()
        df["MA20"] = df["Close"].rolling(20).mean()
        df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()
        macd_obj = MACD(df["Close"])
        df["MACD"] = macd_obj.macd()
        df["MACD_SIGNAL"] = macd_obj.macd_signal()
        df.dropna(inplace=True)
        return df
    except:
        return None

def tech_score(df):
    last = df.iloc[-1]
    s = 0
    if last["MA5"] > last["MA20"]: s +=25
    if last["RSI"] > 50: s +=25
    if last["MACD"] > last["MACD_SIGNAL"]: s +=25
    if last["Close"] > df["Close"].iloc[-2]: s +=25
    return s

def ml_predict(df, future_days):
    df["Future"] = (df["Close"].shift(-future_days) > df["Close"]).astype(int)
    feat = df[["MA5","MA20","RSI","MACD","MACD_SIGNAL"]].dropna()
    tgt = df["Future"].loc[feat.index]

    if len(tgt.unique()) < 2:
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(feat, tgt, test_size=0.2, shuffle=False)
    model = LogisticRegression(max_iter=3000)
    model.fit(X_train, y_train)

    prob = model.predict_proba(feat.iloc[[-1]])[0][1]
    acc = accuracy_score(y_test, model.predict(X_test))
    return round(prob*100,2), round(acc*100,2)

def get_action(prob):
    if prob >=65: return "✅可买入"
    elif prob >=45: return "⚠️搏一搏(轻仓)"
    else: return "❌坚决不买"

# 执行一次选股
def run_one_scan():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    res = []
    for item in candidate_pool:
        code = item["code"]
        name = item["name"]
        df = get_data(code)
        if df is None:
            continue
        ts = tech_score(df)
        row = {"代码":code,"名称":name,"技术得分":ts}

        for p_name, days in period_map.items():
            mp, acc = ml_predict(df, days)
            if mp is None:
                row[f"{p_name}综合概率"] = "数据不足"
                row[f"{p_name}操作建议"] = "无法判断"
            else:
                total = round(ts*0.4 + mp*0.6, 2)
                row[f"{p_name}综合概率"] = total
                row[f"{p_name}操作建议"] = get_action(total)
        res.append(row)
    df = pd.DataFrame(res)
    df_sort = df.sort_values("周内(5天)综合概率", ascending=False)

    # 生成报告
    lines = []
    lines.append("="*120)
    lines.append(f"📅 每日多周期量化选股报告｜更新时间：{now}")
    lines.append("="*120)
    lines.append("\n【完整四周期结果】")
    lines.append(df_sort.to_string(index=False, columns=[
        "代码","名称",
        "日内(1天)综合概率","日内(1天)操作建议",
        "周内(5天)综合概率","周内(5天)操作建议",
        "月内(20天)综合概率","月内(20天)操作建议",
        "年内(250天)综合概率","年内(250天)操作建议"
    ]))

    full_text = "\n".join(lines)

    # 保存日志文件
    log_name = f"选股日志_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(log_name, "w", encoding="utf-8") as f:
        f.write(full_text)

    # 控制台打印
    print(full_text)
    return df_sort

# 定时主循环
if __name__=="__main__":
    print(f"✅ 定时选股程序已启动，启动时立即执行一次，之后每日 {RUN_HOUR}:{RUN_MINUTE} 自动运行")
    # 启动立刻跑一次
    run_one_scan()
    # 进入定时循环
    while True:
        now = datetime.now()
        if now.hour == RUN_HOUR and now.minute == RUN_MINUTE:
            run_one_scan()
            time.sleep(60)
        time.sleep(30)