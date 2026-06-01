# ================================
# AI多维度量化上涨概率分析系统【最终完整版·带秒级时间】
# 多重因子分析｜双周期预测｜中文清晰输出｜三类操作建议｜精确到秒数据时间
# 决策规则：可买入｜搏一搏｜坚决不买入
# 数据来源：Yahoo Finance（雅虎财经，手机热点稳定）
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

warnings.filterwarnings("ignore")

# ====================== 你的自选股标的池（含芯片ETF） ======================
watch_pool = [
    {"code": "159172.SZ", "name": "养殖ETF汇添富"},
    {"code": "516670.SS", "name": "畜牧养殖ETF招商"},
    {"code": "159865.SZ", "name": "养殖ETF国泰"},
    {"code": "159819.SZ", "name": "人工智能ETF易方达"},
    {"code": "512880.SS", "name": "证券ETF国泰"},
    {"code": "588000.SS", "name": "科创50ETF华夏"},
    {"code": "562500.SS", "name": "机器人ETF华夏"},
    {"code": "515980.SS", "name": "人工智能ETF华富"},
    {"code": "512480.SS", "name": "半导体ETF国联安"},
    {"code": "512760.SS", "name": "芯片ETF"},
    {"code": "513500.SS", "name": "标普500ETF博时"},
    {"code": "159845.SZ", "name": "中证1000ETF华夏"},
    {"code": "515050.SS", "name": "通信ETF华夏"},
    {"code": "515180.SS", "name": "红利ETF易方达"},
    {"code": "513980.SS", "name": "港股科技ETF景顺"},
    {"code": "000012.SZ", "name": "南玻A"},
    {"code": "688055.SS", "name": "龙腾光电"},
    {"code": "300259.SZ", "name": "新天科技"},
    {"code": "688223.SS", "name": "晶科能源"},
    {"code": "688300.SS", "name": "联瑞新材"},
    {"code": "300620.SZ", "name": "光库科技"},
    {"code": "688507.SS", "name": "索辰科技"},
    {"code": "600221.SS", "name": "海航控股"},
    {"code": "601916.SS", "name": "浙商银行"},
    {"code": "000725.SZ", "name": "京东方A"},
    {"code": "601868.SS", "name": "中国能建"},
    {"code": "600900.SS", "name": "长江电力"},
    {"code": "002003.SZ", "name": "伟星股份"}
]

period_map = {
    "短线(未来1天)": 1,
    "中线(未来1周)": 5
}


# ====================== 数据获取 ======================
def get_analysis_data(ticker):
    try:
        df = yf.Ticker(ticker).history(period="2y", interval="1d", auto_adjust=True)
        if df.empty:
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


# ====================== 技术打分 ======================
def get_technical_score(df):
    latest = df.iloc[-1]
    score = 0
    if latest["MA5"] > latest["MA20"]: score += 25
    if latest["RSI"] > 50: score += 25
    if latest["MACD"] > latest["MACD_SIGNAL"]: score += 25
    if latest["Close"] > df["Close"].iloc[-2]: score += 25
    return score


# ====================== 机器学习预测（异常兼容） ======================
def get_ml_probability(df, future_days):
    df["Future_Rise"] = (df["Close"].shift(-future_days) > df["Close"]).astype(int)
    features = df[["MA5", "MA20", "RSI", "MACD", "MACD_SIGNAL"]].dropna()
    target = df["Future_Rise"].loc[features.index]

    if len(target.unique()) < 2:
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)
    model = LogisticRegression(max_iter=3000)
    model.fit(X_train, y_train)

    rise_prob = model.predict_proba(features.iloc[[-1]])[0][1]
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return round(rise_prob * 100, 2), round(accuracy * 100, 2)


# ====================== 涨跌评级 ======================
def get_final_rating(prob):
    if prob >= 75:
        return "🔥 强势看涨"
    elif prob >= 60:
        return "📈 偏多看好"
    elif prob >= 45:
        return "⚖️ 震荡观望"
    elif prob >= 30:
        return "📉 偏空谨慎"
    else:
        return "❌ 明确看空"


# ====================== 核心：三类操作建议（风险决策） ======================
def get_action_suggest(prob):
    if prob >= 65:
        return "✅ 可以买入"
    elif prob >= 45:
        return "⚠️ 搏一搏（轻仓）"
    else:
        return "❌ 坚决不能买入"


# ====================== 批量分析 ======================
def run_full_analysis():
    result_list = []
    for item in watch_pool:
        code = item["code"]
        name = item["name"]
        df = get_analysis_data(code)
        if df is None:
            continue

        tech_score = get_technical_score(df)
        result_row = {
            "标的代码": code,
            "标的名称": name,
            "技术面得分": tech_score
        }

        for period_name, days in period_map.items():
            ml_prob, acc = get_ml_probability(df, days)
            if ml_prob is None:
                total_prob = tech_score
                result_row[f"{period_name}AI预测概率"] = "数据单一，无AI预测"
            else:
                total_prob = round(tech_score * 0.4 + ml_prob * 0.6, 2)
                result_row[f"{period_name}AI预测概率"] = ml_prob
                result_row[f"{period_name}综合上涨概率"] = total_prob
                result_row[f"{period_name}模型准确率"] = acc
                result_row[f"{period_name}最终评级"] = get_final_rating(total_prob)

        if "中线(未来1周)综合上涨概率" in result_row:
            result_row["最终操作建议"] = get_action_suggest(result_row["中线(未来1周)综合上涨概率"])
        else:
            result_row["最终操作建议"] = "数据不足，无法建议"

        result_list.append(result_row)
    return pd.DataFrame(result_list)


# ====================== 结果输出 + 分组展示 + 精确到秒更新时间 ======================
if __name__ == "__main__":
    # 获取当前精确到秒的更新时间
    update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df_result = run_full_analysis()
    df_sorted = df_result.sort_values("中线(未来1周)综合上涨概率", ascending=False)

    df_buy = df_sorted[df_sorted["最终操作建议"] == "✅ 可以买入"]
    df_gamble = df_sorted[df_sorted["最终操作建议"] == "⚠️ 搏一搏（轻仓）"]
    df_no = df_sorted[df_sorted["最终操作建议"] == "❌ 坚决不能买入"]

    print("=" * 140)
    print(f"📊 AI多维度量化分析决策报告｜数据来源：雅虎财经｜数据更新时间：{update_time}")
    print("=" * 140)
    print("排序规则：按未来一周综合上涨概率从高到低｜操作建议分为三类：✅可买入 / ⚠️搏一搏 / ❌坚决不买")
    print("=" * 140)

    print("\n【全标的量化总览】")
    print(df_sorted.to_string(
        index=False,
        columns=[
            "标的代码", "标的名称", "技术面得分",
            "中线(未来1周)综合上涨概率", "中线(未来1周)最终评级",
            "最终操作建议"
        ]
    ))

    print("\n" + "=" * 80)
    print("【✅ 建议可以买入标的】")
    print(df_buy[["标的代码", "标的名称", "中线(未来1周)综合上涨概率"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("【⚠️ 搏一搏（轻仓试错）标的】")
    print(df_gamble[["标的代码", "标的名称", "中线(未来1周)综合上涨概率"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("【❌ 坚决不能买入标的】")
    print(df_no[["标的代码", "标的名称", "中线(未来1周)综合上涨概率"]].to_string(index=False))

    print("\n" + "=" * 140)
    print("💡 权重说明：综合概率 = 技术面得分(40%) + 机器学习预测(60%)，AI预测占比更高")