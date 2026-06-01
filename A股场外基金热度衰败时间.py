import pandas as pd
import numpy as np
import baostock as bs
import akshare as ak
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from datetime import datetime
import time
import warnings
warnings.filterwarnings("ignore")

# ===================== 全局配置 =====================
lg = bs.login()
print("baostock 登录成功，多因子系统启动")

# 标的池：名称，代码，类型，国家，赛道（无ETF）
symbol_list = [
    # 国内主动场外基金
    ("广发远见智选混合A", "016873", "fund", "CN", "AI科技"),
    ("易方达全球成长精选混合", "012922", "fund", "CN", "海外成长"),
    ("易方达中盘成长混合", "005875", "fund", "CN", "中盘成长"),
    ("大成成长进取混合A", "010371", "fund", "CN", "科创成长"),
    ("富国创新科技混合A", "002692", "fund", "CN", "科技成长"),
    ("华夏国证半导体混合", "008887", "fund", "CN", "半导体"),
    ("诺安成长混合", "320007", "fund", "CN", "芯片科技"),
    ("银河创新成长混合", "519674", "fund", "CN", "硬科技"),
    ("汇添富数字经济混合", "014156", "fund", "CN", "数字经济"),
    ("南方科技创新混合A", "007340", "fund", "CN", "科创龙头"),
    ("嘉实智能汽车混合", "002168", "fund", "CN", "智能汽车"),
    ("工银新能源混合", "001545", "fund", "CN", "新能源"),
    ("农银汇理新能源混合", "002190", "fund", "CN", "光伏锂电"),
    ("国泰智能装备混合", "001545", "fund", "CN", "高端装备"),
    ("中欧先进制造混合A", "004812", "fund", "CN", "制造升级"),
    ("易方达消费精选混合", "009265", "fund", "CN", "大消费"),
    ("汇添富消费行业混合", "000083", "fund", "CN", "必选消费"),
    ("博时消费创新混合", "010434", "fund", "CN", "消费复苏"),
    ("兴全合润混合", "163406", "fund", "CN", "均衡价值"),
    ("易方达蓝筹精选混合", "005827", "fund", "CN", "蓝筹成长"),
    ("富国天惠成长混合", "161005", "fund", "CN", "长期成长"),
    ("工银金融地产混合", "000251", "fund", "CN", "金融地产"),
    ("南方优选价值混合A", "202011", "fund", "CN", "价值低估"),
    ("鹏华匠心精选混合A", "009608", "fund", "CN", "均衡赛道"),
    ("景顺长城景气成长混合", "012783", "fund", "CN", "景气行业"),
    # QDII基金
    ("广发纳斯达克100联接A", "270042", "fund", "US", "美股科技"),
    ("博时标普500联接A", "050021", "fund", "US", "美股大盘"),
    ("华安纳斯达克科技精选", "013457", "fund", "US", "美股AI龙头"),
    ("大成纳斯达克精选混合", "008934", "fund", "US", "美股成长"),
    ("易方达美股成长混合", "011839", "fund", "US", "美股小盘成长"),
    ("华安标普全球石油", "160416", "fund", "US", "全球能源"),
    ("广发全球精选股票", "270023", "fund", "US", "全球龙头"),
    ("工银全球精选股票", "486001", "fund", "US", "海外均衡"),
    ("易方达欧洲精选混合", "013326", "fund", "EU", "欧洲消费"),
    ("广发欧洲精选混合", "014211", "fund", "EU", "欧洲制造"),
    ("工银日本精选混合", "005787", "fund", "JP", "日股科技"),
    ("富国日本精选混合", "012499", "fund", "JP", "日股消费"),
    ("汇添富东南亚精选", "012891", "fund", "SEA", "东南亚成长"),
    ("广发东南亚精选", "013928", "fund", "SEA", "东南亚制造"),
    ("博时亚太精选股票", "050015", "fund", "SEA", "亚太龙头"),
    ("易方达澳洲精选混合", "014532", "fund", "AU", "澳洲资源"),
    ("富国印度精选混合", "013149", "fund", "IN", "印度成长"),
    ("工银巴西精选混合", "012947", "fund", "BR", "拉美资源"),
    ("嘉实全球互联网股票", "000988", "fund", "GLB", "全球互联网"),
    ("汇添富全球移动互联", "001668", "fund", "GLB", "全球科技"),
    ("南方全球精选配置", "202801", "fund", "GLB", "全球均衡"),
    # A股+港股股票
    ("联瑞新材", "688300", "stock", "CN", "半导体材料"),
    ("光库科技", "300620", "stock", "CN", "光模块"),
    ("索辰科技", "688507", "stock", "CN", "工业软件"),
    ("中芯国际", "688981", "stock", "CN", "晶圆制造"),
    ("寒武纪", "688256", "stock", "CN", "AI芯片"),
    ("海光信息", "688041", "stock", "CN", "CPU/GPU"),
    ("兆易创新", "603986", "stock", "CN", "存储芯片"),
    ("韦尔股份", "603501", "stock", "CN", "半导体封测"),
    ("晶科能源", "688223", "stock", "CN", "光伏组件"),
    ("隆基绿能", "601012", "stock", "CN", "光伏龙头"),
    ("宁德时代", "300750", "stock", "CN", "动力电池"),
    ("比亚迪", "002594", "stock", "CN", "新能源车"),
    ("亿纬锂能", "300014", "stock", "CN", "储能电池"),
    ("贵州茅台", "600519", "stock", "CN", "白酒龙头"),
    ("五粮液", "000858", "stock", "CN", "高端白酒"),
    ("海天味业", "603288", "stock", "CN", "调味品"),
    ("伊利股份", "600887", "stock", "CN", "乳制品"),
    ("三一重工", "600031", "stock", "CN", "工程机械"),
    ("中联重科", "000157", "stock", "CN", "装备制造"),
    ("汇川技术", "300124", "stock", "CN", "工业自动化"),
    ("招商银行", "600036", "stock", "CN", "零售银行"),
    ("浙商银行", "601916", "stock", "CN", "股份制银行"),
    ("中国平安", "601318", "stock", "CN", "保险龙头"),
    ("万科A", "000002", "stock", "CN", "地产龙头"),
    ("长江电力", "600900", "stock", "CN", "水电龙头"),
    ("中国能建", "601868", "stock", "CN", "基建能源"),
    ("紫金矿业", "601899", "stock", "CN", "有色黄金"),
    ("京东方A", "000725", "stock", "CN", "面板龙头"),
    ("立讯精密", "002475", "stock", "CN", "消费电子"),
    ("歌尔股份", "002241", "stock", "CN", "声学硬件"),
    ("恒瑞医药", "600276", "stock", "CN", "创新药"),
    ("药明康德", "603259", "stock", "CN", "CXO"),
    ("爱尔眼科", "300015", "stock", "CN", "医疗服务"),
    ("腾讯控股", "0700", "stock", "HK", "互联网社交"),
    ("美团", "3690", "stock", "HK", "本地生活"),
    ("小米集团", "1810", "stock", "HK", "智能硬件"),
    ("阿里巴巴", "9988", "stock", "HK", "电商科技"),
    ("京东集团", "9618", "stock", "HK", "零售物流"),
    ("快手", "1024", "stock", "HK", "短视频"),
    ("网易", "9999", "stock", "HK", "游戏科技")
]

# 打分权重配置（总分100）
WEIGHT_TREND = 30
WEIGHT_VOLATILITY = 25
WEIGHT_CROWD = 20
WEIGHT_RECOVER = 15
WEIGHT_ML = 10

# ===================== 数据拉取 =====================
def get_data(name, code, s_type):
    try:
        if s_type == "stock":
            if len(code) == 4:
                bs_code = f"hk.{code}"
            else:
                bs_code = f"sh.{code}" if code.startswith(("5", "6")) else f"sz.{code}"
            rs = bs.query_history_k_data_plus(
                code=bs_code, fields="date,close,volume",
                start_date="2022-01-01", end_date=datetime.now().strftime("%Y-%m-%d"),
                frequency="d", adjustflag="3"
            )
            df = rs.get_data()
            df.rename(columns={"date":"Date","close":"Close","volume":"Volume"}, inplace=True)
            df["Close"] = df["Close"].astype(float)
            df["Volume"] = df["Volume"].astype(float)
        else:
            df = ak.fund_open_fund_info_em(code, "单位净值走势")
            df.rename(columns={"净值日期":"Date","单位净值":"Close"}, inplace=True)
            df["Volume"] = 0.0
            df["Close"] = df["Close"].astype(float)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        return df.dropna()
    except:
        return pd.DataFrame()

# ===================== 核心因子计算（修复未来函数：全部shift(1)） =====================
def calc_factors(df):
    # 基础收益
    df["ret"] = df["Close"].pct_change()
    # 均线
    df["ma5"] = df["Close"].rolling(5).mean().shift(1)
    df["ma20"] = df["Close"].rolling(20).mean().shift(1)
    df["ma60"] = df["Close"].rolling(60).mean().shift(1)
    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta>0,0).rolling(14).mean()
    loss = -delta.where(delta<0,0).rolling(14).mean()
    df["rsi"] = (100 - 100/(1+gain/loss)).shift(1)
    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = (ema12 - ema26).shift(1)
    # 趋势加速度：MA20斜率 > MA60斜率
    df["slope20"] = df["ma20"].diff(5).shift(1)
    df["slope60"] = df["ma60"].diff(5).shift(1)
    df["acc"] = (df["slope20"] > df["slope60"]).astype(int)
    # 反弹恢复率：当前/近2年最低
    low_2y = df["Close"].rolling(500).min()
    df["recovery"] = (df["Close"] / low_2y).shift(1)
    # 拥挤度1：波动率拥挤
    vol10 = df["ret"].rolling(10).std()
    vol120 = df["ret"].rolling(120).std()
    df["vol_crowd"] = (vol10 / vol120).shift(1)
    # 拥挤度2：净值偏离度
    df["dev60"] = ((df["Close"] - df["ma60"]) / df["ma60"]).shift(1)
    # 拥挤度3：量能拥挤（股票专用）
    vol_ma20 = df["Volume"].rolling(20).mean()
    df["vol_crowd_stock"] = (df["Volume"] / vol_ma20).shift(1)
    # 年化波动率（弹性）
    df["ann_vol"] = df["ret"].rolling(250).std() * np.sqrt(250)
    # 未来6个月趋势标签（用于机器学习）
    df["future_6m"] = df["Close"].shift(-120) / df["Close"] - 1
    return df.dropna()

# ===================== 多模型融合打分 =====================
def ml_score(df):
    feat = ["ma5","ma20","ma60","rsi","macd","acc","recovery","vol_crowd","dev60","ann_vol"]
    train = df.dropna(subset=["future_6m"])
    if len(train) < 100:
        return 5.0
    X = train[feat]
    y = train["future_6m"]
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    # 三模型融合
    lgbm = lgb.LGBMRegressor(n_estimators=80, max_depth=4, verbose=-1)
    xg = xgb.XGBRegressor(n_estimators=80, max_depth=4)
    rf = RandomForestRegressor(n_estimators=80, max_depth=4)
    lgbm.fit(X_sc, y)
    xg.fit(X_sc, y)
    rf.fit(X_sc, y)
    # 最新特征预测
    last_feat = scaler.transform(df[feat].iloc[-1:])
    pred = (lgbm.predict(last_feat)[0] + xg.predict(last_feat)[0] + rf.predict(last_feat)[0]) / 3
    # 映射为0-10分
    return np.clip((pred + 0.3) * 20, 0, 10)

# ===================== 翻倍潜力判定（强化版） =====================
def double_potential(df):
    if len(df) < 500:
        return "历史不足，无法判定"
    max_annual = df["ret"].rolling(250).max().max()
    rec = df["recovery"].iloc[-1]
    acc = df["acc"].iloc[-1]
    dev = df["dev60"].iloc[-1]
    if max_annual >= 0.8 and rec >= 1.2 and acc == 1 and dev < 0.25:
        return "✅ 高翻倍潜力（1年100%+）"
    elif max_annual >= 0.5 and rec >= 1.1:
        return "⚠️ 高弹性（50%-80%）"
    else:
        return "❌ 稳健型，难翻倍"

# ===================== 风控判定 =====================
def risk_check(df):
    close = df["Close"].iloc[-1]
    ma60 = df["ma60"].iloc[-1]
    if close < ma60 * 0.95:
        return "⚠️ 接近止损线，谨慎"
    elif close < ma60:
        return "❌ 跌破MA60，建议减仓"
    else:
        return "✅ 趋势安全"

# ===================== 主计算 =====================
def calc_total(df, s_type):
    df = calc_factors(df)
    if len(df) < 120:
        return np.nan, 0, 0, 0, 0
    # 趋势分
    trend = 0
    if df["ma20"].iloc[-1] > df["ma60"].iloc[-1]: trend += 12
    if df["acc"].iloc[-1] == 1: trend += 10
    if df["rsi"].iloc[-1] < 70: trend += 8
    # 弹性分
    vol = np.clip(df["ann_vol"].iloc[-1] * 100, 0, 25)
    # 拥挤度分
    crowd = 0
    if df["vol_crowd"].iloc[-1] < 2.5: crowd += 10
    if df["dev60"].iloc[-1] < 0.25: crowd += 10
    # 恢复力分
    rec = np.clip((df["recovery"].iloc[-1] - 1) * 15, 0, 15)
    # 机器学习分
    ml = ml_score(df)
    total = (trend/30*WEIGHT_TREND) + (vol/25*WEIGHT_VOLATILITY) + crowd + rec + (ml/10*WEIGHT_ML)
    return round(total,2), trend, vol, crowd, rec

# ===================== 执行 =====================
if __name__ == "__main__":
    res = []
    for name, code, s_type, country, sector in symbol_list:
        df = get_data(name, code, s_type)
        if len(df) < 150:
            res.append([name, code, s_type, country, sector, np.nan, "数据不足", "数据不足", 0, 0, 0])
            continue
        total, trend, vol, crowd, rec = calc_total(df, s_type)
        dp = double_potential(df) if s_type=="fund" else ""
        risk = risk_check(df)
        res.append([name, code, s_type, country, sector, total, dp, risk, trend, vol, crowd])
        time.sleep(0.1)
    bs.logout()

    df_out = pd.DataFrame(res, columns=[
        "标的","代码","类型","国家","赛道","综合得分(100)","基金翻倍潜力","风控提示","趋势分","弹性分","拥挤分"
    ])
    # 只对有效数字排序，NaN放最后
    df_out = df_out.sort_values("综合得分(100)", ascending=False, na_position="last")
    pd.set_option('display.width', 2000)
    print("\n===== 机构级多因子筛选结果（按上涨概率排序） =====")
    print(df_out.to_string(index=False))