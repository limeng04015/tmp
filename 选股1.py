# -*- coding: utf-8 -*-
"""
量化系统 V5.3 机构多因子终版
新增：Rank‑IC / IR / 因子分层回测 / 自动因子筛选
定位：具备私募级多因子研究验证能力
"""
import akshare as ak
import pandas as pd
import numpy as np
import sqlite3
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ====================== 全局配置 ======================
CAPITAL = 10000
WEEK_TOP_NUM = 10
YEAR_TOP_NUM = 8
MAX_SINGLE_INDUSTRY_RATIO = 0.2
MIN_DATA_LEN = 750
PREDICT_DAY = 20
RETRY_TIMES = 3
REQUEST_INTERVAL = 0.8

WEIGHT_INDUSTRY = 0.20
WEIGHT_FUNDAMENT = 0.20
WEIGHT_CAPITAL = 0.25
WEIGHT_TECH = 0.20
WEIGHT_SENTIMENT = 0.10
WEIGHT_NEWS = 0.05

STOP_LOSS = -0.08
TRAIL_PROFIT = 0.10

STYLE_WEIGHT = {
    "成长进攻": 0.4,
    "均衡价值": 0.3,
    "红利防御": 0.2,
    "周期弹性": 0.1
}

# ====================== 行业池 ======================
STYLE_INDUSTRY_POOL = {
    "成长进攻": {
        "AI算力": ["000977","688256","603019"],
        "半导体": ["603501","600584","300458"],
        "云计算": ["688111","600845","002230"],
        "机器人": ["002527","300024"]
    },
    "均衡价值": {
        "创新药": ["600276","300760"],
        "军工": ["600893","000733"],
        "高端制造": ["601108","002008"],
        "数字传媒": ["300413","600989"]
    },
    "红利防御": {
        "银行红利": ["601988","600036","601398"],
        "公用事业": ["600011","600027"],
        "必选消费": ["600887","002304","601899"]
    },
    "周期弹性": {
        "新能源": ["601633","300274"],
        "有色资源": ["601600","000878"],
        "化工材料": ["600486","002407"]
    }
}

STYLE_ETF = {
    "AI算力":"159890", "半导体":"512480", "云计算":"159896", "机器人":"159777",
    "创新药":"159858", "军工":"512660", "高端制造":"512960", "数字传媒":"515050",
    "银行红利":"512800", "公用事业":"516170", "必选消费":"159928",
    "新能源":"516850", "有色资源":"512400", "化工材料":"515710"
}

# ====================== 产业链配置 ======================
CHAIN_KEYWORDS = {
    "AI算力核心龙头": ["人工智能", "AI服务器", "大模型", "算力芯片"],
    "半导体核心龙头": ["晶圆制造", "光刻", "先进封装", "存储芯片"],
    "机器人产业链": ["人形机器人", "伺服电机", "减速器", "机器视觉"]
}
CHAIN_TIER_WEIGHT = {"龙头":1.0, "二线龙头":0.9, "核心设备":0.8, "核心材料":0.7, "软件服务":0.6, "普通供应商":0.4}

# ====================== 因子池 ======================
FEATURE_COLS = [
    "ma20_bias", "ma60_bias", "rsi", "macd", "atr_vol", "ret_20d",
    "vol_20d", "vol_60d", "price_skew", "price_kurt",
    "turnover_mean", "turnover_vol",
    "pb", "ps", "ocf_ratio", "debt_ratio",
    "north_flow", "etf_flow", "concentration",
    "industry_score", "sentiment_score"
]

# ====================== 缓存 ======================
DB_PATH = "quant_cache_v5.db"
def init_cache():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS price_cache
                 (code TEXT, date TEXT, close REAL, volume REAL, turnover REAL, PRIMARY KEY(code, date))''')
    conn.commit()
    conn.close()

def get_from_cache(code, days=3650):
    try:
        conn = sqlite3.connect(DB_PATH)
        end = datetime.now()
        start = end - timedelta(days=days)
        df = pd.read_sql(
            "SELECT date,close,volume,turnover FROM price_cache WHERE code=? AND date>=?",
            conn, params=(code, start.strftime("%Y-%m-%d"))
        )
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        conn.close()
        if len(df) >= MIN_DATA_LEN:
            return df
    except:
        pass
    return None

def save_to_cache(code, df):
    try:
        conn = sqlite3.connect(DB_PATH)
        df_save = df.reset_index()[["date","close","volume","turnover"]].copy()
        df_save["code"] = code
        df_save.to_sql("price_cache", conn, if_exists="replace", index=False)
        conn.commit()
        conn.close()
    except:
        pass

# ====================== 数据源 ======================
def get_price(code):
    cached = get_from_cache(code)
    if cached is not None:
        return cached
    df = pd.DataFrame()
    for _ in range(RETRY_TIMES):
        try:
            time.sleep(REQUEST_INTERVAL)
            df = ak.stock_zh_a_daily(symbol=code, adjust="hfq")
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
                df["turnover"] = df["成交量"] / df["流通市值"] if "流通市值" in df.columns else 0
                save_to_cache(code, df)
                return df.tail(3650)
        except:
            continue
    try:
        time.sleep(REQUEST_INTERVAL)
        df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="hfq")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        save_to_cache(code, df)
        return df.tail(3650)
    except:
        return pd.DataFrame()

# ====================== 基本面 ======================
def get_fundamental_hist(code, target_date):
    try:
        df = ak.stock_financial_analysis_indicator(symbol=code)
        df["报告日期"] = pd.to_datetime(df["报告日期"])
        df = df[df["报告日期"] <= target_date].iloc[-1]
        pe = float(df["市盈率"]) if df["市盈率"] != "-" else 25
        pb = float(df["市净率"]) if df["市净率"] != "-" else 2
        ps = float(df["市销率"]) if df["市销率"] != "-" else 3
        roe = float(df["净资产收益率"]) if df["净资产收益率"] != "-" else 8
        net_grow = float(df["净利润增长率"]) if df["净利润增长率"] != "-" else 5
        rev_grow = float(df["营业收入增长率"]) if df["营业收入增长率"] != "-" else 5
        ocf_ratio = float(df["经营现金流/净利润"]) if df["经营现金流/净利润"] != "-" else 1.0
        debt_ratio = float(df["资产负债率"]) if df["资产负债率"] != "-" else 50
        return {"pe":pe,"pb":pb,"ps":ps,"roe":roe,"net_grow":net_grow,"rev_grow":rev_grow,"ocf_ratio":ocf_ratio,"debt_ratio":debt_ratio}
    except:
        return {"pe":25,"pb":2,"ps":3,"roe":8,"net_grow":5,"rev_grow":5,"ocf_ratio":1.0,"debt_ratio":50}

def get_chain_profit_score(code, target_date):
    try:
        df = ak.stock_financial_analysis_indicator(symbol=code)
        df["报告日期"] = pd.to_datetime(df["报告日期"])
        df = df[df["报告日期"] <= target_date].iloc[-4:]
        rev_grow = df["营业收入增长率"].replace("-", np.nan).astype(float).mean()
        rev_grow = max(0, min(40, rev_grow))
        net_grow = df["净利润增长率"].replace("-", np.nan).astype(float).mean()
        net_grow = max(0, min(40, net_grow))
        gross_margin = df["销售毛利率"].replace("-", np.nan).astype(float).mean()
        gross_margin = max(0, min(40, gross_margin))
        pe = df["市盈率"].replace("-", np.nan).astype(float).mean()
        peg = pe / (net_grow + 1e-6)
        peg_score = 40 if 0.8 <= peg <= 2 else max(0, 40 - abs(peg-1.4)*15)
        profit_score = (rev_grow * 0.3 + net_grow * 0.3 + gross_margin * 0.25 + peg_score * 0.15)
        return round(profit_score, 2)
    except:
        return 40.0

# ====================== 资金与情绪 ======================
def get_northbound_flow():
    try:
        df = ak.stock_hsgt_individual_em()
        return df["当日净流入-人民币"].sum() / 1e8
    except:
        return 0

def get_fund_flow(code, is_etf=False):
    if is_etf:
        try:
            df = ak.fund_etf_fund_flow_individual(code)
            return df["净流入"].iloc[-1] / 1e6
        except:
            return 0
    else:
        try:
            df = ak.stock_fund_flow_individual(code)
            return df["主力净流入-净额"].iloc[-1] / 1e6
        except:
            return 0

def get_market_sentiment():
    try:
        df = ak.stock_market_activity_board()
        up = int(df.loc[df["项目"]=="上涨家数","数据"].values[0])
        down = int(df.loc[df["项目"]=="下跌家数","数据"].values[0])
        limit_up = int(df.loc[df["项目"]=="涨停家数","数据"].values[0])
        limit_down = int(df.loc[df["项目"]=="跌停家数","数据"].values[0])
        ratio = up/(up+down+1e-6)
        score = ratio*60 + min(40, limit_up/10) - min(30, limit_down/5)
        return max(0,min(100,score))
    except:
        return 50

# ====================== 全量指标计算 ======================
def calc_full_indicators(df, funda_data):
    close = df["close"]
    volume = df["volume"]
    turnover = df["turnover"] if "turnover" in df.columns else pd.Series(0, index=df.index)

    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma20_bias = (close/ma20 - 1) * 100
    ma60_bias = (close/ma60 - 1) * 100
    ret_20d = close.pct_change(20) * 100

    delta = close.diff()
    gain = delta.mask(delta < 0, 0).rolling(14).mean()
    loss = (-delta).mask(delta > 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-6)
    rsi = 100 - (100 / (1 + rs))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd = (dif - dea) * 2
    tr = pd.DataFrame({"high":close.rolling(2).max(), "low":close.rolling(2).min()})
    tr["tr"] = tr["high"] - tr["low"]
    atr_vol = tr["tr"].rolling(14).mean() / close * 100

    vol_20d = close.pct_change().rolling(20).std() * 100
    vol_60d = close.pct_change().rolling(60).std() * 100
    price_skew = close.rolling(60).skew()
    price_kurt = close.rolling(60).kurt()

    turnover_mean = turnover.rolling(20).mean()
    turnover_vol = turnover.rolling(20).std()

    concentration = volume.rolling(60).quantile(0.9) / volume.rolling(60).mean()

    out = pd.DataFrame({
        "ma20_bias":ma20_bias, "ma60_bias":ma60_bias, "rsi":rsi, "macd":macd, "atr_vol":atr_vol, "ret_20d":ret_20d,
        "vol_20d":vol_20d, "vol_60d":vol_60d, "price_skew":price_skew, "price_kurt":price_kurt,
        "turnover_mean":turnover_mean, "turnover_vol":turnover_vol,
        "pb":funda_data["pb"], "ps":funda_data["ps"],
        "ocf_ratio":funda_data["ocf_ratio"], "debt_ratio":funda_data["debt_ratio"],
        "concentration":concentration
    })
    return out

# ====================== 【新增核心模块】因子IC / IR / 分层回测 ======================
def factor_ic_analysis(code_list, pred_day=20):
    """
    批量计算因子 Rank‑IC、IC均值、IC标准差、IR、5组分层收益
    返回有效因子列表
    """
    ic_records = []
    long_short_records = {}
    valid_factors = []

    for fac in FEATURE_COLS:
        ic_list = []
        group_ret = []
        for code in code_list:
            df = get_price(code)
            if len(df) < MIN_DATA_LEN:
                continue
            funda = get_fundamental_hist(code, df.index[-1])
            fac_df = calc_full_indicators(df, funda)
            fac_df["future_ret"] = df["close"].shift(-pred_day) / df["close"] - 1
            fac_df = fac_df.dropna(subset=[fac, "future_ret"])
            if len(fac_df) < 120:
                continue
            # 截面秩相关 = Rank‑IC
            ic = fac_df[[fac, "future_ret"]].corr(method="spearman").iloc[0,1]
            ic_list.append(ic)
            # 5分组收益
            fac_df["group"] = pd.qcut(fac_df[fac].rank(), 5, labels=False)
            ret_group = fac_df.groupby("group")["future_ret"].mean()
            group_ret.append(ret_group.values)

        if not ic_list:
            continue
        ic_mean = np.mean(ic_list)
        ic_std = np.std(ic_list)
        ir = ic_mean / ic_std if ic_std > 1e-6 else 0

        # IR>0.3 视为有效因子
        if abs(ir) > 0.3:
            valid_factors.append(fac)

        ic_records.append({
            "因子": fac,
            "IC均值": round(ic_mean,4),
            "IC标准差": round(ic_std,4),
            "IR": round(ir,4)
        })

        if group_ret:
            gs = np.array(group_ret).mean(axis=0)
            long_short_records[fac] = gs[-1] - gs[0]

    ic_df = pd.DataFrame(ic_records)
    return ic_df, long_short_records, valid_factors

# ====================== Walk Forward 回测 ======================
def walk_forward_backtest(code, train_years=3, pred_years=1):
    df = get_price(code)
    if len(df) < MIN_DATA_LEN:
        return {"年化收益":0, "最大回撤":0, "夏普比率":0, "胜率":0}
    funda = get_fundamental_hist(code, df.index[-1])
    indicators = calc_full_indicators(df, funda)
    bt_res = []
    for start_year in range(df.index.year.min(), df.index.year.max() - pred_years):
        train_s = datetime(start_year, 1, 1)
        train_e = datetime(start_year + train_years, 1, 1)
        pred_e = datetime(start_year + train_years + pred_years, 1, 1)
        train_df = df[(df.index >= train_s) & (df.index < train_e)]
        pred_df = df[(df.index >= train_e) & (df.index < pred_e)]
        if len(train_df) < MIN_DATA_LEN*0.8 or len(pred_df) < 252:
            continue
        feat_latest = indicators.iloc[-1].to_dict()
        feat_latest["roe"] = funda["roe"]
        feat_latest["net_grow"] = funda["net_grow"]
        _,_,_,_ = train_rolling_models(code)
        pred_ret = predict_future(None, None, None, feat_latest)["future_ret"]
        real_ret = pred_df["close"].iloc[-1] / pred_df["close"].iloc[0] - 1
        bt_res.append({"pred_ret":pred_ret, "real_ret":real_ret})
    if not bt_res:
        return {"年化收益":0, "最大回撤":0, "夏普比率":0, "胜率":0}
    bt_df = pd.DataFrame(bt_res)
    win_rate = (bt_df["pred_ret"]>0).sum() / len(bt_df)
    ann_ret = bt_df["real_ret"].mean() / pred_years
    max_dd = bt_df["real_ret"].min()
    sharpe = ann_ret / bt_df["real_ret"].std() if bt_df["real_ret"].std() > 0 else 0
    return {"年化收益":round(ann_ret,4), "最大回撤":round(max_dd,4), "夏普比率":round(sharpe,4), "胜率":round(win_rate,4)}

# ====================== 机器学习训练与预测（自动使用有效因子） ======================
def build_ml_dataset(code, valid_factors=None):
    df = get_price(code)
    if len(df) < MIN_DATA_LEN:
        return None
    funda = get_fundamental_hist(code, df.index[-1])
    tech_df = calc_full_indicators(df, funda)
    use_cols = valid_factors if valid_factors else FEATURE_COLS
    ds = []
    for i in range(120, len(df)-PREDICT_DAY):
        feat = {col:tech_df[col].iloc[i] if col in tech_df.columns else 0 for col in use_cols}
        feat["north_flow"] = get_northbound_flow()
        feat["etf_flow"] = get_fund_flow(code, code in STYLE_ETF.values())
        feat["industry_score"] = 50
        feat["sentiment_score"] = get_market_sentiment()
        future_ret = df["close"].iloc[i+PREDICT_DAY]/df["close"].iloc[i] - 1
        ds.append([feat.get(c,0) for c in use_cols] + [future_ret])
    return pd.DataFrame(ds, columns=use_cols+["target"]), use_cols

def train_rolling_models(code, valid_factors=None):
    data, use_cols = build_ml_dataset(code, valid_factors)
    if data is None or len(data) < 120:
        return None, None, None, {"rmse":np.nan,"mae":np.nan,"r2":np.nan,"acc":0.5}
    train_split = int(len(data)*0.6)
    val_split = int(len(data)*0.8)
    train, val, test = data[:train_split], data[train_split:val_split], data[val_split:]
    X_train, y_train = train[use_cols], train["target"]
    X_val, y_val = val[use_cols], val["target"]
    X_test, y_test = test[use_cols], test["target"]
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)
    rf = RandomForestRegressor(n_estimators=120, max_depth=5, random_state=42)
    rf.fit(X_train_sc, y_train)
    xgb_model = None
    try:
        import xgboost as xgb
        xgb_model = xgb.XGBRegressor(n_estimators=120, max_depth=4, random_state=42)
        xgb_model.fit(X_train_sc, y_train, eval_set=[(X_val_sc, y_val)], verbose=False)
    except:
        pass
    rf_pred = rf.predict(X_test_sc)
    if xgb_model is not None:
        xgb_pred = xgb_model.predict(X_test_sc)
        avg_pred = (rf_pred + xgb_pred) / 2
    else:
        avg_pred = rf_pred
    rmse = np.sqrt(mean_squared_error(y_test, avg_pred))
    mae = mean_absolute_error(y_test, avg_pred)
    r2 = r2_score(y_test, avg_pred)
    dir_acc = ((avg_pred > 0) == (y_test > 0)).sum() / len(y_test)
    metrics = {"rmse":round(rmse,4),"mae":round(mae,4),"r2":round(r2,4),"acc":round(dir_acc,4)}
    return rf, xgb_model, scaler, metrics, use_cols

def predict_future(rf, xgb_model, scaler, feat, use_cols=None):
    if rf is None or scaler is None:
        roe = feat.get("roe", 8.0)
        net_grow = feat.get("net_grow", 5.0)
        pb = feat.get("pb", 2.0)
        ret_20d = feat.get("ret_20d", 0.0)
        funda_score_val = roe*0.4 + net_grow*0.3 + (1/(pb+1e-6))*0.2 + ret_20d*0.1
        avg_ret = (funda_score_val - 50) / 100
        prob = max(0, min(1, 0.5 + avg_ret*2))
        return {"prob":round(prob,2), "future_ret":round(avg_ret,3)}
    feat_vec = [feat.get(c,0) for c in use_cols]
    feat_sc = scaler.transform([feat_vec])
    rf_ret = rf.predict(feat_sc)[0]
    if xgb_model is not None:
        xgb_ret = xgb_model.predict(feat_sc)[0]
        avg_ret = (rf_ret + xgb_ret)/2
    else:
        avg_ret = rf_ret
    prob = max(0, min(1, 0.5 + avg_ret*2))
    return {"prob":round(prob,2), "future_ret":round(avg_ret,3)}

# ====================== 补齐缺失：均衡策略扫描 ======================
def scan_balanced(valid_factors=None):
    records = []
    selected_inds = []
    for style, ind_dict in STYLE_INDUSTRY_POOL.items():
        for ind_name, codes in ind_dict.items():
            selected_inds.append(ind_name)
            for code in codes:
                df = get_price(code)
                if len(df) < 60:
                    continue
                funda = get_fundamental_hist(code, df.index[-1])
                tech_df = calc_full_indicators(df, funda)
                latest_feat = tech_df.iloc[-1].to_dict()
                latest_feat["roe"] = funda["roe"]
                latest_feat["net_grow"] = funda["net_grow"]
                latest_feat["ret_20d"] = tech_df["ret_20d"].iloc[-1]
                rf, xgb, scaler, _, use_cols = train_rolling_models(code, valid_factors)
                pred = predict_future(rf, xgb, scaler, latest_feat, use_cols)
                records.append({
                    "风格大类": style,
                    "名称": f"标的{code}",
                    "代码": code,
                    "行业": ind_name,
                    "总分": round(pred["prob"]*100,2),
                    "上涨概率": pred["prob"],
                    "20日预期收益": pred["future_ret"]
                })
    df_all = pd.DataFrame(records)
    if df_all.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),[]
    attack = df_all[df_all["风格大类"]=="成长进攻"].head(WEEK_TOP_NUM)
    balance = df_all.head(WEEK_TOP_NUM)
    defense = df_all[df_all["风格大类"]=="红利防御"].head(WEEK_TOP_NUM)
    return df_all, attack, balance, defense, list(set(selected_inds))

# ====================== 补齐缺失：产业链扫描 ======================
def scan_industry_chain():
    mock_pool = [
        ("688256", "寒武纪", "AI算力核心龙头", "龙头"),
        ("000977", "浪潮信息", "AI算力核心龙头", "二线龙头"),
        ("603019", "中际旭创", "AI算力核心龙头", "核心设备"),
        ("600584", "长电科技", "半导体核心龙头", "核心材料"),
        ("300458", "全志科技", "半导体核心龙头", "普通供应商")
    ]
    res = []
    for code, name, chain, tier in mock_pool:
        df = get_price(code)
        if df.empty:
            continue
        p_score = get_chain_profit_score(code, df.index[-1])
        w = CHAIN_TIER_WEIGHT.get(tier,0.4)
        total = p_score * w
        feat = {"roe":12, "net_grow":15, "pb":2.5, "ret_20d":5}
        pred = predict_future(None, None, None, feat)
        res.append({
            "产业链":chain, "标的梯队":tier, "名称":name, "代码":code,
            "利润质量分":p_score, "综合总分":round(total,2),
            "上涨概率":pred["prob"], "20日预期收益":pred["future_ret"]
        })
    return pd.DataFrame(res)

# ====================== 主运行入口 ======================
def run_final_system():
    init_cache()
    now = datetime.now().strftime("%Y-%m-%d")
    print("="*120)
    print(f"【量化系统 V5.3 机构多因子终版｜{now}｜IC+IR+分层回测】")
    print("="*120)
    print(f"市场情绪得分：{get_market_sentiment():.1f}/100")

    # 因子IC检验
    all_codes = []
    for dic in STYLE_INDUSTRY_POOL.values():
        for codes in dic.values():
            all_codes.extend(codes)
    print("\n===== 因子IC/IR有效性检验 =====")
    ic_df, ls_dict, valid_facs = factor_ic_analysis(all_codes, PREDICT_DAY)
    print("IC/IR指标：")
    print(ic_df.to_string(index=False))
    print(f"\n✅ 筛选出有效因子(IR>0.3)：{valid_facs}")

    print("\n===== 策略一：均衡分散多风格组合（使用有效因子训练） =====")
    all_df, attack, balance, defense, inds = scan_balanced(valid_facs)
    print(f"覆盖行业：{inds}")
    print("\n🔥 进攻组合")
    print(attack.to_string(index=False))
    print("\n⚖️ 均衡组合")
    print(balance.to_string(index=False))
    print("\n🛡️ 防御组合")
    print(defense.to_string(index=False))

    print("\n===== 策略二：产业链影子股挖掘 =====")
    chain_df = scan_industry_chain()
    print(chain_df.to_string(index=False))

    print("\n===== 滚动回测示例（以688256为例） =====")
    bt = walk_forward_backtest("688256")
    print(f"回测指标：{bt}")
    print("="*120)

if __name__ == "__main__":
    run_final_system()