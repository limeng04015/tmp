import yfinance as yf

# 全部可运行标的映射
stock_map = {
    "QQQ": "513100｜纳斯达克100ETF｜A股T+0｜长期/日内双模式",
    "VOO": "513500｜标普500ETF｜A股T+0｜长期/日内双模式",
    "QQQM": "纳指100低费率版｜美股T+1｜长期隔夜",
    "SPY": "标普500流动性ETF｜美股T+1｜长期隔夜",
    "AAPL": "苹果｜美股T+1｜长期隔夜",
    "MSFT": "微软｜美股T+1｜长期隔夜",
    "NVDA": "英伟达｜美股T+1｜长期隔夜",
    "JNJ": "强生｜美股T+1｜防御长期",
    "SCHD": "红利ETF｜美股T+1｜稳健长期",
    "XLK": "科技板块ETF｜美股T+1｜弹性长期"
}
stock_list = list(stock_map.keys())

print("="*85)
print("          全部可运行美股/跨境ETF｜每日量化+交易计划检测")
print("="*85)
print("📌 土耳其运行时间：每日04:00后（美股收盘/A股开盘前）")
print("📌 A股跨境ETF：T+0可日内；美股全部T+1，禁止日内交易，只隔夜")
print("📌 通用铁律：止损5%，长期止盈15%+，日内止盈3%-5%")
print("="*85)

for symbol in stock_list:
    name = stock_map[symbol]
    try:
        df = yf.download(symbol, start="2025-01-01", auto_adjust=True, progress=False)
        if len(df) < 20:
            print(f"\n【{name}｜{symbol}】数据不足，暂不操作")
            continue

        # 指标计算
        df["ma5"] = df["Close"].rolling(5).mean()
        df["ma10"] = df["Close"].rolling(10).mean()
        df["ma20"] = df["Close"].rolling(20).mean()
        df["vol5"] = df["Volume"].rolling(5).mean()

        t = df.iloc[-1]
        y = df.iloc[-2]
        ma5 = t["ma5"].item()
        ma10 = t["ma10"].item()
        ma20 = t["ma20"].item()
        close = t["Close"].item()
        vol_t = t["Volume"].item()
        vol_y = y["Volume"].item()
        vol5 = t["vol5"].item()

        # 条件
        均线多头 = (ma5 > ma10) and (ma10 > ma20)
        站上20日线 = close > ma20
        放量 = (vol_t > vol_y) and (vol_t > vol5 * 1.2)
        长期可买 = 均线多头 and 站上20日线 and 放量
        日内可做 = 站上20日线 and 放量

        # 输出
        print(f"\n【{name}｜{symbol}】")
        print(f"  5/10/20均线多头：{均线多头}")
        print(f"  价格站上20日线：{站上20日线}")
        print(f"  成交量放量信号：{放量}")
        print(f"  ⭐ 长期隔夜持仓：{'✅ 买入加仓' if 长期可买 else '❌ 观望'}")
        if "T+0" in name:
            print(f"  ⭐ A股日内T+0：{'✅ 可短线操作' if 日内可做 else '❌ 不做日内'}")

    except Exception as e:
        print(f"\n【{name}｜{symbol}】异常：{str(e)}")

print("\n" + "="*85)
print("💡 严格执行：只买信号✅标的，止损止盈不情绪化，美股禁止日内交易")
print("="*85)