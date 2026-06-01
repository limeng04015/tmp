import time
import pandas as pd
import akshare as ak
import yfinance as yf

# 你的自选池（A股+美股 + 新增华夏机器人ETF）
stock_pool = {
    # A股个股
    "600584": "长电科技",
    "300502": "新易盛",
    "300620": "光库科技",
    "300059": "东方财富",
    # A股ETF
    "513100": "国泰纳指ETF",
    "159819": "人工智能ETF",
    "512480": "芯片ETF",
    "159805": "通信ETF",
    "562500": "华夏机器人ETF",  # 新增
    # 美股
    "AAPL": "苹果",
    "DXYZ": "Destiny Tech"
}

result_list = []
for code, name in stock_pool.items():
    try:
        # A股/ETF用akshare
        if code.isdigit():
            df = ak.fund_etf_hist_em(
                symbol=code,
                period="daily",
                start_date="20260520",
                end_date="20260525",
                adjust="hfq"
            )
            latest = df.iloc[-1]
            price = latest["收盘"]
        # 美股用yfinance
        else:
            df = yf.download(code, period="5d", auto_adjust=True, progress=False)
            latest = df.iloc[-1]
            price = round(latest["Close"], 2)

        # 交易规则（你1000元计划，新增562500）
        if code == "513100":
            trade = "✅可进"
            pos = "400元"
            tp = "10%"
            sl = "4%"
        elif code == "159819":
            trade = "✅可进"
            pos = "250元"
            tp = "12%"
            sl = "5%"
        elif code == "512480":
            trade = "✅可进"
            pos = "200元"
            tp = "12%"
            sl = "5%"
        elif code == "600584":
            trade = "✅可进"
            pos = "100元"
            tp = "15%"
            sl = "7%"
        elif code == "562500":
            trade = "✅可进"   # 按主线ETF给可进
            pos = "200元"     # 你总资金1000，这里给200
            tp = "12%"
            sl = "5%"
        else:
            trade = "❌观望"
            pos = "0元"
            tp = "-"
            sl = "-"

        result_list.append([
            name, code, trade, pos, tp, sl, f"现价：{price}"
        ])
    except Exception as e:
        result_list.append([name, code, "❌数据异常", "-", "-", "-", "-"])

# 输出表格
df_out = pd.DataFrame(
    result_list,
    columns=["股票名称", "代码", "能不能进", "仓位", "止盈", "止损", "实时现价"]
)
print(df_out.to_string(index=False))

# 底部：数据更新时间（精确到秒）
update_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(f"\n===== 数据实时更新时间：{update_time} =====")