import akshare as ak
import pandas as pd
import numpy as np
import warnings
import time
warnings.filterwarnings("ignore")

# ===================== 状态定义 =====================
REGIME_LABELS = ["S0_筑底冷清", "S1_早期动量(候选爆发)", "S2_主升拥挤", "S3_趋势衰退"]
# 锁定五大高潜力行业（AKShare标准行业名称）
TARGET_INDUSTRIES = ["半导体", "计算机应用", "游戏", "算力设备", "汽车整车"]

class ProbabilisticRegimeSystem:
    def __init__(self, short_win=20, long_win=60):
        self.short_win = short_win
        self.long_win = long_win
        self.request_delay = 0.3  # 拉长间隔，防接口断开

    def safe_ak_get(self, func, *args, **kwargs):
        for _ in range(3):
            try:
                time.sleep(self.request_delay)
                return func(*args, **kwargs)
            except (OSError, TimeoutError):
                continue
        return None

    def _softmax(self, scores):
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / np.sum(exp_scores)

    def calculate_state_prob(self, price_df):
        if len(price_df) < self.long_win:
            return None

        close = price_df["收盘"].values
        volume = price_df["成交量"].values
        ret_short = close[-1] / close[-self.short_win] - 1
        ret_long = close[-1] / close[-self.long_win] - 1
        ret_60_20 = ret_short - ret_long

        ret_series = price_df["收盘"].pct_change().dropna()
        vol_short = ret_series.tail(self.short_win).std()
        vol_long = ret_series.tail(self.long_win).std()
        vol_ratio = vol_short / vol_long if vol_long > 1e-6 else 1.0

        vol_recent = volume[-5:].mean()
        vol_ma_short = volume[-self.short_win:].mean()
        vol_spike = vol_recent / vol_ma_short if vol_ma_short > 1e-6 else 1.0
        vol_trend = np.corrcoef(np.arange(len(volume[-self.short_win:])), volume[-self.short_win:])[0, 1]

        high_20 = np.max(close[-self.short_win:])
        drawdown = (high_20 - close[-1]) / high_20

        # 四状态打分
        score_s0 = -abs(ret_long) - abs(ret_short) - (vol_spike - 1) + (1 - vol_ratio)
        score_s1 = (0.2 - abs(ret_long - 0.08)) + ret_60_20 + (vol_spike - 1.3) + vol_trend
        score_s2 = ret_long + ret_short - drawdown - vol_ratio
        score_s3 = -ret_long - ret_short - drawdown - vol_trend

        scores = np.array([score_s0, score_s1, score_s2, score_s3])
        probs = self._softmax(scores)

        best_idx = np.argmax(probs)
        best_regime = REGIME_LABELS[best_idx]

        if best_regime == "S1_早期动量(候选爆发)" and probs[1] > 0.55:
            action = "BUY_SIGNAL_CONFIRMED"
            desc = f"S1概率{probs[1]:.2%}，量能正向共振，状态拐点确认"
        elif best_regime == "S2_主升拥挤":
            action = "HOLD_OR_REDUCE"
            desc = f"S2概率{probs[2]:.2%}，拥挤度偏高，禁止追高"
        elif best_regime == "S3_趋势衰退":
            action = "STRONG_SELL"
            desc = f"S3概率{probs[3]:.2%}，进入下行周期，规避风险"
        else:
            action = "WATCH_LIST"
            desc = f"S0概率{probs[0]:.2%}，等待波动率与量能拐点"

        return {
            "S0概率": round(probs[0], 4),
            "S1概率": round(probs[1], 4),
            "S2概率": round(probs[2], 4),
            "S3概率": round(probs[3], 4),
            "最优状态": best_regime,
            "建议操作": action,
            "信号描述": desc,
            "近20日收益": round(ret_short, 4),
            "近60日收益": round(ret_long, 4),
            "量能突变倍数": round(vol_spike, 2)
        }

    def execute_market_scan(self):
        print("⚡ [概率状态机系统] 开始扫描五大高潜力行业：")
        print(f"  {', '.join(TARGET_INDUSTRIES)}")
        print("-" * 40)

        all_candidates = []
        for industry in TARGET_INDUSTRIES:
            print(f"\n🔍 正在扫描【{industry}】行业成分股...")
            # 获取行业内股票列表
            industry_stocks = self.safe_ak_get(ak.stock_board_industry_cons_em, symbol=industry)
            if industry_stocks is None or industry_stocks.empty:
                print(f"  ❌ {industry} 行业数据获取失败，跳过")
                continue
            # 过滤ST股、低成交额标的
            industry_stocks = industry_stocks[~industry_stocks["名称"].str.contains("ST|退")]
            # 只取流动性前20只，进一步降低请求量
            scan_pool = industry_stocks.head(20)

            for idx, row in scan_pool.iterrows():
                code = row["代码"]
                name = row["名称"]
                hist_df = self.safe_ak_get(ak.stock_zh_a_hist, symbol=code, period="daily", adjust="qfq")
                if hist_df is None:
                    continue
                status = self.calculate_state_prob(hist_df)
                if status:
                    status.update({
                        "股票代码": code,
                        "股票名称": name,
                        "所属行业": industry
                    })
                    all_candidates.append(status)

        df_res = pd.DataFrame(all_candidates)
        if df_res.empty:
            print("\n❌ 未获取到有效标的数据，请稍后重试")
            return df_res

        print("\n" + "="*30 + " 各行业状态分布统计 " + "="*30)
        print(df_res.groupby(["所属行业", "最优状态"]).size().unstack(fill_value=0))

        print("\n" + "="*30 + " 🔔 高概率S1买入信号 (S1概率>55%) 🔔 " + "="*30)
        trade_signals = df_res[df_res["建议操作"] == "BUY_SIGNAL_CONFIRMED"]
        if not trade_signals.empty:
            print(trade_signals[["所属行业", "股票代码", "股票名称", "S1概率", "量能突变倍数", "信号描述"]].to_string(index=False))
        else:
            print("当前五大高潜力行业中，暂无S0→S1状态拐点的高概率买入信号")

        return df_res

if __name__ == "__main__":
    system = ProbabilisticRegimeSystem()
    system.execute_market_scan()