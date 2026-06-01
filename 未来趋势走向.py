import numpy as np
import pandas as pd
import yfinance as yf

# =========================
# 1. 获取数据
# =========================
def get_data(symbol="SPY", period="1y"):
    df = yf.download(symbol, period=period)
    df = df[['Close']].copy()
    df['ret'] = df['Close'].pct_change()
    df['vol'] = df['ret'].rolling(10).std()
    df = df.dropna()
    return df

# =========================
# 2. 初始化 HMM 参数（手写版，修复维度问题）
# =========================
class SimpleHMM:
    def __init__(self):
        # 3个状态
        self.states = ["S0_low_vol", "S1_trend", "S2_down"]

        # 初始概率
        self.pi = np.array([0.4, 0.4, 0.2])

        # 状态转移矩阵（经验初始化）
        self.A = np.array([
            [0.6, 0.3, 0.1],
            [0.2, 0.6, 0.2],
            [0.2, 0.3, 0.5]
        ])

    # 高斯概率密度
    def gaussian(self, x, mean, std):
        return (1 / (np.sqrt(2*np.pi)*std)) * np.exp(-0.5*((x-mean)/std)**2)

    # 每个状态的观测模型（修复：保证返回一维3元素数组）
    def emission_prob(self, ret, vol):
        # S0：低波动
        p0 = self.gaussian(ret, 0.0002, 0.005) * self.gaussian(vol, 0.01, 0.005)
        # S1：趋势
        p1 = self.gaussian(ret, 0.001, 0.01) * self.gaussian(vol, 0.015, 0.007)
        # S2：下跌
        p2 = self.gaussian(ret, -0.001, 0.012) * self.gaussian(vol, 0.02, 0.01)
        # 强制转为一维数组，解决广播报错
        return np.array([p0, p1, p2]).ravel()

    # 前向推断（核心）
    def predict(self, df):
        n = len(df)
        probs = np.zeros((n, 3))

        # 初始化
        obs = self.emission_prob(df.iloc[0]['ret'], df.iloc[0]['vol'])
        probs[0] = self.pi * obs
        probs[0] /= probs[0].sum()

        # 递推
        for t in range(1, n):
            obs = self.emission_prob(df.iloc[t]['ret'], df.iloc[t]['vol'])
            for j in range(3):
                probs[t, j] = obs[j] * np.sum(probs[t-1] * self.A[:, j])
            probs[t] /= probs[t].sum()
        return probs

# =========================
# 3. 信号解释器（交易逻辑）
# =========================
def signal_engine(probs):
    last = probs[-1]
    S0, S1, S2 = last
    # 核心判断
    if S1 > 0.5:
        regime = "TREND (可参与上涨)"
    elif S2 > 0.5:
        regime = "DOWNTREND (风险区)"
    else:
        regime = "LOW VOL (等待突破)"
    # 上涨潜力评分
    upside_score = S1 - S2
    return {
        "S0": float(S0),
        "S1": float(S1),
        "S2": float(S2),
        "regime": regime,
        "upside_score": float(upside_score)
    }

# =========================
# 4. 主函数
# =========================
if __name__ == "__main__":
    df = get_data("SPY", "1y")
    model = SimpleHMM()
    probs = model.predict(df)
    result = signal_engine(probs)
    print("\n=== 当前市场状态 ===")
    print(result)