import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")


# =========================
# 🔥 单资产分析器
# =========================
class AssetAnalyzer:

    def __init__(self, symbol):
        self.symbol = symbol
        self.df = None

    # -------------------------
    # 1. 数据获取
    # -------------------------
    def load(self):
        df = yf.download(self.symbol, period="5y", progress=False)
        if df is None or len(df) < 100:
            return False
        df = df.rename(columns=str.lower)
        self.df = df
        return True

    # -------------------------
    # 2. 指标系统
    # -------------------------
    def indicators(self):
        df = self.df

        df["ma5"] = df["close"].rolling(5).mean()
        df["ma10"] = df["close"].rolling(10).mean()
        df["ma20"] = df["close"].rolling(20).mean()

        # RSI
        delta = df["close"].diff()
        gain = delta.mask(delta < 0, 0)
        loss = -delta.mask(delta > 0, 0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / (avg_loss + 1e-6)
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()

        df["dif"] = ema12 - ema26
        df["dea"] = df["dif"].ewm(span=9, adjust=False).mean()
        df["macd"] = df["dif"] - df["dea"]

        self.df = df.dropna()

    # -------------------------
    # 3. 风险模型（强制返回标量）
    # -------------------------
    def risk(self):
        df = self.df
        ret = df["close"].pct_change().dropna()
        drawdown = df["close"] / df["close"].cummax() - 1
        max_dd = drawdown.min().item()
        vol = ret.std().item() * np.sqrt(252)
        sharpe = (ret.mean().item() * 252 - 0.03) / (vol + 1e-6)
        return max_dd, vol, sharpe

    # -------------------------
    # 4. AI预测
    # -------------------------
    def ml(self):
        df = self.df.copy()
        df["target"] = df["close"].shift(-5) > df["close"]
        df = df.dropna()

        features = ["ma5", "ma10", "ma20", "rsi", "dif", "dea", "macd"]
        X = df[features]
        y = df["target"].astype(int)

        if len(df) < 120:
            return 50

        split = int(len(df) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        xgb = XGBClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        xgb.fit(X_train, y_train)

        latest = X.iloc[-1].values.reshape(1, -1)
        prob = (rf.predict_proba(latest)[0][1] + xgb.predict_proba(latest)[0][1]) / 2
        return prob * 100

    # -------------------------
    # 5. 综合评分（全标量计算）
    # -------------------------
    def score(self):
        self.indicators()
        last = self.df.iloc[-1]
        ma5 = last["ma5"].item()
        ma10 = last["ma10"].item()
        trend_score = 1 if ma5 > ma10 else 0

        dd, vol, sharpe = self.risk()
        ml_score = self.ml()

        final_score = (
            ml_score * 0.55 +
            max(0, (1 + sharpe) * 20) * 0.25 +
            (100 + dd * 100) * 0.2
        )
        return final_score


# =========================
# 🔥 多标的扫描器
# =========================
class MarketScanner:
    def __init__(self, symbols):
        self.symbols = symbols
        self.results = []

    def run(self):
        for s in self.symbols:
            print(f"\n分析: {s}")
            bot = AssetAnalyzer(s)
            if not bot.load():
                print("数据失败")
                continue
            score = bot.score()
            self.results.append((s, score))
            print("Score:", round(score, 2))

        self.results.sort(key=lambda x: x[1], reverse=True)
        print("\n======================")
        print("🔥 最终排名")
        print("======================")
        for r in self.results:
            print(r[0], "→", round(r[1], 2))


# =========================
# 🚀 运行入口（自由修改列表）
# =========================
if __name__ == "__main__":
    symbols = [
        "010371.OF",
        "002692.OF",
        "NVDA",
        "MSFT",
        "SPY",
        "QQQ",
        "BABA",
        "AMD"
    ]
    scanner = MarketScanner(symbols)
    scanner.run()