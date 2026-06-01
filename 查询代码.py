import pandas as pd
import numpy as np
import akshare as ak
import matplotlib.pyplot as plt
import time
import os
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


# ====================== 模块一：数据引擎（本地缓存优先） ======================
class DataEngine:
    @staticmethod
    def fetch_data(code: str, asset_type: str = "stock", start_date: str = "20220101", end_date: str = "20251231",
                   max_retry=3) -> pd.DataFrame:
        # 本地缓存文件名
        cache_file = f"{code}_{asset_type}_{start_date}_{end_date}.csv"

        # 如果缓存存在，直接读取，不联网
        if os.path.exists(cache_file):
            print(f"📂 读取本地缓存数据: {cache_file}")
            df = pd.read_csv(cache_file, parse_dates=["date"])
            return df

        # 缓存不存在，联网下载
        retry_count = 0
        while retry_count < max_retry:
            try:
                if asset_type == "stock":
                    df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date,
                                            adjust="qfq")
                    df = df.rename(columns={
                        "日期": "date", "开盘": "open", "收盘": "close",
                        "最高": "high", "最低": "low", "成交量": "volume"
                    })
                elif asset_type == "etf":
                    df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start_date, end_date=end_date,
                                             adjust="qfq")
                    df = df.rename(columns={
                        "日期": "date", "开盘": "open", "收盘": "close",
                        "最高": "high", "最低": "low", "成交量": "volume"
                    })
                else:
                    raise ValueError("未知资产类型，仅支持 'stock' / 'etf'")

                df['date'] = pd.to_datetime(df['date'])
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']].sort_values('date').reset_index(drop=True)

                # 保存到本地缓存
                df.to_csv(cache_file, index=False)
                print(f"✅ 数据下载成功，已保存至本地: {cache_file}")
                return df

            except Exception as e:
                retry_count += 1
                print(f"⚠️  第{retry_count}次联网失败，等待2秒重试: {e}")
                time.sleep(2)
        print(f"❌ 联网获取失败 [{code}]，重试{max_retry}次结束")
        return pd.DataFrame()

    @staticmethod
    def build_features(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 60:
            return pd.DataFrame()

        df = df.copy()
        close = df['close']
        high = df['high']
        low = df['low']

        df['ma5'] = close.rolling(5).mean()
        df['ma10'] = close.rolling(10).mean()
        df['ma20'] = close.rolling(20).mean()
        df['ma60'] = close.rolling(60).mean()

        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        df['macd_dif'] = exp1 - exp2
        df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = (df['macd_dif'] - df['macd_dea']) * 2

        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()

        df['vol_ma5'] = df['volume'].rolling(5).mean()
        df['vol_ratio'] = df['volume'] / (df['vol_ma5'] + 1e-9)

        df = df.dropna().reset_index(drop=True)
        return df

    @staticmethod
    def generate_labels(df: pd.DataFrame) -> tuple:
        features_list = ['ma5', 'ma10', 'ma20', 'ma60', 'macd_dif', 'macd_dea', 'macd_hist', 'rsi', 'atr', 'vol_ratio']
        X = df[features_list].copy()

        y_1 = (df['close'].shift(-1) > df['close']).astype(int)
        y_5 = (df['close'].shift(-5) > df['close']).astype(int)
        y_20 = (df['close'].shift(-20) > df['close']).astype(int)

        return X, y_1, y_5, y_20


# ====================== 模块二：AI模型引擎 ======================
class MultiPeriodAIEngine:
    def __init__(self):
        self.models = {
            1: self._init_ensemble_models(),
            5: self._init_ensemble_models(),
            20: self._init_ensemble_models()
        }

    def _init_ensemble_models(self):
        return {
            'rf': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
            'xgb': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, eval_metric='logloss',
                                 random_state=42),
            'lgb': LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, verbose=-1)
        }

    def train_and_predict(self, X, y_1, y_5, y_20, latest_features):
        predictions = {}
        labels = {1: y_1, 5: y_5, 20: y_20}
        feat_array = np.array(latest_features).reshape(1, -1)

        for period in [1, 5, 20]:
            y = labels[period]
            valid_idx = y.dropna().index
            X_train = X.loc[valid_idx]
            y_train = y.loc[valid_idx]

            if len(X_train) < 30:
                predictions[period] = 0.5
                continue

            prob_list = []
            for name, model in self.models[period].items():
                model.fit(X_train, y_train)
                prob = model.predict_proba(feat_array)[0][1]
                prob_list.append(prob)
            predictions[period] = float(np.mean(prob_list))
        return predictions

    def backtest_ensemble(self, X, y, test_ratio=0.2):
        split_idx = int(len(X) * (1 - test_ratio))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        models = self._init_ensemble_models()
        pred_probs = []
        for m in models.values():
            m.fit(X_train, y_train)
            pred_probs.append(m.predict_proba(X_test)[:, 1])

        avg_prob = np.mean(pred_probs, axis=0)
        y_pred = (avg_prob > 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        return acc, report

    def walk_forward_backtest(self, X, y, window_size=252, step=20):
        accuracies = []
        total_samples = 0
        correct_samples = 0

        print(f"\n🔄 开始滚动窗口回测 | 训练窗口: {window_size}日 | 滚动步长: {step}日")
        for start in range(0, len(X) - window_size, step):
            end = start + window_size
            X_train = X.iloc[start:end]
            y_train = y.iloc[start:end]
            X_test = X.iloc[end:end + step]
            y_test = y.iloc[end:end + step]

            if len(X_test) == 0:
                break

            models = self._init_ensemble_models()
            pred_probs = []
            for m in models.values():
                m.fit(X_train, y_train)
                pred_probs.append(m.predict_proba(X_test)[:, 1])

            avg_prob = np.mean(pred_probs, axis=0)
            y_pred = (avg_prob > 0.5).astype(int)

            batch_acc = accuracy_score(y_test, y_pred)
            accuracies.append(batch_acc)
            total_samples += len(y_test)
            correct_samples += sum(y_pred == y_test)

            print(f"窗口{start:03d}-{end:03d} | 批次准确率: {batch_acc:.2%}")

        overall_acc = correct_samples / total_samples if total_samples > 0 else 0
        mean_batch_acc = np.mean(accuracies) if accuracies else 0
        return overall_acc, mean_batch_acc


# ====================== 模块三：风控决策 + 可视化 ======================
class RiskDecisionEngine:
    @staticmethod
    def calc_trend_score(prob_1: float, prob_5: float, prob_20: float) -> float:
        score = prob_1 * 0.3 + prob_5 * 0.4 + prob_20 * 0.3
        return round(score * 100, 2)

    @staticmethod
    def get_risk_level(score: float) -> tuple[str, str]:
        if score >= 65:
            return "低风险 🟢", "低吸 / 持有"
        elif 50 <= score < 65:
            return "中性 🟡", "观望 / 轻仓试错"
        else:
            return "高风险 🔴", "回避 / 不建议入场"

    @staticmethod
    def print_dashboard(code: str, asset_type: str, score: float, prob_dict: dict, risk: str, advice: str):
        print("=" * 60)
        print(f"📊 A股/ETF 多周期量化预测看板 | 标的代码: {code} ({asset_type})")
        print("=" * 60)
        print(f"📈 未来1日上涨概率: {prob_dict[1]:.2%}")
        print(f"📈 未来5日上涨概率: {prob_dict[5]:.2%}")
        print(f"📈 未来20日上涨概率: {prob_dict[20]:.2%}")
        print("-" * 60)
        print(f"🏆 综合趋势评分: {score} / 100")
        print(f"⚠️  风险等级: {risk}")
        print(f"💡 操作建议: {advice}")
        print("=" * 60 + "\n")

    @staticmethod
    def print_backtest_report(period: int, acc: float, report: dict):
        print(f"\n📋 周期{period}日 静态回测结果")
        print(f"✅ 集成模型准确率: {acc:.2%}")
        print(f"📌 上涨类别精确率: {report['1']['precision']:.2%}")
        print(f"📌 上涨类别召回率: {report['1']['recall']:.2%}")
        print("-" * 40)

    @staticmethod
    def plot_price_chart(df):
        plt.figure(figsize=(14, 7))
        plt.plot(df['date'], df['close'], label='收盘价', color='#1f77b4', linewidth=1.2)
        plt.plot(df['date'], df['ma5'], label='MA5', color='#ff7f0e', linewidth=1)
        plt.plot(df['date'], df['ma20'], label='MA20', color='#2ca02c', linewidth=1)
        plt.title('标的价格与均线走势')
        plt.xlabel('日期')
        plt.ylabel('价格')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # ========== 配置区 ==========
    TARGET_CODE = "513500"
    ASSET_TYPE = "etf"
    START_DATE = "20220101"
    END_DATE = "20251231"
    TEST_RATIO = 0.2
    WINDOW_SIZE = 252
    STEP = 20
    # ============================

    data_engine = DataEngine()
    ai_engine = MultiPeriodAIEngine()
    risk_engine = RiskDecisionEngine()

    df_raw = data_engine.fetch_data(TARGET_CODE, ASSET_TYPE, START_DATE, END_DATE)
    df_feature = data_engine.build_features(df_raw)
    if df_feature.empty:
        print("❌ 数据不足或获取失败，程序终止")
        exit()

    X, y1, y5, y20 = data_engine.generate_labels(df_feature)
    latest_feat = X.iloc[-1:]

    # 实时预测
    pred_probs = ai_engine.train_and_predict(X, y1, y5, y20, latest_feat)
    trend_score = risk_engine.calc_trend_score(pred_probs[1], pred_probs[5], pred_probs[20])
    risk_level, trade_advice = risk_engine.get_risk_level(trend_score)
    risk_engine.print_dashboard(TARGET_CODE, ASSET_TYPE, trend_score, pred_probs, risk_level, trade_advice)

    # 静态回测
    print("🔍 执行静态划分回测...")
    for period, y_data in zip([1, 5, 20], [y1, y5, y20]):
        valid_y = y_data.dropna()
        valid_X = X.loc[valid_y.index]
        acc, report = ai_engine.backtest_ensemble(valid_X, valid_y, TEST_RATIO)
        risk_engine.print_backtest_report(period, acc, report)

    # 滚动回测
    print("\n" + "=" * 60)
    print("📊 滚动窗口回测汇总")
    print("=" * 60)
    for period, y_data in zip([1, 5, 20], [y1, y5, y20]):
        valid_y = y_data.dropna()
        valid_X = X.loc[valid_y.index]
        overall_acc, mean_batch_acc = ai_engine.walk_forward_backtest(valid_X, valid_y, WINDOW_SIZE, STEP)
        print(f"\n📅 周期{period}日 滚动回测汇总")
        print(f"✅ 整体准确率: {overall_acc:.2%}")
        print(f"✅ 批次平均准确率: {mean_batch_acc:.2%}")

    # 可视化K线
    risk_engine.plot_price_chart(df_feature)