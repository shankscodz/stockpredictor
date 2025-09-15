# ╔══════════════════════════════════════════════════════════════════╗
# ║               AI-BASED INTRADAY STOCK ANALYZER                  ║
# ║        *single-file edition – preserves original logic*         ║
# ╚══════════════════════════════════════════════════════════════════╝

# ---------- Standard & third-party imports ----------
import warnings, requests, time, sys, os
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from typing import Dict, Any, Tuple
warnings.filterwarnings("ignore")

# ---------- Page configuration ----------
st.set_page_config(
    page_title="AI Intraday Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ╔══════════════════════════════════════════════════════════════════╗
# ║                         DATA ACQUISITION                        ║
# ╚══════════════════════════════════════════════════════════════════╝
class DataAcquisition:
    def __init__(self):
        self.alpha_vantage_key = st.secrets.get("ALPHA_VANTAGE_API_KEY", "")

    # -------------------- public helpers --------------------
    @st.cache_data(ttl=300)
    def get_data(_self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        # Try YFinance first
        try:
            data = _self._get_yfinance_data(ticker, period, interval)
            if data is not None and not data.empty:
                return data
        except Exception as e:
            st.warning(f"YFinance error: {e}")

        # Fallback to Alpha Vantage
        try:
            if _self.alpha_vantage_key:
                data = _self._get_alpha_vantage_data(ticker, interval)
                if data is not None and not data.empty:
                    return data
        except Exception as e:
            st.warning(f"AlphaVantage error: {e}")
        
        # Return empty DataFrame if all methods fail
        return pd.DataFrame()


    @st.cache_data(ttl=86400)
    def get_historical_data(_self, ticker: str, period: str = "1y") -> pd.DataFrame:
        try:
            return yf.download(ticker, period=period, progress=False).dropna()
        except Exception:
            return pd.DataFrame()

    # -------------------- private helpers -------------------
    def _get_yfinance_data(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        try:
            data = yf.download(tickers=ticker, period=period, interval=interval,
                              progress=False, threads=True)
            
            # Check if data is None or empty
            if data is None or data.empty:
                return pd.DataFrame()
            
            if isinstance(data.columns, pd.MultiIndex):
                data = data.droplevel(1, axis=1)
            
            data = data.dropna()
            return data[~data.index.duplicated(keep="first")]
            
        except Exception as e:
            st.warning(f"YFinance download error: {e}")
            return pd.DataFrame()
    def _get_alpha_vantage_data(self, ticker: str, interval: str) -> pd.DataFrame:
        av_interval = {'1m': '1min', '5m': '5min'}.get(interval, '5min')
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": ticker.replace(".NS", ""),
            "interval": av_interval,
            "apikey": self.alpha_vantage_key,
            "outputsize": "compact"
        }
        js = requests.get(url, params=params).json()
        key = f"Time Series ({av_interval})"
        if key not in js:
            return pd.DataFrame()

        rows = [{
            "Datetime": pd.to_datetime(ts),
            "Open": float(v["1. open"]),
            "High": float(v["2. high"]),
            "Low": float(v["3. low"]),
            "Close": float(v["4. close"]),
            "Volume": int(v["5. volume"])
        } for ts, v in js[key].items()]

        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).set_index("Datetime").sort_index()
        return df

# ╔══════════════════════════════════════════════════════════════════╗
# ║                      FEATURE ENGINEERING                         ║
# ╚══════════════════════════════════════════════════════════════════╝
class FeatureEngine:
    def __init__(self):
        self.feature_config = {
            "rsi_oversold": 30, "rsi_overbought": 70,
            "stoch_oversold": 20, "stoch_overbought": 80,
            "adx_strong_trend": 25, "macd_signal_threshold": 0
        }

    # -------------------- main pipeline ---------------------
    @st.cache_data(ttl=300)
    def add_all_indicators(_self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        req = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in req):
            raise ValueError("OHLCV data required")

        df = _self._add_trend(df)
        df = _self._add_momentum(df)
        df = _self._add_volume(df)
        df = _self._add_volatility(df)
        df = _self._add_strength(df)
        df = _self._add_pattern(df)
        df = _self._add_custom(df)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method="ffill", inplace=True)
        df.fillna(method="bfill", inplace=True)
        return df

    # -------------------- indicator categories ---------------
    def _add_trend(self, df):
        # Basic moving averages
        for p in [5, 10, 20, 50]:
            df[f"SMA_{p}"] = ta.sma(df.Close, length=p)
            df[f"EMA_{p}"] = ta.ema(df.Close, length=p)
            df[f"WMA_{p}"] = ta.wma(df.Close, length=p)

        # Bollinger Bands with error handling
        for p in [20, 50]:
            try:
                # Use append=True method for reliability
                df.ta.bbands(length=p, std=2, append=True)
                
                # Find the actual column names
                bb_cols = [col for col in df.columns if f"BB" in col and f"_{p}_" in col]
                
                if len(bb_cols) >= 3:
                    # Map to standardized names
                    for col in bb_cols:
                        if "BBL" in col:
                            df[f"BB_Lower_{p}"] = df[col]
                        elif "BBM" in col:
                            df[f"BB_Middle_{p}"] = df[col]
                        elif "BBU" in col:
                            df[f"BB_Upper_{p}"] = df[col]
                    
                    # Calculate BB Width if all columns exist
                    if all(f"BB_{suffix}_{p}" in df.columns for suffix in ["Lower", "Middle", "Upper"]):
                        df[f"BB_Width_{p}"] = (df[f"BB_Upper_{p}"] - df[f"BB_Lower_{p}"]) / df[f"BB_Middle_{p}"]
                else:
                    # Fallback: calculate manually
                    sma = df[f"SMA_{p}"]
                    std = df.Close.rolling(window=p).std()
                    df[f"BB_Lower_{p}"] = sma - (std * 2)
                    df[f"BB_Middle_{p}"] = sma
                    df[f"BB_Upper_{p}"] = sma + (std * 2)
                    df[f"BB_Width_{p}"] = (df[f"BB_Upper_{p}"] - df[f"BB_Lower_{p}"]) / df[f"BB_Middle_{p}"]
                    
            except Exception as e:
                st.warning(f"Bollinger Bands calculation failed for period {p}: {e}")
        
        # VWAP and other indicators
        try:
            df["VWAP"] = ta.vwap(df.High, df.Low, df.Close, df.Volume)
            df["VWAP_Dev"] = (df.Close - df.VWAP) / df.VWAP
        except:
            df["VWAP"] = df.Close  # Fallback
            df["VWAP_Dev"] = 0
        
        # Price position (with error handling)
        if "BB_Lower_20" in df.columns and "BB_Upper_20" in df.columns:
            df["Price_Position_BB20"] = (df.Close - df.BB_Lower_20) / (df.BB_Upper_20 - df.BB_Lower_20)
        else:
            df["Price_Position_BB20"] = 0.5  # Neutral position as fallback
        
        return df

    def _add_momentum(self, df):
        for p in [14, 21]:
            df[f"RSI_{p}"] = ta.rsi(df.Close, length=p)
        macd = ta.macd(df.Close, fast=12, slow=26, signal=9)
        df["MACD"], df["MACD_Signal"], df["MACD_Histogram"] = macd[
            ["MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9"]].values.T
        stoch = ta.stoch(df.High, df.Low, df.Close)
        df["Stoch_K"], df["Stoch_D"] = stoch[["STOCHk_14_3_3", "STOCHd_14_3_3"]].values.T
        df["Williams_R"] = ta.willr(df.High, df.Low, df.Close)
        for p in [5, 10, 20]:
            df[f"ROC_{p}"] = ta.roc(df.Close, length=p)
        df["CCI"] = ta.cci(df.High, df.Low, df.Close)
        return df

    def _add_volume(self, df):
        df["OBV"] = ta.obv(df.Close, df.Volume)
        df["OBV_SMA"] = ta.sma(df.OBV, length=20)
        df["AD"] = ta.ad(df.High, df.Low, df.Close, df.Volume)
        df["CMF"] = ta.cmf(df.High, df.Low, df.Close, df.Volume)
        df["Volume_ROC"] = ta.roc(df.Volume, length=10)
        df["PVT"] = ta.pvt(df.Close, df.Volume)
        df["MFI"] = ta.mfi(df.High, df.Low, df.Close, df.Volume)
        return df

    def _add_volatility(self, df):
        df["ATR"] = ta.atr(df.High, df.Low, df.Close)
        df["ATR_Percent"] = df.ATR / df.Close * 100
        for p in [10, 20]:
            df[f"Volatility_{p}"] = df.Close.rolling(p).std()
            df[f"Volatility_Percent_{p}"] = df[f"Volatility_{p}"] / df.Close * 100
        kc = ta.kc(df.High, df.Low, df.Close)
        df["KC_Lower"], df["KC_Middle"], df["KC_Upper"] = kc[["KCLe_20_2", "KCBe_20_2", "KCUe_20_2"]].values.T
        return df

    def _add_strength(self, df):
        # ADX indicators
        try:
            adx = ta.adx(df.High, df.Low, df.Close)
            if adx is not None and not adx.empty:
                expected_adx_cols = ["ADX_14", "DMP_14", "DMN_14"]
                if all(col in adx.columns for col in expected_adx_cols):
                    df["ADX"], df["DI_Plus"], df["DI_Minus"] = adx[expected_adx_cols].values.T
                else:
                    # Fallback: set default values
                    df["ADX"] = 25
                    df["DI_Plus"] = 25
                    df["DI_Minus"] = 25
        except Exception as e:
            st.warning(f"ADX calculation failed: {e}")
            df["ADX"] = 25
            df["DI_Plus"] = 25
            df["DI_Minus"] = 25

        # PSAR indicator - FIX FOR THE ERROR
        try:
            psar = ta.psar(df.High, df.Low, df.Close)
            if psar is not None:
                if isinstance(psar, pd.DataFrame):
                    # If PSAR returns multiple columns, take the first one or the specific column
                    if "PSAR" in psar.columns:
                        df["PSAR"] = psar["PSAR"]
                    elif "PSARl_0.02_0.2" in psar.columns:
                        df["PSAR"] = psar["PSARl_0.02_0.2"]
                    elif "PSARs_0.02_0.2" in psar.columns:
                        df["PSAR"] = psar["PSARs_0.02_0.2"]
                    else:
                        # Take the first column if we can't identify the right one
                        df["PSAR"] = psar.iloc[:, 0]
                else:
                    # If it's a Series, assign directly
                    df["PSAR"] = psar
            else:
                # Fallback: use closing price
                df["PSAR"] = df.Close
        except Exception as e:
            st.warning(f"PSAR calculation failed: {e}")
            df["PSAR"] = df.Close

        # Aroon indicators
        try:
            aroon = ta.aroon(df.High, df.Low)
            if aroon is not None and not aroon.empty:
                expected_aroon_cols = ["AROONU_14", "AROOND_14"]
                if all(col in aroon.columns for col in expected_aroon_cols):
                    df["Aroon_Up"], df["Aroon_Down"] = aroon[expected_aroon_cols].values.T
                    df["Aroon_Oscillator"] = df.Aroon_Up - df.Aroon_Down
                else:
                    # Fallback
                    df["Aroon_Up"] = 50
                    df["Aroon_Down"] = 50
                    df["Aroon_Oscillator"] = 0
        except Exception as e:
            st.warning(f"Aroon calculation failed: {e}")
            df["Aroon_Up"] = 50
            df["Aroon_Down"] = 50
            df["Aroon_Oscillator"] = 0

        return df


    def _add_pattern(self, df):
        body = abs(df.Close - df.Open)
        rng = df.High - df.Low
        df["Doji"] = ((body / rng) < 0.1).astype(int)

        lower = df.Low - np.minimum(df.Open, df.Close)
        upper = df.High - np.maximum(df.Open, df.Close)
        df["Hammer"] = ((lower > 2 * body) & (upper < 0.3 * lower)).astype(int)

        df["Bullish_Engulfing"] = (
            (df.Close > df.Open) &
            (df.Close.shift(1) < df.Open.shift(1)) &
            (df.Close > df.Open.shift(1)) &
            (df.Open < df.Close.shift(1))
        ).astype(int)
        return df

    def _add_custom(self, df):
        df["Price_Momentum_5"] = df.Close / df.Close.shift(5) - 1
        df["Price_Momentum_10"] = df.Close / df.Close.shift(10) - 1
        df["Gap_Up"] = (df.Open > df.High.shift(1)).astype(int)
        df["Gap_Down"] = (df.Open < df.Low.shift(1)).astype(int)
        df["Range_Expansion"] = (df.ATR > df.ATR.rolling(10).mean()).astype(int)
        df["Range_Compression"] = (df.ATR < df.ATR.rolling(10).mean() * 0.8).astype(int)

        if hasattr(df.index, "hour"):
            df["Hour"] = df.index.hour
            df["Minute"] = df.index.minute
            df["Is_Opening"] = (df.Hour == 9).astype(int)
            df["Is_Closing"] = (df.Hour == 15).astype(int)

        df["Volume_Above_Average"] = (df.Volume > df.Volume.rolling(20).mean()).astype(int)
        df["High_Volume_Price_Change"] = df.Volume_Above_Average * abs(df.Price_Momentum_5)
        return df

# ╔══════════════════════════════════════════════════════════════════╗
# ║                          ML CORE                                 ║
# ╚══════════════════════════════════════════════════════════════════╝
class MLCore:
    def __init__(self):
        self.models, self.scalers, self.feature_cols = {}, {}, []

    def train_and_predict(self, df: pd.DataFrame, horizon: int = 15) -> Dict[str, Any]:
        X, y, feats = self._prepare(df, horizon)
        if len(X) < 100:
            return self._default_results()

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                              random_state=42, stratify=y)
        sc = StandardScaler()
        Xtr_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)

        rf = self._train_rf(Xtr_s, ytr)
        xgbm = self._train_xgb(Xtr_s, ytr)

        scores = {"RandomForest": (rf, accuracy_score(yte, rf.predict(Xte_s))),
                  "XGBoost": (xgbm, accuracy_score(yte, xgbm.predict(Xte_s)))}
        best_name = max(scores, key=lambda k: scores[k][1])
        best_model = scores[best_name][0]

        self.models["best"], self.scalers["best"], self.feature_cols = best_model, sc, feats

        latest_scaled = sc.transform(X.iloc[[-1]])
        pred = best_model.predict(latest_scaled)[0]
        prob = best_model.predict_proba(latest_scaled)[0]

        fi = (list(zip(feats, best_model.feature_importances_))
              if hasattr(best_model, "feature_importances_") else [])

        return {
            "prediction": pred,
            "probabilities": {"bearish": prob[0] if len(prob) > 0 else 0.33,
                              "neutral": prob[1] if len(prob) > 1 else 0.34,
                              "bullish": prob[2] if len(prob) > 2 else 0.33},
            "feature_importance": sorted(fi, key=lambda x: x[1], reverse=True)[:15],
            "model_performance": {k: {"score": v[1]} for k, v in scores.items()},
            "best_model": best_name
        }

    # ------------------ helpers ------------------
    def _prepare(self, df, horizon) -> Tuple[pd.DataFrame, pd.Series, list]:
        df = df.copy()
        df["Future_Return"] = (df.Close.shift(-horizon) - df.Close) / df.Close

        def classify(x):
            if pd.isna(x):
                return np.nan
            if x > 0.002:
                return 2
            if x < -0.002:
                return 0
            return 1

        df["Target"] = df.Future_Return.apply(classify)
        df.dropna(subset=["Target"], inplace=True)

        excl = ["Open", "High", "Low", "Close", "Volume", "Adj Close",
                "Future_Return", "Target"]
        feats = [c for c in df.columns if c not in excl]
        X = df[feats].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = df.Target.astype(int)
        return X, y, feats

    @st.cache_resource
    def _train_rf(_self, X, y):
        m = RandomForestClassifier(n_estimators=100, max_depth=15,
                                   min_samples_split=5, min_samples_leaf=2,
                                   random_state=42, n_jobs=-1)
        m.fit(X, y)
        return m

    @st.cache_resource
    def _train_xgb(_self, X, y):
        m = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=42, eval_metric="mlogloss")
        m.fit(X, y)
        return m

    def _default_results(self):
        return {"prediction": 1,
                "probabilities": {"bearish": 0.33, "neutral": 0.34, "bullish": 0.33},
                "feature_importance": [], "model_performance": {}, "best_model": "NA"}

# ╔══════════════════════════════════════════════════════════════════╗
# ║                       DECISION ENGINE                            ║
# ╚══════════════════════════════════════════════════════════════════╝
class DecisionEngine:
    def __init__(self): self.risk_free_rate = 0.05

    def generate_decisions(self, df, mlr, conf_thr=0.7, risk_f=1.0):
        latest = df.iloc[-1]
        price = latest.Close

        probs = mlr["probabilities"]; max_p = max(probs.values())
        direction = max(probs, key=probs.get)
        score = self._investability(df, mlr, latest)

        action = self._action(direction, max_p, score, conf_thr)
        alloc = self._allocation(score, max_p, risk_f)
        entry = self._entry(df, price, action)
        sl = self._stop(df, price, action, risk_f)
        tp = self._take(price, sl, action)

        rr = abs(tp - price) / abs(price - sl) if price != sl else 0
        timeline = self._timeline(df, mlr)

        return {"action": action, "investability_score": int(score),
                "allocation": alloc, "entry_zone": entry,
                "stop_loss": sl, "take_profit": tp,
                "risk_reward": rr, "confidence": max_p,
                "timeline": timeline}

    # ---------- scoring & rules ----------
    def _investability(self, df, mlr, lt):
        s = min(30, max(mlr["probabilities"].values()) * 50)

        tech = 0
        rsi = lt.get("RSI_14", np.nan)
        if 30 <= rsi <= 70: tech += 10
        elif 20 <= rsi <= 80: tech += 6
        else: tech += 2

        if lt.get("MACD", 0) > lt.get("MACD_Signal", 0): tech += 7
        else: tech += 3

        tech += 8 if lt.get("Volume_Above_Average", 0) == 1 else 4
        adx = lt.get("ADX", 0)
        if adx > 25: tech += 10
        elif adx > 20: tech += 6
        else: tech += 2
        s += min(40, tech)

        atrp = lt.get("ATR_Percent", 0)
        s += 15 if 0.5 <= atrp <= 2 else 10 if 0.3 <= atrp <= 3 else 5

        struct = 0
        if lt.Close > lt.get("VWAP", lt.Close): struct += 5
        bbpos = lt.get("Price_Position_BB20", 0.5)
        if 0.2 <= bbpos <= 0.8: struct += 5
        if lt.get("SMA_5", 0) > lt.get("SMA_20", 0): struct += 5
        s += min(15, struct)
        return min(100, max(0, s))

    def _action(self, dirn, conf, score, conf_thr):
        if score < 30: return "AVOID"
        if conf < conf_thr: return "HOLD"
        if dirn == "bullish" and score >= 60: return "BUY"
        if dirn == "bearish" and score >= 60: return "SELL"
        return "HOLD"

    def _allocation(self, score, conf, risk_f):
        base = score / 100 * 0.2
        alloc = base * min(1.5, conf * 2) * risk_f
        return min(25, max(1, alloc))

    def _entry(self, df, price, act):
        atr = df.ATR.iat[-1] if "ATR" in df else price * 0.02
        f = 0.5
        if act == "BUY":
            return {"lower": price - atr * f, "upper": price + atr * f * 0.5}
        if act == "SELL":
            return {"lower": price - atr * f * 0.5, "upper": price + atr * f}
        return {"lower": price - atr * f, "upper": price + atr * f}

    def _stop(self, df, price, act, risk_f):
        atr = df.ATR.iat[-1] if "ATR" in df else price * 0.02
        dist = atr * 2 * risk_f
        if act == "BUY": return price - dist
        if act == "SELL": return price + dist
        return price - dist

    def _take(self, price, sl, act):
        risk = abs(price - sl)
        if act == "BUY": return price + risk * 2
        if act == "SELL": return price - risk * 2
        return price + risk

    def _timeline(self, df, mlr):
        cur = df.index[-1]
        return {"current_time": str(cur),
                "next_5min": {"prediction": "bullish" if mlr["probabilities"]["bullish"] > 0.4 else "neutral",
                              "confidence": max(mlr["probabilities"].values())},
                "next_15min": {"prediction": "neutral", "confidence": 0.5},
                "key_levels": {"resistance": df.High.rolling(20).max().iat[-1],
                               "support": df.Low.rolling(20).min().iat[-1],
                               "pivot": (df.High.iat[-1] + df.Low.iat[-1] + df.Close.iat[-1]) / 3}}

# ╔══════════════════════════════════════════════════════════════════╗
# ║                        VISUALIZATION                             ║
# ╚══════════════════════════════════════════════════════════════════╝
class Visualization:
    def __init__(self):
        self.colors = {"bullish": "#00ff88", "bearish": "#ff4444",
                       "neutral": "#ffaa00", "primary": "#1f77b4",
                       "secondary": "#ff7f0e"}

    def candlestick(self, df):
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.6, 0.2, 0.2],
                            vertical_spacing=0.05,
                            subplot_titles=("Price & Indicators", "Volume", "RSI"))
        fig.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High,
                                     low=df.Low, close=df.Close, name="Price",
                                     increasing_line_color=self.colors["bullish"],
                                     decreasing_line_color=self.colors["bearish"]), row=1, col=1)
        if "SMA_20" in df:
            fig.add_trace(go.Scatter(x=df.index, y=df.SMA_20, name="SMA 20",
                                     line=dict(color="blue", width=1)), row=1, col=1)
        if "EMA_20" in df:
            fig.add_trace(go.Scatter(x=df.index, y=df.EMA_20, name="EMA 20",
                                     line=dict(color="orange", width=1)), row=1, col=1)
        if {"BB_Upper_20", "BB_Lower_20"} <= set(df.columns):
            fig.add_trace(go.Scatter(x=df.index, y=df.BB_Upper_20, name="BB Upper",
                                     line=dict(color="gray", dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df.BB_Lower_20, name="BB Lower",
                                     line=dict(color="gray", dash="dot")), row=1, col=1)
        if "VWAP" in df:
            fig.add_trace(go.Scatter(x=df.index, y=df.VWAP, name="VWAP",
                                     line=dict(color="purple", width=2)), row=1, col=1)

        bar_colors = [self.colors["bullish"] if c > o else self.colors["bearish"]
                      for c, o in zip(df.Close, df.Open)]
        fig.add_trace(go.Bar(x=df.index, y=df.Volume, marker_color=bar_colors,
                             name="Volume"), row=2, col=1)

        if "RSI_14" in df:
            fig.add_trace(go.Scatter(x=df.index, y=df.RSI_14, name="RSI"),
                          row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        fig.update_layout(height=800, xaxis_rangeslider_visible=False,
                          showlegend=True, hovermode="x unified",
                          title="Intraday Price Analysis")
        return fig

    def prob_gauge(self, probs):
        bp = probs.get("bullish", 0) * 100
        fig = go.Figure(go.Indicator(mode="gauge+number+delta", value=bp,
                   delta={"reference": 50}, title={"text": "Bullish Probability"},
                   gauge={"axis": {"range": [0, 100]},
                          "bar": {"color": self.colors["primary"]},
                          "steps": [{"range": [0, 30], "color": self.colors["bearish"]},
                                    {"range": [30, 70], "color": self.colors["neutral"]},
                                    {"range": [70, 100], "color": self.colors["bullish"]}],
                          "threshold": {"line": {"color": "red", "width": 4},
                                        "value": 50}}))
        fig.update_layout(height=300)
        return fig

    def feature_importance(self, fi):
        if not fi:
            st.write("No feature importance available")
            return
        features, imp = zip(*fi[:10])
        fig = go.Figure(go.Bar(x=imp, y=features, orientation="h",
                               marker_color=self.colors["primary"]))
        fig.update_layout(height=400, title="Top Contributing Factors",
                          xaxis_title="Importance Score", yaxis_title="Features")
        st.plotly_chart(fig, use_container_width=True)

# ╔══════════════════════════════════════════════════════════════════╗
# ║                           MAIN APP                              ║
# ╚══════════════════════════════════════════════════════════════════╝
def analyze_stock(ticker, interval, period, horizon, conf_thr, risk_f):
    data_acq, fe, ml, viz, dec = DataAcquisition(), FeatureEngine(), MLCore(), Visualization(), DecisionEngine()

    pb = st.progress(0); status = st.empty()
    status.write("📥 Fetching data…"); pb.progress(10)
    raw = data_acq.get_data(ticker, period, interval)
    if raw.empty:
        st.error("No data found."); return

    status.write("⚙️ Calculating indicators…"); pb.progress(30)
    df = fe.add_all_indicators(raw)

    status.write("🧠 Training models…"); pb.progress(60)
    mlr = ml.train_and_predict(df, horizon)

    status.write("🎯 Generating decision…"); pb.progress(80)
    decisions = dec.generate_decisions(df, mlr, conf_thr, risk_f)

    status.write("📈 Building charts…"); pb.progress(90)
    pb.progress(100); status.write("✅ Done")

    # ------- Presentation -------
    st.header(f"📊 {ticker}")
    c1, c2, c3, c4, c5 = st.columns(5)
    cur = df.Close.iat[-1]
    change = (cur - df.Open.iat[0]) / df.Open.iat[0] * 100
    c1.metric("Current", f"₹{cur:.2f}", f"{change:+.2f}%")
    c2.metric("High", f"₹{df.High.max():.2f}")
    c3.metric("Low", f"₹{df.Low.min():.2f}")
    c4.metric("Volume", f"{df.Volume.sum()/1_000_000:.1f}M")
    c5.metric("AI Score", f"{decisions['investability_score']}/100",
              delta_color="normal" if decisions["investability_score"] > 50 else "inverse")

    st.subheader("📈 Price Chart")
    st.plotly_chart(viz.candlestick(df), use_container_width=True)

    l, r = st.columns(2)
    with l:
        st.subheader("🤖 AI Prediction")
        st.plotly_chart(viz.prob_gauge(mlr["probabilities"]), use_container_width=True)
        st.subheader("🎯 Key Features")
        viz.feature_importance(mlr["feature_importance"])

    with r:
        st.subheader("💡 Recommendation")
        colors = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡", "AVOID": "⚫"}
        st.markdown(f"### {colors.get(decisions['action'], '⚪')} **{decisions['action']}**")
        st.info(f"""
**Allocation:** {decisions['allocation']:.1f}%  
**Entry:** ₹{decisions['entry_zone']['lower']:.2f} – ₹{decisions['entry_zone']['upper']:.2f}  
**Stop-loss:** ₹{decisions['stop_loss']:.2f}  
**Take-profit:** ₹{decisions['take_profit']:.2f}  
**Risk-Reward:** 1:{decisions['risk_reward']:.1f}
""")

    st.subheader("⏰ Strategy Timeline")
    st.json(decisions["timeline"])

# ---------------- sidebar ----------------
st.title("🤖 AI-based Intraday Stock Analyzer")
st.markdown("*Real-time stock analysis with 50+ indicators and ML predictions*")

with st.sidebar:
    st.header("📊 Configuration")
    tk = st.text_input("Stock Ticker", "RELIANCE.NS", help="e.g. RELIANCE.NS, SBIN.NS")
    iv = st.selectbox("Interval", ["1m", "5m"], index=1)
    pd_opt = st.selectbox("Period", ["1d", "2d", "5d"], index=0)
    horiz = st.slider("Prediction Horizon (minutes)", 5, 30, 15)
    with st.expander("🔧 Advanced"):
        cthr = st.slider("Confidence Threshold", 0.5, 0.9, 0.7)
        rfac = st.slider("Risk Factor", 0.1, 2.0, 1.0)
    run = st.button("🚀 Analyze Stock", type="primary")

if run:
    analyze_stock(tk, iv, pd_opt, horiz, cthr, rfac)
else:
    st.info("Set parameters and press **Analyze Stock**")
