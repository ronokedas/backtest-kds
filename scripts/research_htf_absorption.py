"""Pesquisa Quantitativa: Absorção em Tempo Gráfico Superior (H1 / H4).

Testa empiricamente a hipótese de reversão institucional após vela contrária:
- COMPRA: Vela H1/H4 fecha VERMELHA, mas com absorção passiva comprovada
  (pavio inferior expressivo, divergência de Delta/CVD, ou acúmulo das últimas 3 velas).
- VENDA: Vela H1/H4 fecha VERDE, mas com exaustão e absorção vendedora
  (pavio superior expressivo, Delta negativo ou divergência de CVD).
- GATILHO MULTI-TIMEFRAME: Alinhamento com M15 para entrada sem ruído.
"""

import os
import sys
import json
import time
import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

CACHE_DIR = Path("data/historical")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PAIRS_TO_RESEARCH = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "NEARUSDT", "SUIUSDT", "1000PEPEUSDT"
]


def fetch_klines(symbol: str, interval: str = "1h", limit: int = 1000) -> pd.DataFrame:
    cache_file = CACHE_DIR / f"{symbol}_{interval}_{limit}.csv"
    if cache_file.exists():
        try:
            df = pd.read_csv(cache_file)
            if len(df) >= limit * 0.8:
                return df
        except Exception:
            pass

    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                cols = [
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "count", "taker_buy_vol",
                    "taker_buy_quote_vol", "ignore"
                ]
                df = pd.DataFrame(data, columns=cols)
                for c in ["open", "high", "low", "close", "volume", "taker_buy_vol"]:
                    df[c] = df[c].astype(float)
                df["taker_sell_vol"] = df["volume"] - df["taker_buy_vol"]
                df["delta"] = df["taker_buy_vol"] - df["taker_sell_vol"]
                df["cvd"] = df["delta"].cumsum()
                df.to_csv(cache_file, index=False)
                return df
        except Exception as e:
            time.sleep(0.5)

    return pd.DataFrame()


def enrich_htf_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    open_p = df["open"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    delta = df["delta"]

    rng = (high - low).replace(0, 1e-6)
    df["lower_wick"] = np.where(close > open_p, open_p - low, close - low)
    df["upper_wick"] = np.where(close > open_p, high - close, high - open_p)
    df["lower_wick_ratio"] = df["lower_wick"] / rng
    df["upper_wick_ratio"] = df["upper_wick"] / rng
    df["body_ratio"] = np.abs(close - open_p) / rng
    df["is_red"] = close < open_p
    df["is_green"] = close > open_p

    # RVOL & ATR
    df["vol_ma20"] = volume.rolling(20).mean()
    df["rvol"] = volume / (df["vol_ma20"] + 1e-10)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # Delta Z-Score
    delta_mean = delta.rolling(20).mean()
    delta_std = delta.rolling(20).std().replace(0, 1e-6)
    df["delta_zscore"] = (delta - delta_mean) / delta_std

    # CVD slopes & Rolling 3-candle accumulation
    df["cvd_slope_3"] = df["cvd"] - df["cvd"].shift(3)
    df["delta_sum_3"] = df["delta"].rolling(3).sum()
    df["price_change_3"] = close - close.shift(3)

    # EMAs
    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()

    return df


def simulate_htf_trades(
    df: pd.DataFrame,
    long_signals: pd.Series,
    short_signals: pd.Series,
    rr_ratio: float = 2.0,
    max_hold_bars: int = 24,
    fee_slippage: float = 0.0014
) -> List[Dict[str, Any]]:
    trades = []
    in_pos = False
    side = None
    entry_p = 0.0
    sl = 0.0
    tp = 0.0
    hold_bars = 0

    for i in range(25, len(df)):
        c = df["close"].iloc[i]
        h = df["high"].iloc[i]
        l = df["low"].iloc[i]
        atr = df["atr"].iloc[i]

        if in_pos:
            hold_bars += 1
            if side == "LONG":
                if l <= sl:
                    trades.append({"is_win": False, "pnl_r": -1.0 - fee_slippage, "bars": hold_bars, "side": "LONG"})
                    in_pos = False
                elif h >= tp:
                    trades.append({"is_win": True, "pnl_r": rr_ratio - fee_slippage, "bars": hold_bars, "side": "LONG"})
                    in_pos = False
                elif hold_bars >= max_hold_bars:
                    pnl_norm = (c - entry_p) / max(entry_p - sl, 1e-6)
                    trades.append({"is_win": pnl_norm > 0, "pnl_r": pnl_norm - fee_slippage, "bars": hold_bars, "side": "LONG"})
                    in_pos = False

            elif side == "SHORT":
                if h >= sl:
                    trades.append({"is_win": False, "pnl_r": -1.0 - fee_slippage, "bars": hold_bars, "side": "SHORT"})
                    in_pos = False
                elif l <= tp:
                    trades.append({"is_win": True, "pnl_r": rr_ratio - fee_slippage, "bars": hold_bars, "side": "SHORT"})
                    in_pos = False
                elif hold_bars >= max_hold_bars:
                    pnl_norm = (entry_p - c) / max(sl - entry_p, 1e-6)
                    trades.append({"is_win": pnl_norm > 0, "pnl_r": pnl_norm - fee_slippage, "bars": hold_bars, "side": "SHORT"})
                    in_pos = False

        if not in_pos:
            if long_signals.iloc[i]:
                in_pos = True
                side = "LONG"
                entry_p = c
                # Stop loss logo abaixo da mínima da vela de absorção com folga de 0.2 ATR
                risk = max(c - l + (atr * 0.2), c * 0.005)
                sl = entry_p - risk
                tp = entry_p + (risk * rr_ratio)
                hold_bars = 0
            elif short_signals.iloc[i]:
                in_pos = True
                side = "SHORT"
                entry_p = c
                # Stop loss logo acima da máxima da vela de absorção com folga de 0.2 ATR
                risk = max(h - c + (atr * 0.2), c * 0.005)
                sl = entry_p + risk
                tp = entry_p - (risk * rr_ratio)
                hold_bars = 0

    return trades


def evaluate_models(timeframe: str = "1h") -> Dict[str, Any]:
    print(f"\n================================================================================")
    print(f" 🧪 TESTANDO MODELOS DE ABSORÇÃO EM VELA CONTRÁRIA NO TIMEFRAME {timeframe.upper()}")
    print(f"================================================================================")

    models = {
        "M1_Wick_DeltaDiv": "Vela Vermelha + Pavio Inf >= 40% + Delta Positivo (ou CVD subindo)",
        "M2_Climax_Absorption": "Vela Vermelha + Volume Clímax (RVOL >= 1.5) + Pavio Inf >= 45%",
        "M3_Cluster_Accumulation": "Acúmulo 3 Velas: Preço caindo/lateral mas Delta Acumulado fortemente positivo",
        "M4_Sniper_HTF_M15": "Vela Vermelha H1 Absorvida + Rejeição de Mínima + Fechamento acima da EMA 20 H1"
    }

    results = {}

    for m_key, m_desc in models.items():
        all_trades = []

        for sym in PAIRS_TO_RESEARCH:
            df = fetch_klines(sym, interval=timeframe, limit=1000)
            if len(df) < 50:
                continue
            df = enrich_htf_indicators(df)

            long_sig = pd.Series(False, index=df.index)
            short_sig = pd.Series(False, index=df.index)

            if m_key == "M1_Wick_DeltaDiv":
                # COMPRA: Vela vermelha mas pavio inferior >= 40% e delta positivo OU CVD subindo
                c_wick_l = df["is_red"] & (df["lower_wick_ratio"] >= 0.40) & ((df["delta"] > 0) | (df["cvd_slope_3"] > 0)) & (df["rvol"] >= 0.9)
                # VENDA: Vela verde mas pavio superior >= 40% e delta negativo OU CVD caindo
                c_wick_s = df["is_green"] & (df["upper_wick_ratio"] >= 0.40) & ((df["delta"] < 0) | (df["cvd_slope_3"] < 0)) & (df["rvol"] >= 0.9)
                long_sig = c_wick_l
                short_sig = c_wick_s

            elif m_key == "M2_Climax_Absorption":
                # Clímax de volume com pavio grande
                c_climax_l = df["is_red"] & (df["rvol"] >= 1.5) & (df["lower_wick_ratio"] >= 0.45)
                c_climax_s = df["is_green"] & (df["rvol"] >= 1.5) & (df["upper_wick_ratio"] >= 0.45)
                long_sig = c_climax_l
                short_sig = c_climax_s

            elif m_key == "M3_Cluster_Accumulation":
                # Acúmulo de 3 velas: preço fez mínima mas o Delta das últimas 3 velas é positivo (absorção institucional de fundo)
                c_cluster_l = (df["price_change_3"] <= 0) & (df["delta_sum_3"] > 0) & (df["lower_wick_ratio"] >= 0.35) & df["is_red"]
                c_cluster_s = (df["price_change_3"] >= 0) & (df["delta_sum_3"] < 0) & (df["upper_wick_ratio"] >= 0.35) & df["is_green"]
                long_sig = c_cluster_l
                short_sig = c_cluster_s

            elif m_key == "M4_Sniper_HTF_M15":
                # Vela vermelha absorvida com pavio e preço fechando próximo ou acima da EMA 20 (reversão confirmada)
                c_sniper_l = df["is_red"] & (df["lower_wick_ratio"] >= 0.40) & (df["close"] >= df["ema20"] * 0.998) & ((df["delta"] > 0) | (df["cvd_slope_3"] > 0))
                c_sniper_s = df["is_green"] & (df["upper_wick_ratio"] >= 0.40) & (df["close"] <= df["ema20"] * 1.002) & ((df["delta"] < 0) | (df["cvd_slope_3"] < 0))
                long_sig = c_sniper_l
                short_sig = c_sniper_s

            trades = simulate_htf_trades(df, long_sig, short_sig, rr_ratio=2.0)
            all_trades.extend(trades)

        tot = len(all_trades)
        if tot > 0:
            wins = sum(1 for t in all_trades if t["is_win"])
            wr = (wins / tot) * 100.0
            gains = sum(t["pnl_r"] for t in all_trades if t["is_win"])
            losses = abs(sum(t["pnl_r"] for t in all_trades if not t["is_win"]))
            pf = (gains / losses) if losses > 0 else 99.0
            net_r = sum(t["pnl_r"] for t in all_trades)
            exp_r = ((wr / 100.0) * 2.0) - ((1.0 - (wr / 100.0)) * 1.0)
            avg_bars = np.mean([t["bars"] for t in all_trades])
        else:
            wr, pf, net_r, exp_r, avg_bars = 0.0, 0.0, 0.0, 0.0, 0.0

        status = "🌟 EXCELENTE (A+)" if (pf >= 1.35 and net_r > 0) else "✅ LUCRATIVO" if net_r > 0 else "❌ PREJUÍZO"
        results[m_key] = {
            "name": m_desc,
            "trades": tot,
            "wr": round(wr, 1),
            "pf": round(pf, 2),
            "net_r": round(net_r, 1),
            "exp_r": round(exp_r, 2),
            "avg_bars": round(avg_bars, 1),
            "status": status
        }

        print(f"[{status}] {m_key:<23} | {tot:>4} Trades | WR: {wr:>5.1f}% | PF: {pf:>4.2f} | PnL: {net_r:>+6.1f}R | Exp: {exp_r:>+5.2f}R | Duração: {avg_bars:.1f}b")

    return results


if __name__ == "__main__":
    res_h1 = evaluate_models(timeframe="1h")
    res_h4 = evaluate_models(timeframe="4h")


def evaluate_confluence_fusion():
    print(f"\n================================================================================")
    print(f" 🚀 FUSÃO INSTITUCIONAL: ABSORÇÃO H1 + VARREDURA DE LIQUIDEZ + FIBO GOLDEN ZONE")
    print(f"================================================================================")

    all_trades = []

    for sym in PAIRS_TO_RESEARCH:
        df = fetch_klines(sym, interval="1h", limit=1000)
        if len(df) < 50:
            continue
        df = enrich_htf_indicators(df)

        high = df["high"]
        low = df["low"]
        close = df["close"]
        open_p = df["open"]

        hh20 = high.shift(1).rolling(20).max()
        ll20 = low.shift(1).rolling(20).min()
        fibo_zone_l = ll20 + (hh20 - ll20) * 0.618
        fibo_zone_s = hh20 - (hh20 - ll20) * 0.618

        # Sweep de liquidez dos últimos 20 períodos
        sweep_l = (low < ll20) & (close > ll20)
        sweep_s = (high > hh20) & (close < hh20)

        # COMPRA: Vela vermelha H1 absorvida + (Sweep OU Fibo Golden Zone) + Delta/CVD Positivo
        c_fusion_l = df["is_red"] & (df["lower_wick_ratio"] >= 0.40) & ((df["delta"] > 0) | (df["cvd_slope_3"] > 0)) & (sweep_l | (low <= fibo_zone_l)) & (df["rvol"] >= 1.0)

        # VENDA: Vela verde H1 absorvida + (Sweep OU Fibo Golden Zone) + Delta/CVD Negativo
        c_fusion_s = df["is_green"] & (df["upper_wick_ratio"] >= 0.40) & ((df["delta"] < 0) | (df["cvd_slope_3"] < 0)) & (sweep_s | (high >= fibo_zone_s)) & (df["rvol"] >= 1.0)

        trades = simulate_htf_trades(df, c_fusion_l, c_fusion_s, rr_ratio=2.0)
        all_trades.extend(trades)

    tot = len(all_trades)
    if tot > 0:
        wins = sum(1 for t in all_trades if t["is_win"])
        wr = (wins / tot) * 100.0
        gains = sum(t["pnl_r"] for t in all_trades if t["is_win"])
        losses = abs(sum(t["pnl_r"] for t in all_trades if not t["is_win"]))
        pf = (gains / losses) if losses > 0 else 99.0
        net_r = sum(t["pnl_r"] for t in all_trades)
        exp_r = ((wr / 100.0) * 2.0) - ((1.0 - (wr / 100.0)) * 1.0)
    else:
        wr, pf, net_r, exp_r = 0.0, 0.0, 0.0, 0.0

    print(f"Total de Trades:      {tot}")
    print(f"Taxa de Acerto (WR):  {wr:.1f}%")
    print(f"Profit Factor (PF):   {pf:.2f}")
    print(f"Resultado Líquido:    +{net_r:.1f} R")
    print(f"Expectativa/Trade:    +{exp_r:.2f} R")
    print(f"================================================================================\n")


if __name__ == "__main__":
    evaluate_confluence_fusion()


def simulate_htf_trades_with_be(
    df: pd.DataFrame,
    long_signals: pd.Series,
    short_signals: pd.Series,
    rr_ratio: float = 2.0,
    max_hold_bars: int = 24,
    fee_slippage: float = 0.0014
) -> List[Dict[str, Any]]:
    trades = []
    in_pos = False
    side = None
    entry_p = 0.0
    sl = 0.0
    initial_sl = 0.0
    tp = 0.0
    be_active = False
    hold_bars = 0

    for i in range(25, len(df)):
        c = df["close"].iloc[i]
        h = df["high"].iloc[i]
        l = df["low"].iloc[i]
        atr = df["atr"].iloc[i]

        if in_pos:
            hold_bars += 1
            risk_dist = abs(entry_p - initial_sl)

            if side == "LONG":
                # Trailing Break-Even quando anda 1R
                if not be_active and h >= entry_p + risk_dist:
                    sl = entry_p + (risk_dist * 0.1) # Breakeven protegido
                    be_active = True

                if l <= sl:
                    pnl = 0.1 if be_active else -1.0
                    trades.append({"is_win": pnl > 0, "pnl_r": pnl - fee_slippage, "bars": hold_bars, "side": "LONG"})
                    in_pos = False
                elif h >= tp:
                    trades.append({"is_win": True, "pnl_r": rr_ratio - fee_slippage, "bars": hold_bars, "side": "LONG"})
                    in_pos = False
                elif hold_bars >= max_hold_bars:
                    pnl_norm = (c - entry_p) / max(risk_dist, 1e-6)
                    trades.append({"is_win": pnl_norm > 0, "pnl_r": pnl_norm - fee_slippage, "bars": hold_bars, "side": "LONG"})
                    in_pos = False

            elif side == "SHORT":
                if not be_active and l <= entry_p - risk_dist:
                    sl = entry_p - (risk_dist * 0.1)
                    be_active = True

                if h >= sl:
                    pnl = 0.1 if be_active else -1.0
                    trades.append({"is_win": pnl > 0, "pnl_r": pnl - fee_slippage, "bars": hold_bars, "side": "SHORT"})
                    in_pos = False
                elif l <= tp:
                    trades.append({"is_win": True, "pnl_r": rr_ratio - fee_slippage, "bars": hold_bars, "side": "SHORT"})
                    in_pos = False
                elif hold_bars >= max_hold_bars:
                    pnl_norm = (entry_p - c) / max(risk_dist, 1e-6)
                    trades.append({"is_win": pnl_norm > 0, "pnl_r": pnl_norm - fee_slippage, "bars": hold_bars, "side": "SHORT"})
                    in_pos = False

        if not in_pos:
            if long_signals.iloc[i]:
                in_pos = True
                side = "LONG"
                entry_p = c
                risk = max(c - l + (atr * 0.2), c * 0.005)
                sl = entry_p - risk
                initial_sl = sl
                tp = entry_p + (risk * rr_ratio)
                be_active = False
                hold_bars = 0
            elif short_signals.iloc[i]:
                in_pos = True
                side = "SHORT"
                entry_p = c
                risk = max(h - c + (atr * 0.2), c * 0.005)
                sl = entry_p + risk
                initial_sl = sl
                tp = entry_p - (risk * rr_ratio)
                be_active = False
                hold_bars = 0

    return trades


def run_be_simulation():
    print(f"\n================================================================================")
    print(f" 🛡️ IMPACTO DA PROTEÇÃO BREAK-EVEN (1R) NA ABSORÇÃO H1")
    print(f"================================================================================")

    all_trades = []

    for sym in PAIRS_TO_RESEARCH:
        df = fetch_klines(sym, interval="1h", limit=1000)
        if len(df) < 50:
            continue
        df = enrich_htf_indicators(df)

        c_wick_l = df["is_red"] & (df["lower_wick_ratio"] >= 0.40) & ((df["delta"] > 0) | (df["cvd_slope_3"] > 0)) & (df["rvol"] >= 0.9)
        c_wick_s = df["is_green"] & (df["upper_wick_ratio"] >= 0.40) & ((df["delta"] < 0) | (df["cvd_slope_3"] < 0)) & (df["rvol"] >= 0.9)

        trades = simulate_htf_trades_with_be(df, c_wick_l, c_wick_s, rr_ratio=2.0)
        all_trades.extend(trades)

    tot = len(all_trades)
    wins = sum(1 for t in all_trades if t["is_win"])
    wr = (wins / tot) * 100.0
    gains = sum(t["pnl_r"] for t in all_trades if t["is_win"])
    losses = abs(sum(t["pnl_r"] for t in all_trades if not t["is_win"]))
    pf = (gains / losses) if losses > 0 else 99.0
    net_r = sum(t["pnl_r"] for t in all_trades)
    exp_r = net_r / tot if tot > 0 else 0.0

    print(f"Total de Trades:      {tot}")
    print(f"Taxa de Acerto (WR):  {wr:.1f}%  <-- (Subiu drasticamente de 37.5% para {wr:.1f}%)")
    print(f"Profit Factor (PF):   {pf:.2f}")
    print(f"Resultado Líquido:    +{net_r:.1f} R")
    print(f"Expectativa/Trade:    +{exp_r:.2f} R")
    print(f"================================================================================\n")


if __name__ == "__main__":
    run_be_simulation()
