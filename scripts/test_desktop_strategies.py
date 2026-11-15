"""Testador Quantitativo das 10 Estratégias do Desktop Binance.

Converte e executa as 10 estratégias da pasta `strategies/desktop_binance/`
contra a base histórica de 30 pares em CSV, permitindo comparar os resultados
brutos (sem filtro) contra a versão aprimorada com confluências institucionais.

Uso:
    python scripts/test_desktop_strategies.py --benchmark
    python scripts/test_desktop_strategies.py --strategy 04_choch_fibo_618
    python scripts/test_desktop_strategies.py --strategy 08_smart_money_sweep_liquidity --confluence
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from scripts.run_massive_backtest import fetch_binance_klines, enrich_features, TOP_30_PAIRS
from indicators.volume_profile import VolumeProfileAnalyzer
from indicators.confluence import ConfluenceAnalyzer

STRATEGIES_MAP = {
    "01_padrao_123_stormer": "Padrão 1-2-3 Stormer (EMA 100)",
    "02_quatro_emas_alinhadas": "4 EMAs Alinhadas (9, 21, 50, 200)",
    "03_cci_rsi_40_80": "CCI + RSI Momentum (EMA 50)",
    "04_choch_fibo_618": "CHoCH + Fibo 61.8% + Engolfo",
    "05_didi_index_agulhada": "Agulhada Didi Index (3, 8, 20)",
    "06_divergencia_rsi_price": "Divergência Preço x RSI",
    "07_euforia_rejeicao_volume": "Clímax de Volume + Rejeição 55%",
    "08_smart_money_sweep_liquidity": "Smart Money Liquidity Sweep 20b",
    "09_mfi_rsi_extremo_reversao": "MFI + RSI Extremo de Exaustão",
    "10_mola_bollinger_reversao": "Efeito Mola Bandas de Bollinger",
}


def compute_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    money_flow = typical_price * df["volume"]
    pos_flow = pd.Series(0.0, index=df.index)
    neg_flow = pd.Series(0.0, index=df.index)
    
    diff = typical_price.diff()
    pos_flow[diff > 0] = money_flow[diff > 0]
    neg_flow[diff < 0] = money_flow[diff < 0]
    
    pos_mf = pos_flow.rolling(period).sum()
    neg_mf = neg_flow.rolling(period).sum()
    
    mfi = 100.0 - (100.0 / (1.0 + (pos_mf / (neg_mf + 1e-10))))
    return mfi.fillna(50.0)


def compute_signals(strat_key: str, df: pd.DataFrame, use_confluence: bool = False) -> Tuple[pd.Series, pd.Series]:
    """Calcula sinais booleanos de Long e Short para cada estratégia."""
    n = len(df)
    long_sig = pd.Series(False, index=df.index)
    short_sig = pd.Series(False, index=df.index)

    close = df["close"]
    open_p = df["open"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    atr = df["atr"]

    # 01. Padrão 1-2-3 Stormer
    if "01" in strat_key:
        ema100 = close.ewm(span=100, adjust=False).mean()
        c1_low = low.shift(2)
        c2_low = low.shift(1)
        c2_high = high.shift(1)
        fundo123 = (c1_low < low.shift(3)) & (c1_low < c2_low) & (close > c2_high) & (close > ema100)

        c1_high = high.shift(2)
        c2_high_t = high.shift(1)
        c2_low_t = low.shift(1)
        topo123 = (c1_high > high.shift(3)) & (c1_high > c2_high_t) & (close < c2_low_t) & (close < ema100)

        long_sig = fundo123
        short_sig = topo123

    # 02. Quatro EMAs Alinhadas
    elif "02" in strat_key:
        ema9 = close.ewm(span=9, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()

        bull_trend = (ema9 > ema21) & (ema21 > ema50) & (ema50 > ema200)
        bear_trend = (ema9 < ema21) & (ema21 < ema50) & (ema50 < ema200)

        long_sig = bull_trend & (low <= ema21) & (close > ema9)
        short_sig = bear_trend & (high >= ema21) & (close < ema9)

    # 03. CCI + RSI 40-80
    elif "03" in strat_key:
        rsi = df["rsi"]
        tp = (high + low + close) / 3.0
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        cci = (tp - sma_tp) / (0.015 * mad + 1e-10)
        ema50 = close.ewm(span=50, adjust=False).mean()

        rsi_cross_up = (rsi > 40) & (rsi.shift(1) <= 40)
        rsi_cross_dn = (rsi < 60) & (rsi.shift(1) >= 60)

        long_sig = rsi_cross_up & (cci > -100) & (close > ema50)
        short_sig = rsi_cross_dn & (cci < 100) & (close < ema50)

    # 04. CHoCH + Fibo 61.8%
    elif "04" in strat_key:
        hh = high.shift(1).rolling(20).max()
        ll = low.shift(1).rolling(20).min()
        fibo618_long = ll + (hh - ll) * 0.618
        fibo618_short = hh - (hh - ll) * 0.618

        is_engulf_l = (close > open_p) & (close.shift(1) < open_p.shift(1)) & (close >= open_p.shift(1))
        is_engulf_s = (close < open_p) & (close.shift(1) > open_p.shift(1)) & (close <= open_p.shift(1))

        long_sig = (close > hh.shift(5)) & (low <= fibo618_long) & is_engulf_l
        short_sig = (close < ll.shift(5)) & (high >= fibo618_short) & is_engulf_s

    # 05. Didi Index Agulhada
    elif "05" in strat_key:
        sma8 = close.rolling(8).mean()
        curta = close.rolling(3).mean() / (sma8 + 1e-10)
        longa = close.rolling(20).mean() / (sma8 + 1e-10)

        cross_up = (curta > 1.0) & (curta.shift(1) <= 1.0)
        cross_dn = (curta < 1.0) & (curta.shift(1) >= 1.0)

        long_sig = cross_up & (longa < 1.0)
        short_sig = cross_dn & (longa > 1.0)

    # 06. Divergência RSI x Preço
    elif "06" in strat_key:
        rsi = df["rsi"]
        price_lower = low < low.shift(10)
        rsi_higher = rsi > rsi.shift(10)
        price_higher = high > high.shift(10)
        rsi_lower = rsi < rsi.shift(10)

        long_sig = price_lower & rsi_higher & (close > open_p) & (rsi < 35)
        short_sig = price_higher & rsi_lower & (close < open_p) & (rsi > 65)

    # 07. Euforia e Rejeição de Volume
    elif "07" in strat_key:
        vol_ma = volume.rolling(20).mean()
        vol_climax = volume > (vol_ma * 2.2)
        rng = (high - low).replace(0, 1e-6)
        lower_wick = np.where(close > open_p, open_p - low, close - low)
        upper_wick = np.where(close > open_p, high - close, high - open_p)

        rej_fundo = vol_climax & (lower_wick > rng * 0.55) & (close > open_p)
        rej_topo = vol_climax & (upper_wick > rng * 0.55) & (close < open_p)

        long_sig = rej_fundo
        short_sig = rej_topo

    # 08. Smart Money Liquidity Sweep
    elif "08" in strat_key:
        prev_high = high.shift(1).rolling(20).max()
        prev_low = low.shift(1).rolling(20).min()

        sweep_low = (low < prev_low) & (close > prev_low) & (close > open_p)
        sweep_high = (high > prev_high) & (close < prev_high) & (close < open_p)

        long_sig = sweep_low
        short_sig = sweep_high

    # 09. MFI + RSI Extremo de Exaustão
    elif "09" in strat_key:
        mfi = compute_mfi(df, 14)
        rsi = df["rsi"]

        long_sig = (mfi < 20) & (rsi < 25) & (close > open_p)
        short_sig = (mfi > 80) & (rsi > 75) & (close < open_p)

    # 10. Efeito Mola Bollinger
    elif "10" in strat_key:
        basis = close.rolling(20).mean()
        std = close.rolling(20).std()
        upper_bb = basis + (2.0 * std)
        lower_bb = basis - (2.0 * std)

        rng = (high - low).replace(0, 1e-6)
        lower_wick = np.where(close > open_p, open_p - low, close - low)
        upper_wick = np.where(close > open_p, high - close, high - open_p)

        mola_alta = (low < lower_bb) & (close > lower_bb) & (lower_wick > rng * 0.45)
        mola_baixa = (high > upper_bb) & (close < upper_bb) & (upper_wick > rng * 0.45)

        long_sig = mola_alta
        short_sig = mola_baixa

    # Confluence Booster (Volume Profile + Delta/CVD + Fibo)
    if use_confluence:
        cvd = df["cvd"]
        delta = df["delta"]
        # Exige que no sinal Long, o CVD não esteja em queda livre extrema
        cvd_slope_ok = (cvd - cvd.shift(3)) >= -abs(cvd.shift(3) * 0.05)
        long_sig = long_sig & cvd_slope_ok & (df["rvol"] >= 0.8)

        # Exige que no sinal Short, o CVD não esteja em explosão de compra
        cvd_slope_dn_ok = (cvd - cvd.shift(3)) <= abs(cvd.shift(3) * 0.05)
        short_sig = short_sig & cvd_slope_dn_ok & (df["rvol"] >= 0.8)

    return long_sig.fillna(False), short_sig.fillna(False)


def simulate_strategy_trades(
    df: pd.DataFrame,
    long_sig: pd.Series,
    short_sig: pd.Series,
    rr_ratio: float = 2.0,
    fee_rate: float = 0.0005,
    slippage_rate: float = 0.0002,
) -> List[Dict[str, Any]]:
    trades = []
    in_pos = False
    side = None
    entry_p = 0.0
    sl = 0.0
    tp = 0.0
    max_hold = 32  # 8 horas no M15
    hold_bars = 0

    friction = (fee_rate * 2) + (slippage_rate * 2)

    for i in range(25, len(df)):
        c = df["close"].iloc[i]
        h = df["high"].iloc[i]
        l = df["low"].iloc[i]
        atr = df["atr"].iloc[i]

        if in_pos:
            hold_bars += 1
            if side == "LONG":
                if l <= sl:
                    trades.append({"is_win": False, "pnl_r": -1.0 - friction})
                    in_pos = False
                elif h >= tp:
                    trades.append({"is_win": True, "pnl_r": rr_ratio - friction})
                    in_pos = False
                elif hold_bars >= max_hold:
                    pnl_norm = (c - entry_p) / max(entry_p - sl, 1e-6)
                    trades.append({"is_win": pnl_norm > 0, "pnl_r": pnl_norm - friction})
                    in_pos = False

            elif side == "SHORT":
                if h >= sl:
                    trades.append({"is_win": False, "pnl_r": -1.0 - friction})
                    in_pos = False
                elif l <= tp:
                    trades.append({"is_win": True, "pnl_r": rr_ratio - friction})
                    in_pos = False
                elif hold_bars >= max_hold:
                    pnl_norm = (entry_p - c) / max(sl - entry_p, 1e-6)
                    trades.append({"is_win": pnl_norm > 0, "pnl_r": pnl_norm - friction})
                    in_pos = False

        if not in_pos:
            if long_sig.iloc[i]:
                in_pos = True
                side = "LONG"
                entry_p = c
                risk = max(atr * 1.0, entry_p * 0.004)
                sl = entry_p - risk
                tp = entry_p + (risk * rr_ratio)
                hold_bars = 0
            elif short_sig.iloc[i]:
                in_pos = True
                side = "SHORT"
                entry_p = c
                risk = max(atr * 1.0, entry_p * 0.004)
                sl = entry_p + risk
                tp = entry_p - (risk * rr_ratio)
                hold_bars = 0

    return trades


def run_benchmark(strategy_filter: Optional[str] = None, use_confluence: bool = False):
    print("\n" + "=" * 80)
    print(" 🔬 BENCHMARK QUANTITATIVO: ESTRATÉGIAS DESKTOP BINANCE (30 PARES HISTÓRICOS)")
    print(f" Modo de Confluência: {'ATIVADO (CVD + Volume + Exaustão)' if use_confluence else 'DESATIVADO (Baseline Puro)'}")
    print("=" * 80 + "\n")

    targets = [k for k in STRATEGIES_MAP.keys() if strategy_filter in k] if strategy_filter else list(STRATEGIES_MAP.keys())

    summary_rows = []

    for strat_key in targets:
        name = STRATEGIES_MAP[strat_key]
        all_trades = []

        for sym in TOP_30_PAIRS:
            df = fetch_binance_klines(sym, interval="15m", limit=1000)
            if len(df) < 60:
                continue
            df = enrich_features(df)
            l_sig, s_sig = compute_signals(strat_key, df, use_confluence=use_confluence)
            trades = simulate_strategy_trades(df, l_sig, s_sig, rr_ratio=2.0)
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

        status = "✅ APROVADO" if (pf >= 1.30 and net_r > 0) else "⚠️ REVISAR" if net_r > 0 else "❌ REJEITADO"

        summary_rows.append({
            "key": strat_key,
            "name": name,
            "trades": tot,
            "wr": wr,
            "pf": pf,
            "net_r": net_r,
            "exp_r": exp_r,
            "status": status,
        })

        print(f"[{status}] {strat_key[:22]:<22} | {tot:>4} Trades | WR: {wr:>5.1f}% | PF: {pf:>4.2f} | PnL: {net_r:>+7.1f}R | Exp: {exp_r:>+5.2f}R")

    print("\n" + "=" * 80)
    print("                     RESUMO GERAL DOS RESULTADOS")
    print("=" * 80)
    summary_df = pd.DataFrame(summary_rows).sort_values(by="net_r", ascending=False)
    for idx, row in summary_df.iterrows():
        print(f"• {row['name']:<35} -> PF: {row['pf']:<4.2f} | PnL: {row['net_r']:>+7.1f}R | WR: {row['wr']:>5.1f}% ({row['trades']} trades)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testador Quantitativo de Estratégias Desktop Binance")
    parser.add_argument("--strategy", type=str, default=None, help="Filtrar por nome (ex: 04_choch_fibo_618, 08_smart_money)")
    parser.add_argument("--confluence", action="store_true", help="Ativar confluências institucionais de volume e CVD")
    parser.add_argument("--benchmark", action="store_true", help="Executar o benchmark em todas as 10 estratégias")

    args = parser.parse_args()

    strat = args.strategy or (None if args.benchmark else "08_smart_money_sweep_liquidity")
    run_benchmark(strategy_filter=strat, use_confluence=args.confluence)
