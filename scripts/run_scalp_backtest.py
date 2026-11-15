"""Scalp Backtest Suite — Janela de 5 a 20 minutos (Timeframe M5).

Audita a estratégia de fluxo institucional no modo SCALP sobre os 30 pares
da Binance Futures usando as velas reais de 5 minutos (data/historical/*_5m):

- Execução walk-forward sem look-ahead (sinal no fechamento [t], entrada na abertura [t+1])
- Fricção real: Taker 0.05% + Slippage 0.02% (ida e volta)
- Duração forçada da operação dentro da janela de 5 a 20 minutos
- Critérios de aprovação para scalp:
    * Total de trades >= 200 (amostra estatística mínima)
    * Win Rate        >= 55%
    * Profit Factor   >= 1.5
    * Expectativa     >= +0.15 R por trade
    * PnL líquido     > 0 R
    * >= 90% das operações dentro da janela de 5-20 minutos
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_massive_backtest import (
    fetch_binance_klines, enrich_features, TOP_30_PAIRS,
)
from backtest.scalp_replay import ScalpReplayBacktester

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ScalpBacktest")

APPROVAL_CRITERIA = {
    "min_trades": 200,
    "min_win_rate_pct": 55.0,
    "min_profit_factor": 1.5,
    "min_expectancy_r": 0.15,
    "min_duration_within_window_pct": 90.0,
}


def run_scalp_backtest(
    symbols: Optional[List[str]] = None,
    min_score: float = 70.0,
    risk_reward: float = 1.5,
    max_hold_minutes: int = 20,
) -> Dict[str, Any]:
    """Executa o backtest de scalp em todos os símbolos selecionados."""
    symbols = symbols or TOP_30_PAIRS
    tester = ScalpReplayBacktester(
        min_score=min_score,
        risk_reward_ratio=risk_reward,
        fee_rate=0.0005,
        slippage_rate=0.0002,
        max_bars_in_trade=max(1, max_hold_minutes // 5),
    )

    by_symbol: Dict[str, Any] = {}
    all_trades: List[Dict[str, Any]] = []

    logger.info(f"Scalp backtest M5 ({max_hold_minutes} min máx) em {len(symbols)} pares | score>={min_score} | R:R {risk_reward}:1")

    for sym in symbols:
        df_m5 = fetch_binance_klines(sym, interval="5m", limit=1000)
        df_m15 = fetch_binance_klines(sym, interval="15m", limit=1000)
        if len(df_m5) < 60 or len(df_m15) < 40:
            continue
        df_m5 = enrich_features(df_m5)
        df_m15 = enrich_features(df_m15)
        res = tester.run(sym, df_m15=df_m15, df_m5=df_m5)
        by_symbol[sym] = res
        all_trades.extend(res.get("trades", []))

    total = len(all_trades)
    wins = [t for t in all_trades if t["is_win"]]
    losses = [t for t in all_trades if not t["is_win"]]
    wr = (len(wins) / total * 100.0) if total else 0.0
    gg = sum(t["pnl_r"] for t in wins)
    gl = abs(sum(t["pnl_r"] for t in losses))
    pf = (gg / gl) if gl > 0 else (99.0 if gg > 0 else 0.0)
    expectancy = (wr / 100.0) * risk_reward - (1 - wr / 100.0) * 1.0
    net_r = sum(t["pnl_r"] for t in all_trades)
    durs = [t["duration_min"] for t in all_trades]
    within = (sum(1 for d in durs if 5 <= d <= max_hold_minutes) / total * 100.0) if total else 0.0
    avg_net_ret = (sum(t["net_return_pct"] for t in all_trades) / total) if total else 0.0

    checks = {
        "amostra >= 200 trades": total >= APPROVAL_CRITERIA["min_trades"],
        "win rate >= 55%": wr >= APPROVAL_CRITERIA["min_win_rate_pct"],
        "profit factor >= 1.5": pf >= APPROVAL_CRITERIA["min_profit_factor"],
        "expectancia >= +0.15R": expectancy >= APPROVAL_CRITERIA["min_expectancy_r"],
        "pnl liquido > 0R": net_r > 0,
        ">= 90% das operacoes entre 5 e 20 min": within >= APPROVAL_CRITERIA["min_duration_within_window_pct"],
    }
    approved = all(checks.values())

    return {
        "mode": "SCALP_M5_5_20_MIN",
        "min_score": min_score,
        "risk_reward_ratio": risk_reward,
        "max_hold_minutes": max_hold_minutes,
        "symbols_tested": len(by_symbol),
        "total_trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "global_win_rate_pct": round(wr, 1),
        "global_profit_factor": round(pf, 2),
        "global_expectancy_r": round(expectancy, 2),
        "global_net_pnl_r": round(net_r, 2),
        "avg_net_return_pct_per_trade": round(avg_net_ret, 3),
        "avg_duration_min": round(sum(durs) / total, 1) if total else 0.0,
        "median_duration_min": sorted(durs)[total // 2] if total else 0.0,
        "duration_within_window_pct": round(within, 1),
        "approval_checks": checks,
        "approved_for_scalp": approved,
        "by_symbol": by_symbol,
    }


if __name__ == "__main__":
    report = run_scalp_backtest()
    print("\n" + "=" * 70)
    print("        RESULTADO DO BACKTEST DE SCALP (M5 — 5 A 20 MINUTOS)")
    print("=" * 70)
    print(f"Pares testados:        {report['symbols_tested']}")
    print(f"Total de trades:       {report['total_trades']}")
    print(f"Win Rate:              {report['global_win_rate_pct']}%")
    print(f"Profit Factor:         {report['global_profit_factor']}")
    print(f"Expectancia:           {report['global_expectancy_r']:+.2f} R/trade")
    print(f"PnL Liquido:           {report['global_net_pnl_r']:+.2f} R")
    print(f"Retorno medio liquido: {report['avg_net_return_pct_per_trade']:+.3f}% por trade")
    print(f"Duracao media:         {report['avg_duration_min']} min | mediana {report['median_duration_min']} min")
    print(f"Dentro da janela 5-20 min: {report['duration_within_window_pct']}%")
    print("-" * 70)
    for k, ok in report["approval_checks"].items():
        print(f"[{'PASS' if ok else 'FAIL'}] {k}")
    print("-" * 70)
    print("VEREDITO:", "APROVADO PARA SCALP ✅" if report["approved_for_scalp"] else "REPROVADO ❌")
    print("=" * 70)
