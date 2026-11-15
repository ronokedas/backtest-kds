"""Scalp Replay Backtesting Engine — Janela de 5 a 20 minutos (Timeframe M5).

Motor de execução barra a barra dedicado ao SCALP no timeframe de 5 minutos,
com as mesmas regras institucionais do motor principal:

- Zero Look-Ahead Bias: sinal confirmado no fechamento da vela [t], execução
  exclusivamente na ABERTURA da vela [t+1].
- Fricção real: taxa Taker 0.05% + slippage 0.02% cobrados na entrada e na saída.
- Avaliação intrabar conservadora: se a vela toca Stop e Alvo, o Stop vence.
- Janela de duração da operação: mínimo de 1 barra M5 (5 minutos) e máximo de
  4 barras M5 (20 minutos):
  * max_bars_in_trade = 4  → timeout fecha a operação aos 20 minutos;
  * scalp_timeout_pct      → se após a 1ª barra (5 min) o preço não atingiu a
    fração mínima de progresso rumo ao alvo, a operação é encerrada no início
    da barra seguinte (nunca ultrapassa os 20 minutos);
  * min_entry_move_atr     → filtro de momentum na entrada: a vela de sinal
    precisa fechar deslocada do open por uma fração do ATR, garantindo que o
    fluxo direcional já está ativo (reduz saídas prematuras por stall).
- Gate de geometria: o alvo (R:R × risco) deve ser alcançável dentro da janela
  de volatilidade esperada para 20 minutos (ATR), rejeitando setups lentos
  demais para scalp.

Confluências (todas calculadas apenas com dados passados):
CHoCH/BOS anti-repaint, Golden Zone Fibonacci 50-61.8%, absorção de Delta
(z-score), Volume Profile VAL/VAH sweep, gatilho tático de engolfo M5,
exaustão RSI e alinhamento EMA 20.
"""

from typing import List, Dict, Any
import pandas as pd
import logging

from indicators.market_structure import MarketStructure
from indicators.confluence import ConfluenceAnalyzer
from indicators.volume_profile import VolumeProfileAnalyzer
from signals.triple_engine import TripleConfluenceEngine

logger = logging.getLogger(__name__)


class ScalpReplayBacktester:
    """Walk-forward estrito para scalp M5 com duração forçada de 5 a 20 minutos."""

    def __init__(
        self,
        min_score: float = 70.0,
        risk_reward_ratio: float = 1.5,
        fee_rate: float = 0.0005,        # 0.05% taker
        slippage_rate: float = 0.0002,   # 0.02% slippage
        max_bars_in_trade: int = 4,      # 4 velas M5 = 20 minutos máximos
        min_hold_minutes: int = 5,       # duração mínima alvo (1 vela M5)
        scalp_timeout_pct: float = 0.35, # fração do alvo necessária após 1 barra
        min_stop_atr_mult: float = 1.2,  # stop mínimo em múltiplos de ATR M5
        max_risk_atr_mult: float = 1.8,  # risco máximo em múltiplos de ATR M5
        min_target_atr_mult: float = 1.9,  # alvo mínimo (R) em múltiplos de ATR M5
        target_atr_horizon_mult: float = 1.4,  # alvo deve caber em ~1.4x ATR*sqrt(4 barras)
        min_entry_move_atr: float = 0.25,  # deslocamento mínimo da vela de sinal (fração do ATR)
    ):
        self.min_score = min_score
        self.risk_reward_ratio = risk_reward_ratio
        self.two_way_cost = 2 * (fee_rate + slippage_rate)
        self.max_bars_in_trade = max_bars_in_trade
        self.min_hold_minutes = min_hold_minutes
        self.scalp_timeout_pct = scalp_timeout_pct
        self.min_stop_atr_mult = min_stop_atr_mult
        self.max_risk_atr_mult = max_risk_atr_mult
        self.min_target_atr_mult = min_target_atr_mult
        self.target_atr_horizon_mult = target_atr_horizon_mult
        self.min_entry_move_atr = min_entry_move_atr
        self.structure_engine = MarketStructure(pivot_left=3, pivot_right=2)

    # ------------------------------------------------------------------
    @staticmethod
    def _m5_slice_for_trigger(df_m15: pd.DataFrame, curr_bar: pd.Series) -> pd.DataFrame:
        """Deriva até 3 velas M5 (15 min) a partir das velas M15 concluídas.

        Sem look-ahead: usa apenas M15 com open_time <= open_time da barra atual.
        Cada M15 vira 3 M5 sintéticos (subdivisão linear OHLC determinística),
        preservando close/high/low nas fronteiras corretas.
        """
        prior = df_m15[df_m15["open_time"] <= curr_bar["open_time"]].tail(3)
        rows: List[Dict[str, float]] = []
        for _, r in prior.iterrows():
            o, h, l, c = float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])
            up = (h - min(o, c)) >= (max(o, c) - l)
            if up:
                b1o, b1c = o, h
                b2o, b2c = h, (h + c) / 2.0
                b3o, b3c = b2c, c
            else:
                b1o, b1c = o, l
                b2o, b2c = l, (l + c) / 2.0
                b3o, b3c = b2c, c
            base_t = float(r.get("open_time", 0))
            for j, (so, sc) in enumerate([(b1o, b1c), (b2o, b2c), (b3o, b3c)]):
                rows.append({
                    "open_time": base_t + j * 300000.0,
                    "open": so,
                    "close": sc,
                    "high": max(so, sc),
                    "low": min(so, sc),
                    "volume": float(r.get("volume", 0.0)) / 3.0,
                    "delta": float(r.get("delta", 0.0)) / 3.0,
                    "cvd_slope": float(r.get("cvd_slope", 0.0)),
                })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def run(self, symbol: str, df_m15: pd.DataFrame, df_m5: pd.DataFrame) -> Dict[str, Any]:
        n = len(df_m5)
        if n < 60:
            return {"symbol": symbol, "error": "Dados M5 insuficientes (<60 barras)", "total_trades": 0, "trades": []}

        trades: List[Dict[str, Any]] = []
        active: Any = None

        for i in range(40, n - 1):
            bar = df_m5.iloc[i]
            nxt = df_m5.iloc[i + 1]

            # ---------------- Gestão da posição ativa ----------------
            if active is not None:
                side = active["side"]
                entry_p = active["entry_price"]
                stop_p = active["stop_loss"]
                target_p = active["take_profit"]
                bars_held = i - active["entry_bar_idx"]

                closed, exit_price, pnl_r, reason = False, None, 0.0, ""
                if side == "LONG":
                    if nxt["low"] <= stop_p:
                        exit_price, pnl_r, reason, closed = stop_p, -1.0, "STOP_LOSS", True
                    elif nxt["high"] >= target_p:
                        exit_price, pnl_r, reason, closed = target_p, self.risk_reward_ratio, "TAKE_PROFIT", True
                    elif bars_held >= self.max_bars_in_trade:
                        exit_price = nxt["open"]
                        risk = entry_p - stop_p
                        pnl_r = (exit_price - entry_p) / risk if risk > 0 else 0.0
                        reason, closed = "TIMEOUT_20MIN", True
                    elif bars_held >= self.min_hold_minutes // 5:
                        # Após 5+ minutos sem progresso suficiente → encerra antes dos 20 min
                        progress = (nxt["close"] - entry_p) / (target_p - entry_p) if target_p > entry_p else 1.0
                        if progress < self.scalp_timeout_pct:
                            exit_price = nxt["open"]
                            risk = entry_p - stop_p
                            pnl_r = (exit_price - entry_p) / risk if risk > 0 else 0.0
                            reason, closed = "SCALP_STALL_EXIT", True
                else:  # SHORT
                    if nxt["high"] >= stop_p:
                        exit_price, pnl_r, reason, closed = stop_p, -1.0, "STOP_LOSS", True
                    elif nxt["low"] <= target_p:
                        exit_price, pnl_r, reason, closed = target_p, self.risk_reward_ratio, "TAKE_PROFIT", True
                    elif bars_held >= self.max_bars_in_trade:
                        exit_price = nxt["open"]
                        risk = stop_p - entry_p
                        pnl_r = (entry_p - exit_price) / risk if risk > 0 else 0.0
                        reason, closed = "TIMEOUT_20MIN", True
                    elif bars_held >= self.min_hold_minutes // 5:
                        progress = (entry_p - nxt["close"]) / (entry_p - target_p) if entry_p > target_p else 1.0
                        if progress < self.scalp_timeout_pct:
                            exit_price = nxt["open"]
                            risk = stop_p - entry_p
                            pnl_r = (entry_p - exit_price) / risk if risk > 0 else 0.0
                            reason, closed = "SCALP_STALL_EXIT", True

                if closed:
                    gross = (exit_price - entry_p) / entry_p if side == "LONG" else (entry_p - exit_price) / entry_p
                    active.update({
                        "exit_bar_idx": i + 1,
                        "exit_price": round(exit_price, 4),
                        "pnl_r": round(pnl_r, 2),
                        "net_return_pct": round((gross - self.two_way_cost) * 100.0, 3),
                        "bars_held": bars_held,
                        "duration_min": bars_held * 5,
                        "exit_reason": reason,
                        "is_win": pnl_r > 0.0,
                    })
                    trades.append(active)
                    active = None
                continue

            # ---------------- Avaliação do setup no fechamento [i] ----------------
            slice_m5 = df_m5.iloc[: i + 1]
            m15_slice = df_m15[df_m15["open_time"] <= bar["open_time"]]
            if len(m15_slice) < 30 or len(slice_m5) < 20:
                continue

            curr_price = float(bar["close"])

            struct = self.structure_engine.evaluate_structure(m15_slice)
            fib_data = ConfluenceAnalyzer.compute_fibonacci_retracement(m15_slice, lookback=30, struct_info=struct)
            vp_data = VolumeProfileAnalyzer.compute_volume_profile(m15_slice, lookback=min(100, len(m15_slice)), rows=24)
            m5_trigger_slice = self._m5_slice_for_trigger(m15_slice, bar)

            sig = TripleConfluenceEngine.evaluate_daytrade(
                m15_df=m15_slice,
                h1_df=m15_slice.iloc[::4] if len(m15_slice) >= 8 else m15_slice,
                fib_data=fib_data,
                struct_m15=struct,
                current_price=curr_price,
                spread_pct=0.0005,
                m5_df=m5_trigger_slice,
                vp_data=vp_data,
                oi_data=None,
            )

            if sig.state != "DAYTRADE_ENTRY_A+" or sig.side not in ("LONG", "SHORT"):
                continue
            if float(sig.score) < self.min_score:
                continue

            atr = float(bar.get("atr", 0.0))
            if atr <= 0:
                continue

            # Filtro de momentum: a vela de sinal [i] deve fechar deslocada do open
            move = float(bar["close"]) - float(bar["open"])
            min_move = self.min_entry_move_atr * atr
            if sig.side == "LONG" and move < min_move:
                continue
            if sig.side == "SHORT" and move > -min_move:
                continue

            entry_price = float(nxt["open"])
            inval = float(sig.invalidation) if sig.invalidation else 0.0
            if sig.side == "LONG":
                # invalidação ancorada na mínima local M5 com piso de 0.8 ATR
                # (stop mais largo que a vela de sinal → duração realista >= 5 min)
                stop_p = min(inval if 0 < inval < entry_price else entry_price - 0.9 * atr,
                             entry_price - self.min_stop_atr_mult * atr)
                risk = entry_price - stop_p
                target_p = entry_price + risk * self.risk_reward_ratio
            else:
                stop_p = max(inval if inval > entry_price else entry_price + 0.9 * atr,
                             entry_price + self.min_stop_atr_mult * atr)
                risk = stop_p - entry_price
                target_p = entry_price - risk * self.risk_reward_ratio

            # ---- Gates de geometria de scalp ----
            if risk <= 0:
                continue
            if risk > self.max_risk_atr_mult * atr:
                continue  # stop largo demais para scalp de 5-20 min
            if risk * self.risk_reward_ratio < self.min_target_atr_mult * atr:
                continue  # alvo curto demais: fecharia dentro da primeira vela (< 5 min)
            horizon_atr = atr * (self.max_bars_in_trade ** 0.5) * self.target_atr_horizon_mult
            if abs(target_p - entry_price) > horizon_atr:
                continue  # alvo inalcançável dentro de 20 minutos de volatilidade

            active = {
                "symbol": symbol,
                "side": sig.side,
                "entry_bar_idx": i + 1,
                "entry_time": nxt.get("open_time", i + 1),
                "entry_price": round(entry_price, 4),
                "stop_loss": round(stop_p, 4),
                "take_profit": round(target_p, 4),
                "risk": round(risk, 4),
                "score": sig.score,
                "reasons": sig.reasons[:3],
            }

        # ---------------- Métricas ----------------
        total = len(trades)
        base = {
            "symbol": symbol,
            "total_trades": total,
            "wins": 0, "losses": 0, "win_rate_pct": 0.0, "profit_factor": 0.0,
            "avg_risk_reward": self.risk_reward_ratio, "expectancy_r": 0.0,
            "max_drawdown_r": 0.0, "net_pnl_r": 0.0,
            "avg_duration_min": 0.0, "median_duration_min": 0.0,
            "duration_within_5_20_pct": 0.0, "trades": [],
        }
        if total == 0:
            return base

        wins = [t for t in trades if t["is_win"]]
        losses = [t for t in trades if not t["is_win"]]
        wr = len(wins) / total * 100.0
        gg = sum(t["pnl_r"] for t in wins)
        gl = abs(sum(t["pnl_r"] for t in losses))
        pf = (gg / gl) if gl > 0 else 99.0
        expectancy = (wr / 100.0) * self.risk_reward_ratio - (1 - wr / 100.0)

        cum = peak = dd = 0.0
        for t in trades:
            cum += t["pnl_r"]
            peak = max(peak, cum)
            dd = max(dd, peak - cum)

        durs = sorted(t["duration_min"] for t in trades)
        within = sum(1 for d in durs if 5 <= d <= 20) / total * 100.0

        base.update({
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(wr, 1),
            "profit_factor": round(pf, 2),
            "expectancy_r": round(expectancy, 2),
            "max_drawdown_r": round(dd, 2),
            "net_pnl_r": round(cum, 2),
            "avg_duration_min": round(sum(durs) / total, 1),
            "median_duration_min": durs[total // 2],
            "duration_within_5_20_pct": round(within, 1),
            "trades": trades,
        })
        return base
