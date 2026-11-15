"""Triple Confluence Engine for Binance Futures USDⓈ-M.

Features 3 independent institutional execution engines:
1. Long-Term Accumulation Engine (D1 + H4 + H1 + Macro Discounts)
2. Day Trade Reversal Engine (M15 + H1 + CHoCH + Fibonacci Golden Zone 50-61.8%)
3. Scalp Pre-Explosion & Expansion Engine (M1 + M5 + M15 + Volatility Squeeze + OBI)

Strict Anti-Noise Principle:
No single indicator triggers an entry. Confluence across multiple independent
data dimensions (order flow, structure, derivative sentiment, volatility) is mandatory.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from indicators.confluence import ConfluenceAnalyzer


@dataclass
class EngineSignal:
    """Standardized signal output for each operational timeframe engine."""
    engine_name: str
    state: str
    score: float
    side: str  # "LONG", "SHORT", or "NEUTRAL"
    price: float
    invalidation: Optional[float]
    target_1: Optional[float]
    target_2: Optional[float]
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class TripleConfluenceEngine:
    """Evaluates the 3 independent operational setups."""

    # Thresholds
    THRESHOLDS = {
        "long_term_watch": 55.0,
        "long_term_confirmed": 70.0,
        "daytrade_watch": 55.0,
        "daytrade_entry": 70.0,
        "scalp_watch": 58.0,
        "scalp_entry": 72.0,
    }

    # =========================================================================
    # 1. LONG-TERM ACCUMULATION ENGINE (D1 + H4 + H1)
    # =========================================================================
    @classmethod
    def evaluate_long_term(
        cls,
        daily_df: pd.DataFrame,
        h4_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        current_price: float,
        current_oi: float = 0.0,
        funding_rate: Optional[float] = None,
    ) -> EngineSignal:
        """Evaluates macro discount and institutional accumulation (D1 + H4 + H1)."""
        if len(daily_df) < 14 or len(h4_df) < 14 or len(h1_df) < 14:
            return EngineSignal(
                engine_name="LONG_TERM",
                state="INSUFFICIENT_DATA",
                score=0.0,
                side="NEUTRAL",
                price=current_price,
                invalidation=None,
                target_1=None,
                target_2=None,
                reasons=["Dados históricos insuficientes em D1/H4/H1"],
            )

        d = daily_df.iloc[-1]
        h = h4_df.iloc[-1]
        score = 0.0
        reasons = []

        # 1. RSI Deprimido D1 (Desconto Macro)
        rsi_d1 = float(d.get("rsi", 50.0))
        if rsi_d1 < 35.0:
            score += 15.0
            reasons.append(f"RSI Diário deprimido ({rsi_d1:.1f})")
            if rsi_d1 < 25.0:
                score += 5.0
                reasons.append("RSI Diário em sobrevenda extrema (< 25)")

        # 2. RSI Deprimido H4
        rsi_h4 = float(h.get("rsi", 50.0))
        if rsi_h4 < 35.0:
            score += 15.0
            reasons.append(f"RSI H4 deprimido ({rsi_h4:.1f})")
            if rsi_h4 < 20.0:
                score += 5.0
                reasons.append("RSI H4 em sobrevenda extrema (< 20)")

        # 3. Divergência Bullish de CVD em H4
        cvd_div_h4 = ConfluenceAnalyzer.detect_cvd_divergence(h4_df, lookback=10)
        if cvd_div_h4.get("bullish_divergence"):
            score += 20.0
            reasons.append("Divergência Bullish de CVD no H4 (absorção de venda)")

        # 4. H1 começando a recuperar momentum
        if len(h1_df) >= 4:
            rsi_h1_now = float(h1_df["rsi"].iloc[-1])
            rsi_h1_prev = float(h1_df["rsi"].iloc[-3])
            if rsi_h1_now > rsi_h1_prev:
                score += 10.0
                reasons.append(f"RSI H1 recuperando ({rsi_h1_prev:.1f} → {rsi_h1_now:.1f})")

            cvd_slope_now = float(h1_df.get("cvd_slope", pd.Series([0])).iloc[-1])
            cvd_slope_prev = float(h1_df.get("cvd_slope", pd.Series([0])).iloc[-3])
            if cvd_slope_now > cvd_slope_prev:
                score += 10.0
                reasons.append("CVD H1 inclinando para cima (pressão vendedora arrefecendo)")

        # 5. Vela H4 fechando na metade superior (rejeição de mínimas)
        close_loc_h4 = float(h.get("close_location", 0.5))
        if close_loc_h4 >= 0.55:
            score += 10.0
            reasons.append(f"Candle H4 fechou no terço superior (loc {close_loc_h4*100:.0f}%)")

        # 6. Desconto em relação à EMA 200 diária
        ema_200_d1 = float(d.get("ema_200", 0.0))
        if ema_200_d1 > 0 and current_price < ema_200_d1:
            discount_pct = ((ema_200_d1 - current_price) / ema_200_d1) * 100.0
            if discount_pct >= 5.0:
                score += 5.0
                reasons.append(f"Preço com {discount_pct:.1f}% de desconto da EMA 200 D1")

        # 7. Funding Rate Negativo (combustível para Short Squeeze)
        if funding_rate is not None and funding_rate < 0:
            score += 5.0
            reasons.append(f"Funding rate negativo ({funding_rate*100:.4f}% — mercado pesado de shorts)")

        score = min(100.0, score)

        if score >= cls.THRESHOLDS["long_term_confirmed"]:
            state = "ACCUMULATION_CONFIRMED"
        elif score >= cls.THRESHOLDS["long_term_watch"]:
            state = "ACCUMULATION_WATCH"
        else:
            state = "NO_TRADE"

        # Targets técnicos macro
        recent_lo = float(daily_df["low"].tail(30).min())
        invalidation = recent_lo * 0.985  # 1.5% abaixo da mínima
        risk = max(current_price * 0.02, current_price - invalidation)
        target_1 = current_price + (risk * 2.0)
        target_2 = current_price + (risk * 4.0)

        return EngineSignal(
            engine_name="LONG_TERM",
            state=state,
            score=round(score, 1),
            side="LONG" if score >= cls.THRESHOLDS["long_term_watch"] else "NEUTRAL",
            price=current_price,
            invalidation=round(invalidation, 2),
            target_1=round(target_1, 2),
            target_2=round(target_2, 2),
            reasons=reasons,
            metrics={
                "rsi_d1": round(rsi_d1, 1),
                "rsi_h4": round(rsi_h4, 1),
                "close_loc_h4": round(close_loc_h4, 2),
                "funding_rate": funding_rate,
            },
        )

    # =========================================================================
    # 2. DAY TRADE REVERSAL ENGINE (M15 + H1 + Fibonacci Golden Zone 50-61.8%)
    # =========================================================================
    @classmethod
    def evaluate_daytrade(
        cls,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        fib_data: Dict[str, Any],
        struct_m15: Dict[str, Any],
        current_price: float,
        spread_pct: float = 0.002,
        m5_df: Optional[pd.DataFrame] = None,
        vp_data: Optional[Dict[str, Any]] = None,
        oi_data: Optional[Dict[str, Any]] = None,
    ) -> EngineSignal:
        """Evaluates intraday A+ reversal confluences (M15 + H1 + Fib 50-61.8% + CHoCH/BOS Retest + M5 Trigger).
        
        Institutional Dual-Phase Architecture:
        Phase 1: Bottom/Top Absorption Climax + CHoCH / BOS Breakout.
        Phase 2: Golden Zone Retest (50% - 61.8%) with Volume Dry-Up & Retest CVD Absorption Divergence.
        Phase 3: Volume Profile (POC / VAH / VAL Sweeps) & Open Interest Traps (Short/Long Trap).
        Phase 4: M5 Tactical Sniper Trigger (Engulfing candle closing above prior high).
        """
        if len(m15_df) < 15 or len(h1_df) < 10:
            return EngineSignal(
                engine_name="DAY_TRADE",
                state="INSUFFICIENT_DATA",
                score=0.0,
                side="NEUTRAL",
                price=current_price,
                invalidation=None,
                target_1=None,
                target_2=None,
                reasons=["Dados históricos insuficientes em M15/H1"],
            )

        curr = m15_df.iloc[-1]
        rvol = float(curr.get("rvol", 1.0))
        delta_z = float(curr.get("delta_zscore", 0.0))
        close_loc = float(curr.get("close_location", 0.5))
        lower_wick = float(curr.get("lower_wick_ratio", 0.0))
        upper_wick = float(curr.get("upper_wick_ratio", 0.0))
        rsi_m15 = float(curr.get("rsi", 50.0))
        prev_rsi_m15 = float(m15_df["rsi"].iloc[-2]) if len(m15_df) > 1 else rsi_m15
        ema_20 = float(curr.get("ema_20", 0.0))

        in_gz = fib_data.get("in_golden_zone", False)
        dist_gz = fib_data.get("distance_to_golden_zone_pct", 99.0)
        is_retest = struct_m15.get("is_choch_retest", False) or struct_m15.get("is_bos_retest", False) or in_gz or dist_gz <= 0.6

        # Institutional Retest CVD Divergence & M5 Tactical Trigger Analysis
        retest_cvd = ConfluenceAnalyzer.detect_retest_cvd_divergence(m15_df, struct_m15)
        m5_trig = ConfluenceAnalyzer.detect_m5_tactical_trigger(m5_df) if m5_df is not None and len(m5_df) >= 3 else {}

        # --- EVALUATE LONG SETUP ---
        score_long = 0.0
        reasons_long = []

        # 1. Structural CHoCH / BOS / Liquidity Sweep (Base)
        if struct_m15.get("bos_bullish"):
            bars_ago = struct_m15.get("bos_bars_ago", 0)
            score_long += 25.0
            b_txt = f"{bars_ago}b atrás" if bars_ago and bars_ago > 0 else "recente"
            reasons_long.append(f"Estrutura M15 com BULLISH BOS confirmado ({b_txt} — continuação de alta)")
        elif struct_m15.get("choch_bullish"):
            bars_ago = struct_m15.get("choch_bars_ago", 0)
            score_long += 25.0
            b_txt = f"{bars_ago}b atrás" if bars_ago and bars_ago > 0 else "recente"
            reasons_long.append(f"Estrutura M15 com BULLISH CHoCH confirmado ({b_txt})")
        elif struct_m15.get("liquidity_sweep_bullish"):
            score_long += 15.0
            reasons_long.append("Liquidity Sweep institucional de mínimas M15")

        # 2. Fibonacci Retracement (Golden Zone 50% - 61.8%)
        if in_gz:
            score_long += 25.0
            reasons_long.append(f"Preço na Golden Zone de Fibonacci (${fib_data.get('fib_618')} - ${fib_data.get('fib_50')})")
        elif dist_gz <= 0.6:
            score_long += 15.0
            reasons_long.append(f"Preço testando borda da zona de Fibonacci 50%/61.8% ({dist_gz:.2f}% de dist)")

        # 3. Microstructure Volume (Volume Dry-Up on Retest vs Volume Climax on Breakout)
        if is_retest or in_gz:
            if rvol <= 0.9:
                score_long += 15.0
                reasons_long.append(f"Volume Dry-Up no recuo (RVOL {rvol:.2f}x — exaustão completa da pressão vendedora)")
            elif rvol >= 1.3:
                score_long += 12.0
                reasons_long.append(f"Absorção ativa com volume no reteste (RVOL {rvol:.2f}x)")
            else:
                score_long += 8.0
        else:
            if rvol >= 1.8:
                score_long += 15.0
                reasons_long.append(f"Choque de volume institucional no M15 (RVOL {rvol:.2f}x)")
            elif rvol >= 1.2:
                score_long += 8.0
                reasons_long.append(f"Volume acima da média no M15 (RVOL {rvol:.2f}x)")

        # 4. Retest CVD Absorption Divergence (Higher Lows on yellow CVD line during pullback)
        if retest_cvd.get("bullish_absorption_divergence"):
            score_long += 20.0
            reasons_long.append("Divergência Bullish de Absorção no CVD (preço corrigindo mas linha amarela sustentando fundos mais altos)")

        # 5. Delta & Order Flow Dynamics
        if is_retest or in_gz:
            if delta_z <= -1.5 and close_loc >= 0.45:
                score_long += 18.0
                reasons_long.append(f"Agressão vendedora absorvida por Bids na Golden Zone (Z {delta_z:.2f})")
            elif abs(delta_z) < 1.0 or delta_z > 0:
                score_long += 14.0
                reasons_long.append(f"Delta estabilizado no teste (Z {delta_z:.2f} — ausência de despejo a mercado)")
            elif delta_z <= -1.0 and close_loc >= 0.40:
                score_long += 10.0
                reasons_long.append("Venda agressiva contida no suporte")
        else:
            if delta_z <= -1.5 and close_loc >= 0.50:
                score_long += 18.0
                reasons_long.append(f"Forte agressão vendedora absorvida (Z {delta_z:.2f}, loc {close_loc*100:.0f}%)")
            elif delta_z <= -1.0 and close_loc >= 0.40:
                score_long += 10.0
                reasons_long.append("Venda agressiva contida no fundo")

        # 6. Price Action Rejection / Candlestick Anatomy
        if lower_wick >= 0.25:
            score_long += 15.0
            reasons_long.append(f"Pavio de rejeição inferior ({lower_wick*100:.0f}%) no suporte institucional")
        elif close_loc >= 0.50:
            score_long += 10.0
            reasons_long.append(f"Vela sustentando na metade superior (loc {close_loc*100:.0f}%)")

        # 7. M5 Tactical Trigger (Engulfing / Reversal)
        if m5_trig.get("bullish_engulfing"):
            score_long += 15.0
            reasons_long.append("Gatilho tático M5 acionado: vela de engolfo verde rompendo máxima anterior")
        elif m5_trig.get("m5_aligned_long"):
            score_long += 8.0
            reasons_long.append("M5 alinhado a favor com Delta e momentum positivos")

        # 8. Regular CVD Divergence / Slope
        cvd_div = ConfluenceAnalyzer.detect_cvd_divergence(m15_df, lookback=10)
        if cvd_div.get("bullish_divergence"):
            score_long += 12.0
            reasons_long.append("Divergência Regular Bullish de CVD no M15")
        elif curr.get("cvd_slope", 0.0) > 0 or curr.get("cvd_change_3", 0.0) >= 0:
            score_long += 7.0
            reasons_long.append("CVD estabilizado com inclinação positiva")

        # 9. Context H1 & EMA 20 Alignment & False-Candle Absorption Check
        htf_red_absorbed = False
        htf_green_absorbed = False
        if len(h1_df) >= 3:
            rsi_h1_now = float(h1_df["rsi"].iloc[-1])
            rsi_h1_prev = float(h1_df["rsi"].iloc[-2])
            if rsi_h1_now >= rsi_h1_prev:
                score_long += 7.0
                reasons_long.append("Alinhamento favorável de contexto H1")

            # Avaliação de Absorção Institucional em Vela H1 Contrária
            last_h1 = h1_df.iloc[-1]
            rng_h1 = max(float(last_h1["high"] - last_h1["low"]), 1e-6)
            c_h1 = float(last_h1["close"])
            o_h1 = float(last_h1["open"])
            l_h1 = float(last_h1["low"])
            h_h1 = float(last_h1["high"])
            lower_wick_h1 = (o_h1 - l_h1) if c_h1 < o_h1 else (c_h1 - l_h1)
            lower_wick_ratio = lower_wick_h1 / rng_h1

            delta_h1 = float(last_h1.get("delta", 0.0))
            cvd_diff_h1 = float(last_h1.get("cvd", 0.0)) - float(h1_df["cvd"].iloc[-2]) if "cvd" in h1_df else 0.0
            delta_sum_3_h1 = float(h1_df["delta"].iloc[-3:].sum()) if "delta" in h1_df else 0.0
            price_change_3_h1 = c_h1 - float(h1_df["close"].iloc[-3])

            # COMPRA: Vela H1 fechou VERMELHA, mas com absorção passiva expressiva
            if c_h1 < o_h1:
                if (lower_wick_ratio >= 0.38 and (delta_h1 > 0 or cvd_diff_h1 > 0)) or (delta_sum_3_h1 > 0 and price_change_3_h1 <= 0 and lower_wick_ratio >= 0.30):
                    htf_red_absorbed = True
                    score_long += 15.0
                    reasons_long.append(f"Vela H1 Vermelha com Absorção Passiva: Pavio Inferior {lower_wick_ratio*100:.0f}% e Delta/CVD divergente (Armadilha de Venda)")

        if ema_20 > 0 and current_price >= ema_20 * 0.998:
            score_long += 7.0
            reasons_long.append("Preço sustentando sobre a média estrutural EMA 20")

        # 10. Volume Profile Integration (VAL Liquidity Sweep & POC Proximity)
        if vp_data:
            if vp_data.get("val_sweep_bullish"):
                score_long += 15.0
                reasons_long.append(f"Liquidity Sweep no VAL (${vp_data.get('val')}) com absorção e rejeição para dentro da Value Area")
            elif vp_data.get("val", 0) > 0 and abs(current_price - vp_data["val"]) / current_price <= 0.006:
                score_long += 8.0
                reasons_long.append(f"Preço testando suporte da Value Area Low (VAL: ${vp_data.get('val')})")

        # 11. Open Interest Trap Integration (Short Trap Squeeze)
        if oi_data:
            if oi_data.get("short_trap_bullish"):
                score_long += 20.0
                reasons_long.extend(oi_data.get("reasons", []))
            elif oi_data.get("capitulation_bullish"):
                score_long += 15.0
                reasons_long.extend(oi_data.get("reasons", []))

        score_long = min(100.0, score_long)

        # --- EVALUATE SHORT SETUP ---
        score_short = 0.0
        reasons_short = []

        if struct_m15.get("bos_bearish"):
            bars_ago = struct_m15.get("bos_bars_ago", 0)
            score_short += 25.0
            b_txt = f"{bars_ago}b atrás" if bars_ago and bars_ago > 0 else "recente"
            reasons_short.append(f"Estrutura M15 com BEARISH BOS confirmado ({b_txt} — continuação de baixa)")
        elif struct_m15.get("choch_bearish"):
            bars_ago = struct_m15.get("choch_bars_ago", 0)
            score_short += 25.0
            b_txt = f"{bars_ago}b atrás" if bars_ago and bars_ago > 0 else "recente"
            reasons_short.append(f"Estrutura M15 com BEARISH CHoCH confirmado ({b_txt})")
        elif struct_m15.get("liquidity_sweep_bearish"):
            score_short += 15.0
            reasons_short.append("Liquidity Sweep institucional de topos M15")

        if in_gz:
            score_short += 25.0
            reasons_short.append(f"Preço na Golden Zone de Fibonacci (${fib_data.get('fib_618')} - ${fib_data.get('fib_50')})")
        elif dist_gz <= 0.6:
            score_short += 15.0
            reasons_short.append(f"Preço testando borda da zona de Fibonacci 50%/61.8% ({dist_gz:.2f}% de dist)")

        if is_retest or in_gz:
            if rvol <= 0.9:
                score_short += 15.0
                reasons_short.append(f"Volume Dry-Up no repique (RVOL {rvol:.2f}x — exaustão compradora)")
            elif rvol >= 1.3:
                score_short += 12.0
                reasons_short.append(f"Absorção ativa com volume no reteste (RVOL {rvol:.2f}x)")
            else:
                score_short += 8.0
        else:
            if rvol >= 1.8:
                score_short += 15.0
                reasons_short.append(f"Choque de volume institucional no M15 (RVOL {rvol:.2f}x)")
            elif rvol >= 1.2:
                score_short += 8.0
                reasons_short.append(f"Volume acima da média no M15 (RVOL {rvol:.2f}x)")

        if retest_cvd.get("bearish_absorption_divergence"):
            score_short += 20.0
            reasons_short.append("Divergência Bearish de Absorção no CVD (preço subindo sem apoio da linha amarela do CVD)")

        if is_retest or in_gz:
            if delta_z >= 1.5 and close_loc <= 0.55:
                score_short += 18.0
                reasons_short.append(f"Agressão compradora absorvida por Asks na Golden Zone (Z {delta_z:.2f})")
            elif abs(delta_z) < 1.0 or delta_z < 0:
                score_short += 14.0
                reasons_short.append(f"Delta estabilizado no teste (Z {delta_z:.2f} — arrefecimento comprador)")
        else:
            if delta_z >= 1.5 and close_loc <= 0.50:
                score_short += 18.0
                reasons_short.append(f"Forte agressão compradora absorvida (Z {delta_z:.2f})")

        if upper_wick >= 0.25:
            score_short += 15.0
            reasons_short.append(f"Pavio de rejeição superior ({upper_wick*100:.0f}%) na resistência institucional")
        elif close_loc <= 0.50:
            score_short += 10.0
            reasons_short.append("Vela fechando na metade inferior")

        if m5_trig.get("bearish_engulfing"):
            score_short += 15.0
            reasons_short.append("Gatilho tático M5 acionado: vela de engolfo vermelho rompendo mínima anterior")
        elif m5_trig.get("m5_aligned_short"):
            score_short += 8.0
            reasons_short.append("M5 alinhado a favor com Delta negativo")

        if cvd_div.get("bearish_divergence"):
            score_short += 12.0
            reasons_short.append("Divergência Bearish de CVD no M15")

        # 10. Volume Profile Integration (VAH Liquidity Sweep & POC Proximity)
        if vp_data:
            if vp_data.get("vah_sweep_bearish"):
                score_short += 15.0
                reasons_short.append(f"Liquidity Sweep no VAH (${vp_data.get('vah')}) com absorção e rejeição na Value Area High")
            elif vp_data.get("vah", 0) > 0 and abs(current_price - vp_data["vah"]) / current_price <= 0.006:
                score_short += 8.0
                reasons_short.append(f"Preço testando resistência da Value Area High (VAH: ${vp_data.get('vah')})")

        # 11. Open Interest Trap Integration (Long Trap Squeeze)
        if oi_data:
            if oi_data.get("long_trap_bearish"):
                score_short += 20.0
                reasons_short.extend(oi_data.get("reasons", []))

        # 12. HTF False-Candle Absorption Check (H1 / H4 context for Short)
        if len(h1_df) >= 3:
            last_h1 = h1_df.iloc[-1]
            rng_h1 = max(float(last_h1["high"] - last_h1["low"]), 1e-6)
            c_h1 = float(last_h1["close"])
            o_h1 = float(last_h1["open"])
            h_h1 = float(last_h1["high"])
            upper_wick_h1 = (h_h1 - c_h1) if c_h1 > o_h1 else (h_h1 - o_h1)
            upper_wick_ratio = upper_wick_h1 / rng_h1

            delta_h1 = float(last_h1.get("delta", 0.0))
            cvd_diff_h1 = float(last_h1.get("cvd", 0.0)) - float(h1_df["cvd"].iloc[-2]) if "cvd" in h1_df else 0.0
            delta_sum_3_h1 = float(h1_df["delta"].iloc[-3:].sum()) if "delta" in h1_df else 0.0
            price_change_3_h1 = c_h1 - float(h1_df["close"].iloc[-3])

            # VENDA: Vela H1 fechou VERDE, mas com exaustão e absorção vendedora
            if c_h1 > o_h1:
                if (upper_wick_ratio >= 0.38 and (delta_h1 < 0 or cvd_diff_h1 < 0)) or (delta_sum_3_h1 < 0 and price_change_3_h1 >= 0 and upper_wick_ratio >= 0.30):
                    htf_green_absorbed = True
                    score_short += 15.0
                    reasons_short.append(f"Vela H1 Verde com Absorção Passiva: Pavio Superior {upper_wick_ratio*100:.0f}% e Delta/CVD divergente (Armadilha de Compra)")

        score_short = min(100.0, score_short)

        # Decide dominant setup
        has_bearish_structure = bool(struct_m15.get("choch_bearish") or struct_m15.get("bos_bearish"))
        has_bullish_structure = bool(struct_m15.get("choch_bullish") or struct_m15.get("bos_bullish"))

        if score_short > score_long and (has_bearish_structure or score_short >= cls.THRESHOLDS["daytrade_watch"]):
            side = "SHORT"
            final_score = score_short
            reasons = reasons_short
            has_structure = has_bearish_structure
            if is_retest or in_gz:
                local_hi = float(m15_df["high"].tail(3).max())
                invalidation = local_hi * 1.0015
            else:
                origin_hi = struct_m15.get("bos_origin_high") or struct_m15.get("choch_origin_high") or float(m15_df["high"].tail(10).max())
                invalidation = origin_hi * 1.002
            risk = max(current_price * 0.003, invalidation - current_price)
            target_1 = current_price - (risk * 2.0)
            poc = vp_data.get("poc", 0.0) if vp_data else 0.0
            if 0 < poc < current_price and (current_price - poc) >= (risk * 1.8):
                target_1 = poc
            target_2 = current_price - (risk * 3.0)
        else:
            side = "LONG"
            final_score = score_long
            reasons = reasons_long
            has_structure = has_bullish_structure
            if is_retest or in_gz:
                local_lo = float(m15_df["low"].tail(3).min())
                invalidation = local_lo * 0.9985
            else:
                origin_lo = struct_m15.get("bos_origin_low") or struct_m15.get("choch_origin_low") or float(m15_df["low"].tail(10).min())
                invalidation = origin_lo * 0.998
            risk = max(current_price * 0.003, current_price - invalidation)
            target_1 = current_price + (risk * 2.0)
            poc = vp_data.get("poc", 0.0) if vp_data else 0.0
            if poc > current_price and (poc - current_price) >= (risk * 1.8):
                target_1 = poc
            target_2 = current_price + (risk * 3.0)

        if final_score >= cls.THRESHOLDS["daytrade_entry"] and (has_structure or in_gz):
            state = "DAYTRADE_ENTRY_A+"
        elif final_score >= cls.THRESHOLDS["daytrade_watch"]:
            state = "DAYTRADE_WATCH"
        else:
            state = "NO_TRADE"

        return EngineSignal(
            engine_name="DAY_TRADE",
            state=state,
            score=round(final_score, 1),
            side=side if final_score >= cls.THRESHOLDS["daytrade_watch"] else "NEUTRAL",
            price=current_price,
            invalidation=round(invalidation, 2),
            target_1=round(target_1, 2),
            target_2=round(target_2, 2),
            reasons=reasons,
            metrics={
                "rvol": round(rvol, 2),
                "delta_z": round(delta_z, 2),
                "rsi_m15": round(rsi_m15, 1),
                "in_golden_zone": in_gz,
                "is_retest": is_retest,
                "fib_50": fib_data.get("fib_50"),
                "fib_618": fib_data.get("fib_618"),
                "choch_bars_ago": struct_m15.get("choch_bars_ago"),
                "bos_bullish": struct_m15.get("bos_bullish", False),
                "retest_cvd_div": retest_cvd.get("bullish_absorption_divergence", False),
                "m5_engulfing": m5_trig.get("bullish_engulfing", False),
                "poc": vp_data.get("poc") if vp_data else None,
                "vah": vp_data.get("vah") if vp_data else None,
                "val": vp_data.get("val") if vp_data else None,
                "val_sweep": vp_data.get("val_sweep_bullish", False) if vp_data else False,
                "vah_sweep": vp_data.get("vah_sweep_bearish", False) if vp_data else False,
                "short_trap": oi_data.get("short_trap_bullish", False) if oi_data else False,
                "long_trap": oi_data.get("long_trap_bearish", False) if oi_data else False,
                "oi_change_15m_pct": oi_data.get("oi_change_15m_pct", 0.0) if oi_data else 0.0,
                "htf_red_absorbed": htf_red_absorbed,
                "htf_green_absorbed": htf_green_absorbed,
            },
        )

    # =========================================================================
    # 3. SCALP & GATILHO TÁTICO M5 (M5 Engulfing + Squeeze + OBI + Micro-M1)
    # =========================================================================
    @classmethod
    def evaluate_scalp(
        cls,
        m1_df: pd.DataFrame,
        m5_df: pd.DataFrame,
        m15_df: pd.DataFrame,
        orderbook_imbalance: float,
        current_price: float,
        spread_pct: float = 0.002,
    ) -> EngineSignal:
        """Evaluates Scalp & M5 Tactical Sniper Trigger (M5 Engulfing + ATR Squeeze + OBI).
        
        Designed to filter pure M1 micro-noise by using M5 structure shift / candle engulfing
        as the definitive tactical execution trigger for M15 expansions!
        """
        if len(m5_df) < 5:
            return EngineSignal(
                engine_name="SCALP",
                state="INSUFFICIENT_DATA",
                score=0.0,
                side="NEUTRAL",
                price=current_price,
                invalidation=None,
                target_1=None,
                target_2=None,
                reasons=["Dados históricos insuficientes em M5"],
            )

        curr_m5 = m5_df.iloc[-1]
        prev_m5 = m5_df.iloc[-2] if len(m5_df) > 1 else curr_m5

        # 1. M5 Tactical Trigger (Bullish/Bearish Engulfing / High Breakout)
        m5_trig = ConfluenceAnalyzer.detect_m5_tactical_trigger(m5_df)

        # 2. Compressão de Volatilidade (ATR Squeeze em M5)
        squeeze_m5 = ConfluenceAnalyzer.detect_volatility_squeeze(m5_df["atr_pct"], lookback=30, quantile_threshold=0.35)

        # --- EVALUATE LONG SCALP ---
        score_long = 0.0
        reasons_long = []

        if m5_trig.get("bullish_engulfing"):
            score_long += 30.0
            reasons_long.append("Gatilho tático M5 acionado: vela de engolfo verde rompendo máxima da vela anterior")
        elif m5_trig.get("m5_aligned_long"):
            score_long += 15.0
            reasons_long.append("M5 com Delta positivo e momentum ascendente")

        if squeeze_m5.get("is_squeezed"):
            score_long += 20.0
            reasons_long.append(f"Compressão de Volatilidade M5 ativa (ATR {squeeze_m5['current_atr_pct']:.3f}% ≤ {squeeze_m5['threshold_atr_pct']:.3f}%)")

        if orderbook_imbalance >= 0.15:
            score_long += 15.0
            reasons_long.append(f"Muralha compradora no Book de Ofertas (OBI +{orderbook_imbalance*100:.0f}%)")
        elif orderbook_imbalance >= 0.05:
            score_long += 8.0
            reasons_long.append(f"Desequilíbrio de Bids no Book (OBI +{orderbook_imbalance*100:.0f}%)")

        # Alinhamento M15
        if len(m15_df) > 0:
            ema_20_m15 = float(m15_df["ema_20"].iloc[-1]) if "ema_20" in m15_df.columns else 0.0
            if ema_20_m15 > 0 and current_price >= ema_20_m15 * 0.998:
                score_long += 15.0
                reasons_long.append("Alinhado com a média estrutural EMA 20 de M15")

        # Micro-confirmação M1 (se disponível)
        if len(m1_df) >= 5:
            curr_m1 = m1_df.iloc[-1]
            rvol_m1 = float(curr_m1.get("rvol", 1.0))
            delta_z_m1 = float(curr_m1.get("delta_zscore", 0.0))
            if rvol_m1 >= 1.5 or delta_z_m1 <= -1.2:
                score_long += 10.0
                reasons_long.append(f"Volume e agressão absorvida confirmados no M1 ({rvol_m1:.1f}x)")

        score_long = min(100.0, score_long)

        # --- EVALUATE SHORT SCALP ---
        score_short = 0.0
        reasons_short = []

        if m5_trig.get("bearish_engulfing"):
            score_short += 30.0
            reasons_short.append("Gatilho tático M5 acionado: vela de engolfo vermelho rompendo mínima anterior")
        elif m5_trig.get("m5_aligned_short"):
            score_short += 15.0
            reasons_short.append("M5 com Delta negativo e momentum descendente")

        if squeeze_m5.get("is_squeezed"):
            score_short += 20.0
            reasons_short.append(f"Compressão de Volatilidade M5 ativa ({squeeze_m5['current_atr_pct']:.3f}%)")

        if orderbook_imbalance <= -0.15:
            score_short += 15.0
            reasons_short.append(f"Pressão vendedora no Book de Ofertas (OBI {orderbook_imbalance*100:.0f}%)")
        elif orderbook_imbalance <= -0.05:
            score_short += 8.0
            reasons_short.append(f"Desequilíbrio de Asks no Book (OBI {orderbook_imbalance*100:.0f}%)")

        if len(m15_df) > 0:
            ema_20_m15 = float(m15_df["ema_20"].iloc[-1]) if "ema_20" in m15_df.columns else 0.0
            if ema_20_m15 > 0 and current_price <= ema_20_m15 * 1.002:
                score_short += 15.0
                reasons_short.append("Alinhado abaixo da EMA 20 de M15")

        score_short = min(100.0, score_short)

        if score_short > score_long and (m5_trig.get("bearish_engulfing") or score_short >= cls.THRESHOLDS["scalp_watch"]):
            side = "SHORT"
            final_score = score_short
            reasons = reasons_short
            trigger_hi = max(float(curr_m5["high"]), float(prev_m5["high"]))
            invalidation = trigger_hi * 1.0012
            risk = max(current_price * 0.0025, invalidation - current_price)
            target_1 = current_price - (risk * 1.5)
            target_2 = current_price - (risk * 2.5)
            if final_score >= cls.THRESHOLDS["scalp_entry"] or (final_score >= 68.0 and m5_trig.get("bearish_engulfing")):
                state = "SCALP_M5_ENGULFING_SHORT" if m5_trig.get("bearish_engulfing") else "SCALP_PRE_EXPLOSION_SHORT"
            elif final_score >= cls.THRESHOLDS["scalp_watch"]:
                state = "SCALP_WATCH"
            else:
                state = "NO_TRADE"
        else:
            side = "LONG"
            final_score = score_long
            reasons = reasons_long
            trigger_lo = min(float(curr_m5["low"]), float(prev_m5["low"]))
            invalidation = trigger_lo * 0.9988
            risk = max(current_price * 0.0025, current_price - invalidation)
            target_1 = current_price + (risk * 1.5)
            target_2 = current_price + (risk * 2.5)
            if final_score >= cls.THRESHOLDS["scalp_entry"] or (final_score >= 68.0 and m5_trig.get("bullish_engulfing")):
                state = "SCALP_M5_ENGULFING_LONG" if m5_trig.get("bullish_engulfing") else "SCALP_PRE_EXPLOSION_LONG"
            elif final_score >= cls.THRESHOLDS["scalp_watch"]:
                state = "SCALP_WATCH"
            else:
                state = "NO_TRADE"

        return EngineSignal(
            engine_name="SCALP",
            state=state,
            score=round(final_score, 1),
            side=side if final_score >= cls.THRESHOLDS["scalp_watch"] else "NEUTRAL",
            price=current_price,
            invalidation=round(invalidation, 2),
            target_1=round(target_1, 2),
            target_2=round(target_2, 2),
            reasons=reasons,
            metrics={
                "m5_engulfing": m5_trig.get("bullish_engulfing", False) if side == "LONG" else m5_trig.get("bearish_engulfing", False),
                "m5_delta": m5_trig.get("m5_delta", 0.0),
                "obi": round(orderbook_imbalance, 3),
                "is_squeezed": squeeze_m5.get("is_squeezed", False),
                "atr_pct_m5": squeeze_m5.get("current_atr_pct", 0.0),
            },
        )

