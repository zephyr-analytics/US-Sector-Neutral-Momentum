"""
Momentum and Historical Band Ceiling Sizing Algorithm.

This module implements a sector-neutral large-cap momentum strategy. 
It uses a universe-wide breadth indicator to manage risk-on/risk-off regimes
and applies historical band-based ceilings to individual position sizing.
"""

from AlgorithmImports import *
from collections import defaultdict, deque
import numpy as np

# ====================================================
# Sector-Neutral Large-Cap Universe
# ====================================================
class SectorTopUniverse(FundamentalUniverseSelectionModel):
    """
    Selection model for a sector-neutral large-cap universe.

    Filters for primary exchange listing, minimum price, and minimum market cap, 
    then selects the top 75 stocks by market capitalization within each 
    Morningstar sector.
    """
    def __init__(self, algo, blacklist=None):
        self.algo = algo
        self.blacklist = set(blacklist or [])
        super().__init__(self._select)

    def _select(self, fundamentals):
        buckets = defaultdict(list)

        for f in fundamentals:
            if not f.has_fundamental_data:
                continue
            if f.symbol.Value in self.blacklist:
                continue
            if f.company_reference.primary_exchange_id not in ("NYS", "NAS", "ASE"):
                continue
            if f.price is None or f.price <= 5:
                continue
            if f.market_cap is None or f.market_cap < 5_000_000_000:
                continue

            sector = f.asset_classification.morningstar_sector_code
            if sector is None:
                continue

            buckets[sector].append(f)

        symbols = []
        for _, stocks in buckets.items():
            stocks.sort(key=lambda x: x.market_cap, reverse=True)
            symbols.extend(s.symbol for s in stocks[:100])

        return symbols


# ====================================================
# Momentum + Historical Band Ceiling Sizing
# ====================================================
class StockOnlyMomentum(QCAlgorithm):
    """
    Momentum strategy with dynamic position sizing based on price band history.

    This algorithm selects top momentum stocks, verifies they are above 
    their EMA, and sizes them based on their current price position 
    relative to historical peak price bands (Z-score based).
    """

    def Initialize(self):
        # --------------------
        # Momentum parameters
        # --------------------
        self.set_start_date(2004, 1, 1)
        self.lookbacks = [21, 63, 126, 189, 252]
        self.stock_count = 10
        self.max_weight = 0.20

        # --------------------
        # Band parameters
        # --------------------
        self.band_len = 189
        self.hist_len = 126

        self.UniverseSettings.Resolution = Resolution.Daily
        self.UniverseSettings.DataNormalizationMode = DataNormalizationMode.TOTAL_RETURN

        # -------- BREADTH STATE --------
        self.allow_universe = True
        self.BOTTOM_LEVELS = {0, 1, 2, 3, 4}

        self.max_stress_level = 0.0
        self.was_risk_off = False

        self.SetUniverseSelection(
            SectorTopUniverse(self, blacklist={"GME", "AMC"})
        )

        # -------- STOP LOSS --------
        self.period_stop_loss = 0.25
        self.period_entry_value = {}

        self.symbols = set()
        self.adx_limit = 35
        self.adx_period = 14

        # Per-symbol state
        self.ma = {}
        self.adx = {}
        self.stretch_ema = {}
        self.stretch_max = {}
        self.close_win = {}
        self.band_hist = {}

        self.SetWarmUp(300)

        self.Schedule.On(
            self.DateRules.month_end("SPY"),
            self.TimeRules.before_market_close("SPY", 120),
            self.Rebalance
        )

    def OnSecuritiesChanged(self, changes):
        for sec in changes.AddedSecurities:
            sec.SetFeeModel(ConstantFeeModel(0))
            s = sec.Symbol
            self.symbols.add(s)

            self.ma[s] = self.EMA(s, self.band_len, Resolution.Daily)
            self.adx[s] = self.ADX(s, self.adx_period, Resolution.Daily)
            self.stretch_ema[s] = self.EMA(s, self.band_len, Resolution.Daily)
            self.close_win[s] = RollingWindow[float](self.band_len)
            self.band_hist[s] = RollingWindow[int](self.hist_len)
            self.stretch_max[s] = 0.0

        for sec in changes.RemovedSecurities:
            s = sec.Symbol
            self.symbols.discard(s)
            for store in (self.ma, self.adx, self.stretch_ema,
                          self.close_win, self.band_hist, self.stretch_max):
                store.pop(s, None)

    def OnData(self, data):
        for s in list(self.symbols):
            if not data.ContainsKey(s):
                continue

            bar = data[s]
            if bar is None:
                continue

            close = bar.Close
            self.close_win[s].Add(close)

            if not self.close_win[s].IsReady or not self.ma[s].IsReady:
                continue

            dev = np.std(list(self.close_win[s]))
            if dev <= 0:
                continue

            mid = self.ma[s].Current.Value
            stretch = abs(close - mid) / dev
            self.stretch_ema[s].Update(self.Time, stretch)

            if stretch > self.stretch_max[s]:
                self.stretch_max[s] = stretch

            # Per-symbol cumulative period stop-loss
            if (not self.IsWarmingUp
                    and self.Portfolio[s].Invested
                    and s in self.period_entry_value
                    and self.period_entry_value[s] > 0):
                entry = self.period_entry_value[s]
                period_loss = (entry - close) / entry
                if period_loss >= self.period_stop_loss:
                    self.Liquidate(s)
                    self.Debug(
                        f"STOP-LOSS {s.Value} @ {self.Time.strftime('%Y-%m-%d')}: "
                        f"period loss {period_loss:.1%}"
                    )

    def _band_index(self, price, bands):
        for i in range(len(bands) - 1):
            if bands[i] <= price < bands[i + 1]:
                return i
        return len(bands) - 2

    def _compute_bands(self, mid, dev, lm):
        lm2 = lm / 2.0
        lm3 = lm2 * 0.38196601
        lm4 = lm * 1.38196601
        lm5 = lm * 1.61803399
        lm6 = (lm + lm2) / 2.0

        return [
            mid - dev * lm5,
            mid - dev * lm4,
            mid - dev * lm,
            mid - dev * lm6,
            mid - dev * lm2,
            mid - dev * lm3,
            mid,
            mid + dev * lm3,
            mid + dev * lm2,
            mid + dev * lm6,
            mid + dev * lm,
            mid + dev * lm4,
            mid + dev * lm5,
        ]

    def _compute_breadth_bands(self, mid, dev):
        return [
            mid - dev * 1.618,
            mid - dev * 1.382,
            mid - dev,
            mid - dev * 0.809,
            mid - dev * 0.5,
            mid - dev * 0.382,
            mid,
            mid + dev * 0.382,
            mid + dev * 0.5,
            mid + dev * 0.809,
            mid + dev,
            mid + dev * 1.382,
            mid + dev * 1.618,
        ]

    def Rebalance(self):
        """
        Main execution logic for rebalancing the portfolio at month-end.

        All band calculations (breadth and sizing) happen here so that the
        band state is always consistent with the moment a trade decision is made.
        """
        if self.IsWarmingUp:
            return

        self.period_entry_value = {}

        # -------- COMPUTE BANDS AND BREADTH AT REBALANCE TIME --------
        band_indices = {}
        for s in list(self.symbols):
            if not self.close_win[s].IsReady or not self.ma[s].IsReady:
                continue

            closes = list(self.close_win[s])
            dev = np.std(closes)
            if dev <= 0:
                continue

            mid = self.ma[s].Current.Value
            price = self.Securities[s].Price

            bands = self._compute_breadth_bands(mid, dev)
            band_indices[s] = self._band_index(price, bands)

        if len(band_indices) < 50:
            return

        # -------- BREADTH REGIME --------
        bottom_frac = sum(
            i in self.BOTTOM_LEVELS for i in band_indices.values()
        ) / len(band_indices)

        self.max_stress_level = max(self.max_stress_level, bottom_frac)

        if bottom_frac >= 0.45:
            self.allow_universe = False
            self.was_risk_off = True

        elif self.was_risk_off:
            denominator = max(self.max_stress_level, 0.10)
            improvement = (self.max_stress_level - bottom_frac) / denominator

            if improvement >= 0.60 or bottom_frac < 0.15:
                self.Debug(f"Recovery! Stress: {bottom_frac:.1%}. Resetting ceilings.")
                for s in self.symbols:
                    if s in self.band_hist:
                        self.band_hist[s] = RollingWindow[int](self.hist_len)

                self.allow_universe = True
                self.was_risk_off = False
                self.max_stress_level = 0.0
        else:
            self.allow_universe = True

        if not self.allow_universe:
            self.Liquidate()
            self.Debug("Risk-Off.")
            return

        # -------- MOMENTUM RANKING --------
        hist = self.History(
            list(self.symbols),
            max(self.lookbacks) + 1,
            Resolution.Daily
        )

        if hist.empty:
            return

        closes_df = hist["close"].unstack(0)
        momentum = {}

        for s in self.symbols:
            if s not in closes_df:
                continue

            px = closes_df[s]
            if len(px) < max(self.lookbacks) + 1:
                continue

            if not self.adx[s].IsReady or self.adx[s].Current.Value > self.adx_limit:
                continue

            mom = np.mean([
                px.iloc[-1] / px.iloc[-lb - 1] - 1
                for lb in self.lookbacks
            ])

            if not self.ma[s].IsReady:
                continue

            price = self.Securities[s].Price
            if price <= self.ma[s].Current.Value:
                continue

            if mom > 0:
                momentum[s] = mom

        if not momentum:
            self.Liquidate()
            return

        top = sorted(momentum, key=momentum.get, reverse=True)[:self.stock_count]

        # -------- SIZING BANDS (computed fresh at rebalance) --------
        scaled = {}
        for s in top:
            if not self.ma[s].IsReady or not self.stretch_ema[s].IsReady:
                continue
            if not self.close_win[s].IsReady:
                continue

            dev = np.std(list(self.close_win[s]))
            if dev <= 0:
                continue

            mid = self.ma[s].Current.Value
            lm = self.stretch_ema[s].Current.Value
            price = self.Securities[s].Price

            bands = self._compute_bands(mid, dev, lm)
            idx = self._band_index(price, bands)

            self.band_hist[s].Add(idx)
            hist_idx = list(self.band_hist[s])
            historical_high = max(hist_idx) if hist_idx else idx

            if historical_high <= 0:
                scale = 1.0
            elif idx >= historical_high:
                scale = 0.0
            else:
                scale = max(0.2, 1.0 - idx / historical_high)

            # Anticipatory exhaustion scaling
            current_stretch = self.stretch_ema[s].Current.Value
            peak_stretch = self.stretch_max.get(s, 0.0)

            if idx >= 10 and peak_stretch > 0:
                if current_stretch < (peak_stretch * 0.80):
                    scale = 0.2
                    self.Debug(f"ANTICIPATION: Scaling down {s.Value} due to Stretch Exhaustion.")

            scaled[s] = momentum[s] * scale

        # -------- FINAL WEIGHTING --------
        if not scaled:
            self.Liquidate()
            self.Debug("No Assets to trade.")
            return

        # Drop zero-scaled entries before normalizing
        # scaled = {s: v for s, v in scaled.items() if v > 1e-9}

        if not scaled:
            self.Liquidate()
            self.Debug("No Assets passed scaling.")
            return

        total_scaled = sum(scaled.values())
        raw_weights = {s: v / total_scaled for s, v in scaled.items()}
        capped_weights = {s: min(self.max_weight, w) for s, w in raw_weights.items()}

        current_sum = sum(capped_weights.values())
        final_weights = (
            {s: w / current_sum for s, w in capped_weights.items()}
            if current_sum > 0 else {}
        )

        # Exit positions not in the new target set
        for holding in self.Portfolio.Values:
            if holding.Invested and holding.Symbol not in final_weights:
                self.Liquidate(holding.Symbol)

        # Enter/resize target positions — skip if a liquidation order is still open
        for s, w in final_weights.items():
            if w > 0 and not self.Transactions.GetOpenOrders(s):
                self.SetHoldings(s, w)
                self.period_entry_value[s] = self.Securities[s].Price

        output = ", ".join(
            f"{s.Value}: {w*100:.1f}%"
            for s, w in final_weights.items() if w > 0
        )
        if output:
            self.Debug(f"Weights @ {self.Time.strftime('%Y-%m-%d')}: {output}")
        else:
            self.Debug(f"Weights @ {self.Time.strftime('%Y-%m-%d')}: No active positions")
