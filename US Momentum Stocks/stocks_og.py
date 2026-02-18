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
        """
        Initializes the SectorTopUniverse.

        Parameters
        ----------
        algo : QCAlgorithm
            The algorithm instance.
        blacklist : list of str, optional
            List of ticker strings to exclude from selection.
        """
        self.algo = algo
        self.blacklist = set(blacklist or [])
        super().__init__(self._select)

    def _select(self, fundamentals):
        """
        Performs the fundamental selection logic.

        Parameters
        ----------
        fundamentals : list[Fundamental]
            The list of fundamental data objects.

        Returns
        -------
        list[Symbol]
            The symbols to include in the universe.
        """
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
        """
        Initializes the algorithm state, parameters, and scheduling.
        """
        self.SetStartDate(2004, 1, 1)
        self.SetCash(100_000)

        # --------------------
        # Momentum parameters
        # --------------------
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
        self.current_band_idx = {}
        self.bottom_frac_hist = deque(maxlen=3)
        self.BOTTOM_LEVELS = {0, 1, 2, 3, 4}

        # track worst breadth
        self.min_bottom_frac = 1.0
        self.was_risk_off = False  
        self.SetUniverseSelection(
            SectorTopUniverse(self, blacklist={"GME", "AMC"})
        )

        self.symbols = set()

        self.adx_limit = 35
        self.adx_period = 14

        # Per-symbol state
        self.ma = {}
        self.adx = {}
        self.stretch_max = {}
        self.close_win = {}
        self.stretch_ema = {}
        self.band_hist = {}

        self.SetWarmUp(300)

        self.Schedule.On(
            self.DateRules.MonthEnd("SPY"),
            self.TimeRules.BeforeMarketClose("SPY", 5),
            self.Rebalance
        )

    def OnSecuritiesChanged(self, changes):
        """
        Initializes/cleans up indicators when securities enter or leave the universe.

        Parameters
        ----------
        changes : SecurityChanges
            The added and removed securities.
        """
        for sec in changes.AddedSecurities:
            sec.SetFeeModel(ConstantFeeModel(0))
            s = sec.Symbol
            self.symbols.add(s)

            self.stretch_max[sec.Symbol] = 0.0
            self.ma[s] = self.EMA(s, self.band_len, Resolution.Daily)
            self.adx[s] = self.ADX(s, self.adx_period, Resolution.Daily)
            self.stretch_ema[s] = self.EMA(s, self.band_len, Resolution.Daily)
            self.close_win[s] = RollingWindow[float](self.band_len)
            self.band_hist[s] = RollingWindow[int](self.hist_len)

        for sec in changes.RemovedSecurities:
            s = sec.Symbol
            self.symbols.discard(s)
            self.ma.pop(s, None)
            self.stretch_max.pop(sec.Symbol, None)            
            self.stretch_ema.pop(s, None)
            self.close_win.pop(s, None)
            self.band_hist.pop(s, None)
            self.current_band_idx.pop(s, None)

    def OnData(self, data):
        """
        Updates technical indicators and band indices on every new data slice.
        Also tracks peak stretch (Z-score) to anticipate momentum blow-offs.
        """
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

            # Calculate Standard Deviation and Mean
            dev = np.std(list(self.close_win[s]))
            if dev <= 0:
                continue

            mid = self.ma[s].Current.Value
            
            # 1. Calculate the current Stretch (Z-score)
            stretch = abs(close - mid) / dev
            self.stretch_ema[s].Update(self.Time, stretch)

            # 2. Track the long-term peak Stretch for this symbol
            # This identifies the "Maximum Velocity" of the current multi-year trend
            if s not in self.stretch_max:
                self.stretch_max[s] = 0.0
            
            # Update the peak stretch seen so far
            if stretch > self.stretch_max[s]:
                self.stretch_max[s] = stretch

            # 3. Calculate the Price Bands
            bands = [
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
                mid + dev * 1.618
            ]

            # 4. Map price to Band Index
            idx = self._band_index(close, bands)
            self.current_band_idx[s] = idx

    def _band_index(self, price, bands):
        """
        Determines which index a price occupies within a set of bands.

        Parameters
        ----------
        price : float
            Current price of the asset.
        bands : list[float]
            List of price levels defining the bands.

        Returns
        -------
        int
            The index of the band.
        """
        for i in range(len(bands) - 1):
            if bands[i] <= price < bands[i + 1]:
                return i
        return len(bands) - 2

    def Rebalance(self):
        """
        Main execution logic for rebalancing the portfolio at month-end.
        
        Evaluates market breadth stress, ranks momentum, and applies 
        historical high band scaling to position sizing.
        """
        if self.IsWarmingUp:
            return

        # -------- UNIVERSE-WIDE BREADTH --------
        idxs = list(self.current_band_idx.values())
        if len(idxs) < 50:
            return

        bottom_frac = sum(i in self.BOTTOM_LEVELS for i in idxs) / len(idxs)

        if not hasattr(self, 'max_stress_level'): self.max_stress_level = 0.0
        self.max_stress_level = max(self.max_stress_level, bottom_frac)

        # -------- BREADTH REGIME --------
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
            self.Debug("Risk-Off.") # Corrected from lowercase debug
            return

        # -------- NORMAL REBALANCE LOGIC --------
        hist = self.History(
            list(self.symbols),
            max(self.lookbacks) + 1,
            Resolution.Daily
        )

        if hist.empty:
            return

        closes = hist["close"].unstack(0)
        momentum = {}

        for s in self.symbols:
            if s not in closes:
                continue

            px = closes[s]
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
            ema = self.ma[s].Current.Value

            if price <= ema:
                continue

            if mom > 0:
                momentum[s] = mom

        if not momentum:
            self.Liquidate()
            return

        top = sorted(momentum, key=momentum.get, reverse=True)[:self.stock_count]

        scaled = {}
        for s in top:
            if not self.ma[s].IsReady or not self.stretch_ema[s].IsReady:
                continue

            dev = np.std(list(self.close_win[s]))
            if dev <= 0:
                continue

            mid = self.ma[s].Current.Value
            lm = self.stretch_ema[s].Current.Value

            lm2 = lm / 2.0
            lm3 = lm2 * 0.38196601
            lm4 = lm * 1.38196601
            lm5 = lm * 1.61803399
            lm6 = (lm + lm2) / 2.0

            bands = [
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
                mid + dev * lm5
            ]

            price = self.Securities[s].Price
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

            scaled[s] = momentum[s] * scale

            # 2. NEW: Anticipatory Exhaustion Scaling
            # Pull the current stretch and the peak stretch we recorded in OnData
            current_stretch = self.stretch_ema[s].Current.Value
            peak_stretch = self.stretch_max.get(s, 0.0)

            # If we are in high bands (idx >= 10) but the stretch has decayed 20% from peak
            if idx >= 10 and peak_stretch > 0:
                if current_stretch < (peak_stretch * 0.80):
                    # We override the scale to its minimum (0.2) because 
                    # the momentum is 'exhausted' even if the price is still high.
                    scale = 0.2
                    self.Debug(f"ANTICIPATION: Scaling down {s.Value} due to Stretch Exhaustion.")

            # 3. Apply the final scale to momentum
            scaled[s] = momentum[s] * scale

        # -------- FINAL WEIGHTING LOGIC --------
        if not scaled:
            self.Liquidate()
            self.Debug("No Assets to trade.")
            return

        # 1. Proportional Weights
        total_scaled = sum(scaled.values())
        raw_weights = {s: (v / total_scaled) for s, v in scaled.items()}

        # 2. Apply 20% Cap
        capped_weights = {s: min(self.max_weight, w) for s, w in raw_weights.items()}

        # 3. Re-scale to ensure we are actually using our capital
        current_sum = sum(capped_weights.values())
        if current_sum > 0:
            final_weights = {s: w / current_sum for s, w in capped_weights.items()}
        else:
            final_weights = {}

        # 4. Execution
        self.Liquidate()
        for s, w in final_weights.items():
            if w > 0:
                self.SetHoldings(s, w)

        # Filter for weights > 0 before joining the string
        output = ", ".join([f"{s.Value}: {w*100:.1f}%" for s, w in final_weights.items() if w > 0])

        # Only print if there's actually something to show
        if output:
            self.Debug(f"Weights @ {self.Time.strftime('%Y-%m-%d')}: {output}")
        else:
            self.Debug(f"Weights @ {self.Time.strftime('%Y-%m-%d')}: No active positions (All weights 0%)")
