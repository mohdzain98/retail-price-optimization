"""Horizon KPIs of promo scenarios: totals over the forecast horizon and % changes.

HorizonKpis adds each plan's incremental volume, revenue and profit to the baseline
at recommended prices, next to current and recommended prices with no promotion.
Shared by the promo stage and the Streamlit simulator.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from src.schema import Col, PromoScenario

TOTALS: tuple[Col, ...] = (Col.VOLUME, Col.REVENUE, Col.PROFIT)


class HorizonKpis:
    """Scenario totals for one objective; changes are against current prices, no promo.

    Profit is net of fixed fees; the discount itself is already in revenue.
    """

    def summary(
        self, forecast: pd.DataFrame, prices: pd.DataFrame, plans: Mapping[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """One row per scenario: totals, trade spend, promos and % changes.

        `forecast` holds UPC and both baseline forecasts; `prices` holds UPC, unit cost,
        current and regular price; `plans` maps a scenario name to its chosen options.
        """
        indexed = prices.drop_duplicates(Col.UPC).set_index(Col.UPC)
        upc = forecast[Col.UPC]
        cost = upc.map(indexed[Col.UNIT_COST])

        def no_promo(volume: pd.Series, price: pd.Series) -> dict[Col, float]:
            return {Col.VOLUME: volume.sum(), Col.REVENUE: (volume * price).sum(),
                    Col.PROFIT: (volume * (price - cost)).sum(), Col.TRADE_SPEND: 0.0,
                    Col.N_PROMOS: 0}  # fmt: skip

        current = no_promo(forecast[Col.FORECAST_CURRENT], upc.map(indexed[Col.CURRENT_PRICE]))
        regular = no_promo(forecast[Col.FORECAST_BASELINE], upc.map(indexed[Col.REGULAR_PRICE]))
        rows: dict[str, dict[Col, float]] = {
            PromoScenario.CURRENT_NO_PROMO.value: current,
            PromoScenario.RECOMMENDED_NO_PROMO.value: regular,
        }
        for scenario, plan in plans.items():
            rows[str(scenario)] = {
                Col.VOLUME: regular[Col.VOLUME] + plan[Col.INCREMENTAL].sum(),
                Col.REVENUE: regular[Col.REVENUE] + plan[Col.INCREMENTAL_REVENUE].sum(),
                Col.PROFIT: regular[Col.PROFIT] + plan[Col.INCREMENTAL_PROFIT].sum(),
                Col.TRADE_SPEND: plan[Col.TRADE_SPEND].sum(),
                Col.N_PROMOS: len(plan),
            }
        frame = pd.DataFrame.from_dict(rows, orient="index").rename_axis(Col.SCENARIO).reset_index()
        for col in TOTALS:
            frame[f"{col.value}_change"] = frame[col] / current[col] - 1
        return frame
