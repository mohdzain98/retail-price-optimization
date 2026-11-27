"""Forecast and promo layer (Phase 5): trade promotion optimization under a budget.

forecasters.py (naive, seasonal index, SARIMAX, XGBoost), forecast.py (future
calendar, backtest, baseline forecast), candidates.py (tactic effects,
option economics), planner.py (MILP + greedy),
frontier.py (budget frontier), service.py (facade + factory).
"""

from src.promo.service import PromoFactory, PromoOutputs, PromoReport, PromoService

__all__ = ["PromoFactory", "PromoOutputs", "PromoReport", "PromoService"]
