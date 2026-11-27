"""What-if simulator layer (Phase 6): logic behind the Streamlit app in app/.

data.py (processed outputs), pricing.py (re-optimize or score regular prices),
promo.py (re-plan promotions with the MILP) and charts.py (Altair charts).
The app pages stay thin: they read widgets, call these classes and render.
"""

from src.simulator.charts import SimulatorCharts
from src.simulator.data import SimulatorData
from src.simulator.pricing import PriceResult, PriceSettings, PriceSimulator
from src.simulator.promo import PromoResult, PromoSettings, PromoSimulator

__all__ = [
    "PriceResult",
    "PriceSettings",
    "PriceSimulator",
    "PromoResult",
    "PromoSettings",
    "PromoSimulator",
    "SimulatorCharts",
    "SimulatorData",
]
