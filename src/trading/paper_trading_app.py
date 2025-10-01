"""PaperTradingApp
Concrete implementation of TradingAppBase for the PaperTradingAccount.
Provides account connection hooks and reuses all decision / scheduled auto trading logic.
"""
from __future__ import annotations

from typing import Optional

from .trading_app_base import TradingAppBase
from .paper_trading import PaperTradingAccount, PaperTradingConfig


class PaperTradingApp(TradingAppBase):
    """Trading application wired to the paper trading account."""

    def __init__(self, starting_capital: float = 100000.0, commission_rate: float = 0.0):
        super().__init__(starting_capital=starting_capital, commission_rate=commission_rate)
        # Instantiate account lazily at connect time for clarity
        self._paper_config = PaperTradingConfig(initial_cash=starting_capital)

    async def connect_account(self) -> bool:
        if self.account and getattr(self.account, '_is_connected', False):  # type: ignore[attr-defined]
            return True
        self.account = PaperTradingAccount(self._paper_config)
        return await self.account.connect()

    async def disconnect_account(self) -> bool:
        if self.account:
            return await self.account.disconnect()
        return True

__all__ = ["PaperTradingApp"]
