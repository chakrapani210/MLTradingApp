"""Scheduled Trading Task
Runs daily at configured market time to:
1. Load configured symbols
2. Apply a lightweight position rule (no ML orchestrator dependency):
     - If no open position: allocate a configurable fraction of cash to a BUY
     - If position already exists: skip (no rebalance logic yet)
3. Place paper trading orders using `PaperTradingAccount` only (no direct orchestrator access)

Environment Variables:
    SCHEDULED_BUY_FRACTION  -> Fraction of available cash to allocate per new symbol (default 0.05 = 5%)

Time: Intended for execution at 14:30 CST (Central Time US). Use Windows Task Scheduler or a cron-like service to trigger this script.
"""
from __future__ import annotations
import asyncio
import os
from datetime import datetime
import zoneinfo
from typing import List

from config_manager import get_trading_symbols
from src.trading.paper_trading import PaperTradingAccount, PaperTradingConfig, Order, OrderType, OrderSide
from src.trading.paper_trading_app import PaperTradingApp

TARGET_TZ = zoneinfo.ZoneInfo("America/Chicago")  # CST/CDT automatically handled

async def run_scheduled_trading(symbol_limit: int | None = None):
    run_time_local = datetime.now(TARGET_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')
    print(f"\n=== SCHEDULED TRADING RUN @ {run_time_local} ===")

    try:
        allocation_fraction = float(os.getenv("SCHEDULED_BUY_FRACTION", "0.05"))
        if allocation_fraction <= 0 or allocation_fraction > 1:
            raise ValueError
    except ValueError:
        print("[CONFIG] Invalid SCHEDULED_BUY_FRACTION; using default 0.05")
        allocation_fraction = 0.05

    app = PaperTradingApp(starting_capital=100000)
    # Ensure account connected
    await app.connect_account()

    result = await app.run_scheduled_auto_trading(
        allocation_fraction=allocation_fraction,
        symbol_limit=symbol_limit,
        rule="no_position_buy",
        auto_disconnect=True,
        verbose=True
    )

    # Show performance summary if available
    if hasattr(app.account, 'get_performance_summary') and app.account:
        perf = app.account.get_performance_summary()  # type: ignore[call-arg]
        print("\n=== PERFORMANCE SUMMARY ===")
        print(f"Current cash: {perf['current_cash']:.2f}  Total equity: {perf['current_total_equity']:.2f}  Return %: {perf['total_return_pct']:.2f}")

if __name__ == '__main__':
    asyncio.run(run_scheduled_trading())
