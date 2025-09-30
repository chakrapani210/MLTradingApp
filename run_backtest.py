"""Convenience script to run enhanced backtests from the command line.

Usage (PowerShell):
  python run_backtest.py --symbol AAPL --months 3
  python run_backtest.py --symbol AAPL,MSFT,GOOGL --months 6 --benchmark SPY

Outputs:
  - Writes JSON results for each symbol to results/backtests/<symbol>_<timestamp>.json
  - Prints a concise performance summary to stdout

This script ensures the package is imported as `src.*` so that relative imports work
inside the project modules.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Lazy import to show clearer errors if dependencies missing
try:
    from src.enhanced_orchestrator import ProductionTradingOrchestrator
except Exception as e:  # pragma: no cover
    print(f"[ERROR] Failed to import orchestrator: {e}")
    sys.exit(1)

def run_backtest(symbol: str, months: int, benchmark: str, rebalance: str) -> Dict[str, Any]:
    orchestrator = ProductionTradingOrchestrator()
    result = orchestrator.run_comprehensive_backtest(
        symbol=symbol,
        backtest_period_months=months,
        benchmark_symbol=benchmark,
        rebalance_frequency=rebalance,
    )
    return result

def save_result(symbol: str, result: Dict[str, Any]) -> Path:
    out_dir = ROOT / "results" / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{symbol}_{stamp}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    return out_path

def summarize(result: Dict[str, Any]) -> str:
    perf = result.get("performance") or {}
    trades = result.get("trading_metrics") or {}
    return (
        f"Return: {perf.get('total_return')} | Annualized: {perf.get('annualized_return')} | "
        f"Sharpe: {perf.get('sharpe_ratio')} | MaxDD: {perf.get('max_drawdown')} | Trades: {trades.get('total_trades')}"
    )

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run enhanced ML backtests")
    p.add_argument("--symbol", required=True, help="Single symbol or comma separated list (e.g. AAPL,MSFT)")
    p.add_argument("--months", type=int, default=3, help="Backtest period in months (default 3)")
    p.add_argument("--benchmark", default="SPY", help="Benchmark symbol (default SPY)")
    p.add_argument("--rebalance", default="daily", choices=["daily", "weekly", "monthly"], help="Rebalance frequency")
    p.add_argument("--no-save", action="store_true", help="Do not write JSON result files")
    return p.parse_args()


def main():  # pragma: no cover - CLI wrapper
    args = parse_args()
    symbols: List[str] = [s.strip().upper() for s in args.symbol.split(",") if s.strip()]

    print("\n=== ML Trading System Backtest Runner ===")
    print(f"Symbols: {', '.join(symbols)} | Period: {args.months}m | Benchmark: {args.benchmark} | Rebalance: {args.rebalance}")
    print("------------------------------------------------------------")

    all_results: Dict[str, Dict[str, Any]] = {}
    for sym in symbols:
        try:
            print(f"[RUN] Backtesting {sym} ...")
            result = run_backtest(sym, args.months, args.benchmark, args.rebalance)
            all_results[sym] = result
            if not args.no_save:
                path = save_result(sym, result)
                print(f"[OK ] Saved -> {path.relative_to(ROOT)}")
            print(f"[SUM] {sym}: {summarize(result)}\n")
        except KeyboardInterrupt:
            print("[INTERRUPT] User cancelled")
            break
        except Exception as e:
            print(f"[FAIL] {sym}: {e}")

    # Aggregate summary
    if all_results:
        print("=== SUMMARY ===")
        for sym, res in all_results.items():
            print(f"{sym}: {summarize(res)}")
    else:
        print("No successful backtests.")

    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
