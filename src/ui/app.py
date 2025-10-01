"""Simple FastAPI web UI for Paper Trading and Robinhood App selection.

Features implemented (MVP):
- Landing page: choose Paper Trading or (placeholder) Robinhood App
- Paper Trading page:
  * Loads existing account state (data/paper_account_state.json) or creates default
  * Input form for tickers (comma separated), default TSLA
  * Run Trade button triggers mock trade logic: fetch price, generate naive signal, execute order via PaperTradingAccount
  * Displays current account summary (cash, equity, positions)
  * Transactions tab shows transaction history (auto-refresh after trade)

Reasoning:
We reuse PaperTradingAccount for state persistence. For simplicity, signals are placeholder (basic rule). Future enhancement: integrate ProductionTradingOrchestrator decision pipeline.
"""
from __future__ import annotations
import os
import asyncio
from typing import List, Optional
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.trading.paper_trading import PaperTradingAccount, PaperTradingConfig, Order, OrderType, OrderSide
from src.trading.paper_trading_app import PaperTradingApp

# --- Setup FastAPI and Jinja2 ---
app = FastAPI(title="Trading UI")

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
if not os.path.exists(TEMPLATE_DIR):
    os.makedirs(TEMPLATE_DIR, exist_ok=True)

env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(['html', 'xml'])
)

# Global (singleton) PaperTradingApp instance
_paper_app: Optional[PaperTradingApp] = None

async def get_paper_app() -> PaperTradingApp:
    global _paper_app
    if _paper_app is None:
        _paper_app = PaperTradingApp(starting_capital=100000.0)
        await _paper_app.connect_account()
    elif not _paper_app.account or not getattr(_paper_app.account, '_is_connected', False):  # type: ignore[attr-defined]
        await _paper_app.connect_account()
    return _paper_app

STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
if os.path.exists(STATIC_DIR):
    app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')

# --- Routes ---
@app.get('/', response_class=HTMLResponse)
async def index():
    tpl = env.get_template('index.html')
    return HTMLResponse(tpl.render())

@app.get('/paper', response_class=HTMLResponse)
async def paper_view(reset: str | None = None):
    app_inst = await get_paper_app()
    account = app_inst.account
    summary = account.get_performance_summary()
    positions = [p.__dict__ for p in await account.get_all_positions()]
    tx = [t.to_dict() for t in account.get_transactions()][-50:][::-1]
    tpl = env.get_template('paper.html')
    return HTMLResponse(tpl.render(
        summary=summary,
        positions=positions,
        transactions=tx,
        symbols=app_inst.symbols,
        reset=reset
    ))

@app.post('/paper/run', response_class=HTMLResponse)
async def run_auto_trading():
    app_inst = await get_paper_app()
    # Allocation fraction via env or default
    import os
    try:
        allocation_fraction = float(os.getenv('UI_AUTO_ALLOC_FRACTION', '0.05'))
        if allocation_fraction <= 0 or allocation_fraction > 1:
            raise ValueError
    except ValueError:
        allocation_fraction = 0.05
    await app_inst.run_scheduled_auto_trading(allocation_fraction=allocation_fraction, verbose=True)
    return RedirectResponse('/paper', status_code=303)

@app.post('/paper/refresh')
async def refresh_prices():
    app_inst = await get_paper_app()
    await app_inst.refresh_account_prices(verbose=True)
    return RedirectResponse('/paper', status_code=303)

@app.post('/paper/add_symbol')
async def add_symbol(new_symbol: str = Form(...)):
    app_inst = await get_paper_app()
    sym = new_symbol.strip().upper()
    if sym and sym not in app_inst.symbols:
        app_inst.symbols.append(sym)
        # Initialize strategy for new symbol
        try:
            strategy = app_inst.production_orchestrator.create_enhanced_strategy(
                symbol=sym,
                order_sizing_strategy="percentage",
                portfolio_pct=0.05,
                golden_cross_enabled=True,
                short_term_patterns_enabled=True,
                rsi_enabled=True,
                macd_enabled=True,
                bb_enabled=True,
                sma_crossover_enabled=True,
                ema_enabled=True,
                volume_analysis_enabled=True
            )
            app_inst.trading_strategies[sym] = strategy
            print(f"[UI] Added & initialized strategy for {sym}")
            # Kick off model training if none exists
            try:
                existing = app_inst.production_orchestrator.model_manager.list_models(sym)
                if not existing:
                    print(f"[UI] No existing model for {sym} - starting training")
                    # Trigger model training asynchronously and the algorithm should be based on config. Create a method in orchestrator
                    asyncio.create_task(
                        app_inst.production_orchestrator.train_ml_model(symbol=sym, algorithm='RandomForest', training_period_days=365)
                    )
                else:
                    print(f"[UI] Model already exists for {sym}, skipping training")
            except Exception as e:
                print(f"[UI] Model training trigger failed for {sym}: {e}")
        except Exception as e:
            print(f"[UI] Failed to init strategy for {sym}: {e}")
            app_inst.trading_strategies[sym] = None
    return RedirectResponse('/paper', status_code=303)

@app.post('/paper/reset')
async def reset_paper_account():
    app_inst = await get_paper_app()
    account = app_inst.account
    success = await account.reset_account_state()
    status = 'reset=success' if success else 'reset=failed'
    return RedirectResponse(f'/paper?{status}', status_code=303)

@app.get('/paper/api/summary')
async def api_summary():
    app_inst = await get_paper_app()
    return JSONResponse(app_inst.account.get_performance_summary())

@app.get('/paper/api/transactions')
async def api_transactions():
    app_inst = await get_paper_app()
    return JSONResponse([t.to_dict() for t in app_inst.account.get_transactions()][-200:][::-1])

# Graceful shutdown
@app.on_event('shutdown')
async def shutdown_event():
    global _paper_app
    if _paper_app and _paper_app.account:
        await _paper_app.disconnect_account()

# Entry point for uvicorn
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('src.ui.app:app', host='0.0.0.0', port=8000, reload=False)
