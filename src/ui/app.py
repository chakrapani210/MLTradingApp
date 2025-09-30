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

# --- Setup FastAPI and Jinja2 ---
app = FastAPI(title="Trading UI")

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
if not os.path.exists(TEMPLATE_DIR):
    os.makedirs(TEMPLATE_DIR, exist_ok=True)

env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(['html', 'xml'])
)

# Global (singleton) paper account instance reused across requests
_paper_account: Optional[PaperTradingAccount] = None

async def get_paper_account() -> PaperTradingAccount:
    global _paper_account
    if _paper_account is None:
        _paper_account = PaperTradingAccount(PaperTradingConfig())
        await _paper_account.connect()
    return _paper_account

STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
if os.path.exists(STATIC_DIR):
    app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')

# --- Routes ---
@app.get('/', response_class=HTMLResponse)
async def index():
    tpl = env.get_template('index.html')
    return HTMLResponse(tpl.render())

@app.get('/paper', response_class=HTMLResponse)
async def paper_view(tickers: str = 'TSLA'):
    account = await get_paper_account()
    summary = account.get_performance_summary()
    positions = [p.__dict__ for p in await account.get_all_positions()]
    tx = [t.to_dict() for t in account.get_transactions()][-50:][::-1]
    tpl = env.get_template('paper.html')
    return HTMLResponse(tpl.render(tickers=tickers, summary=summary, positions=positions, transactions=tx))

@app.post('/paper/run', response_class=HTMLResponse)
async def run_trade(tickers: str = Form(...)):
    account = await get_paper_account()
    symbols = [s.strip().upper() for s in tickers.split(',') if s.strip()]
    # For MVP: trade first symbol only
    if not symbols:
        return RedirectResponse('/paper', status_code=303)
    symbol = symbols[0]
    # Simple rule: if no position -> buy 1 share, if have position -> sell 1 share
    pos = await account.get_position(symbol)
    qty = 1
    side = OrderSide.BUY if (not pos or pos.quantity <= 0) else OrderSide.SELL
    order = Order(symbol=symbol, quantity=qty, order_type=OrderType.MARKET, side=side)
    await account.place_order(order)
    return RedirectResponse(f'/paper?tickers={symbol}', status_code=303)

@app.get('/paper/api/summary')
async def api_summary():
    account = await get_paper_account()
    return JSONResponse(account.get_performance_summary())

@app.get('/paper/api/transactions')
async def api_transactions():
    account = await get_paper_account()
    return JSONResponse([t.to_dict() for t in account.get_transactions()][-200:][::-1])

# Graceful shutdown
@app.on_event('shutdown')
async def shutdown_event():
    global _paper_account
    if _paper_account:
        await _paper_account.disconnect()

# Entry point for uvicorn
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('src.ui.app:app', host='0.0.0.0', port=8000, reload=False)
