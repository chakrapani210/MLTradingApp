# ML Automated Trading Application

> Enhanced Machine Learning based Automated Trading System

This is an enhanced version of the original ML Automated Trading system with improved algorithms, optimized parameters, and comprehensive performance analysis. The system uses technical indicators and machine learning to generate trading signals.

## 🚀 Recent Enhancements

- Automated daily scheduled trading script (2:30 PM US Central) using pre-trained models

## Scheduled Daily Trading (2:30 PM CST)
To enable an automatic run that fetches market data and places orders based on model predictions:

1. Review / modify `scripts/scheduled_trading_task.py` for sizing logic.
2. Create the Windows Scheduled Task (machine time should align with Central Time or adjust trigger time):
	```powershell
	powershell -ExecutionPolicy Bypass -File scripts/create_windows_task.ps1 -TaskName MLTrading_Scheduled_1430CST -Time 14:30
	```
3. Task runs `python scripts/scheduled_trading_task.py` daily. Logs appear in console history; redirect output if desired:
	- Edit the PowerShell script and append `> logs\scheduled_run.txt 2>&1` to the argument for persistent logging.

Override run manually:
```powershell
python scripts/scheduled_trading_task.py
```

Note: The script currently places BUY orders only on positive signals; SELL / position reduction logic can be added later.

## 📊 Performance Highlights

### Tesla (TSLA) - July-December 2025
- **Cumulative Return**: +94.07% (6 months)
- **Sharpe Ratio**: 3.68 (Exceptional)
- **Strategy**: ML-driven buy and hold
- **Starting Portfolio**: $3,000 → $5,822

Note that everything in this repo is for educational purposes. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.

## Installation

OS X & Linux:

First, download the unofficial Robinhood API that lets you login and place market orders. See github repo for more information.

```sh
git clone https://github.com/LichAmnesia/Robinhood
cd Robinhood
sudo python3 setup.py install
```
Then, use pip to install the required Python libraries below:
datetime, pandas, numpy, matplotlib,yfinance, yahoofinancials, sklearn

## Robinhood Login 
Robinhood is now requiring a mandatory MFA.

Going to your Robinhood Web App and turning on 2FA is highly recommended because without it your auth tokens will expire every 24 hours. To do this, go to settings, turn on 2FA, select "Authentication App", click "Can't Scan It?", and save the 16-character QR code.

For now, only Python 3 code will work with this.

Use something like this to login:
```
QR = "1234567899qwertd"
my_trader = Robinhood()
my_trader.login(username="username", password="p@ssw0rd", qr_code=QR)
```

## Market Simulation
To make sure your classifier works, run the following: 
```
python3 simulate_results.py
```
You can change the number of shares to buy/sell, ML algorithm, and starting value of your portfolio. This script simulates the market from previos years with your trading decisions and shows the performance of your portfolio. 

## Automated Trading
After simulating your trading strategy, run the following Python script: 
```
python3 automated_trading.py
```
Make sure you login to Robinhood correctly. You can use crontab to execute the script daily. You should uncomment the market buy and sell orders in the script to make sure orders get through. Those lines are commented for your sake.

One important thing with this is the .txt files. You need to manually create txt files for each stock you own or want to own. Just enter the number of shares you currently have or 0. Examples are provided in the repo. This helps with automating, as the files get updated each time you buy or sell any shares. This would make sure Robinhood will not try to sell any shares you do not have. 

## Example Simulation
The performance of our classifier with TQQQ stock is shown below. Note that the portfolio values are normalized.
![](TQQQ.png)
