from Robinhood import Robinhood 
import datetime as dt  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
import pandas as pd 
import numpy as np 		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
from marketsimcode import *
import matplotlib.pyplot as plt
import yfinance as yf
import yahoofinancials
from indicators import compute_indicators
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from config_manager import get_config


def get_stock_data(symbol,sd=dt.datetime(2018,1,1),ed=dt.datetime(2019,1,1)):
	# Read historical stock data from Yahoo Finance
	stock_df = yf.download(symbol, start=sd, end=ed,progress=False)
	# Handle both single stock and multi-level column formats
	if isinstance(stock_df.columns, pd.MultiIndex):
		adj_close = stock_df[('Close', symbol)]  # yfinance now auto-adjusts by default
	else:
		adj_close = stock_df['Close']  # Use Close instead of Adj Close
	return adj_close.to_frame()

def get_indicator_data(data,symbol,window,train=True):
	# Compute indicators, which are features
	config = get_config()
	indicator_settings = config.get_indicator_settings()
	
	indicators = compute_indicators(data,symbol,window=window)
	indicators.fillna(0,inplace=True)
	indicator_data = indicators.copy()
	if train:
		indicator_data = indicator_data[:-3]

	# Drop indicators based on configuration
	if indicator_settings['drop_upper_bb']:
		indicator_data.drop('Upper_BB', axis=1, inplace=True)
	if indicator_settings['drop_lower_bb']:
		indicator_data.drop('Down_BB', axis=1, inplace=True)
	#indicator_data.drop('Volatility', axis=1, inplace=True)
	#indicator_data.drop('Momentum', axis=1, inplace=True)
	#indicator_data.drop('BB_val', axis=1, inplace=True)
	#indicator_data.drop('MACD', axis=1, inplace=True)
	indicator_data = indicator_data.values # to numpy array
	return indicator_data

def get_labels(prices, market_impact=None):
	# If return after 3 days is high enough, it is a BUY. If it is low, it is a SELL.
	config = get_config()
	if market_impact is None:
		market_impact = config.get_market_impact()
	
	labels = np.zeros(prices.shape[0]-3)
	# 0 label is DO NOTHING
	for i in range(prices.shape[0]-3):
		ret = (prices.values[i+3] - prices.values[i])
		if ret / prices.values[i] < (-1*market_impact - 0.02):
			labels[i] = -1 # SELL
		elif ret / prices.values[i] > (market_impact + 0.02):
			labels[i] = 1 # BUY
	return labels

def create_trades_df(prices,symbol,Y_test):
	# Based on labels, create trades dataframe
	trades = pd.DataFrame(index = prices.index) 
	trades[symbol] = 0
	shares = 0
	config = get_config()
	shares_per_trade = config.get_shares_per_trade()
	
	for i in range(prices.shape[0]-3):
		if shares == 0 and Y_test[i] == 1:
			trades.iloc[i,0] = shares_per_trade
			shares = shares_per_trade
		elif shares > 0 and Y_test[i] == 0:
			trades.iloc[i,0] = -shares_per_trade
			shares = 0
	return trades


# Load configuration
config = get_config()

# First, login to Robinhood with your credentials from configuration
# Assuming 2FA is ON
rh_config = config.get_robinhood_config()

# Only proceed with live trading if enabled
if rh_config['enable_live_trading']:
	my_trader = Robinhood()
	my_trader.login(
		username=rh_config['username'], 
		password=rh_config['password'], 
		qr_code=rh_config['qr_code']
	)
else:
	print("Live trading is disabled in configuration. Running in simulation mode.")
	my_trader = None

# Get stocks from configuration
symbols = config.get_symbols()

# Calculate training dates based on configuration
training_months = config.get_training_period_months()
lookback_days = config.get_lookback_days()

# Calculate dates for live trading
tod = dt.datetime.now()
d = dt.timedelta(days = lookback_days)  # Use configured lookback days
sd_test = tod - d
ed_test = dt.datetime.date(dt.datetime.now())
ed_train = sd_test

# Calculate training start date
sd_train = sd_test - dt.timedelta(days=training_months * 30)  # Approximate months to days

for symbol in symbols:
	stock_df_train = get_stock_data(symbol,sd_train,ed_train)
	stock_df_train.columns = [symbol]
	window_size = config.get_indicator_window()
	indicators_train = get_indicator_data(stock_df_train,symbol,window=window_size,train=True)
	labels_train = get_labels(stock_df_train)	

	stock_df_test = get_stock_data(symbol,sd_test,ed_test)
	stock_df_test.columns = [symbol]
	indicators_test = get_indicator_data(stock_df_test,symbol,window=window_size,train=False)

	# Create ML model based on configuration
	ml_config = config.get_ml_config()
	if ml_config['algorithm'] == 'RandomForest':
		clf = RandomForestClassifier(
			n_estimators=ml_config['n_estimators'],
			max_depth=ml_config['max_depth'],
			random_state=ml_config['random_state']
		)
	else:
		clf = DecisionTreeClassifier(
			max_depth=ml_config['max_depth'],
			random_state=ml_config['random_state']
		)
	clf.fit(indicators_train, labels_train)
	labels_test = clf.predict(indicators_test[-1].reshape(1,-1))
	labels_true = get_labels(stock_df_test)

	# Read the position file to get current shares
	# If using this first time, you need to manually create filename and add number of shares as .txt file
	f_name = config.get_position_filename(symbol)
	try:
		with open(f_name, 'r') as f:
			nums = f.readlines()
			nums = [int(i) for i in nums]
			shares = nums[0]
	except FileNotFoundError:
		# Create file with 0 shares if it doesn't exist
		shares = 0
		with open(f_name, 'w') as f:
			f.write(str(shares))

	# Get trading configuration
	shares_per_trade = config.get_shares_per_trade()

	# Strategy: if it is a BUY signal, buy configured shares. If it is a SELL, sell all shares.
	if shares == 0 and labels_test[-1] == 1:
		print(f'BUYING {shares_per_trade} {symbol} ...')
		if my_trader and rh_config['enable_live_trading']:
			stock_instrument = my_trader.instruments(symbol)[0]
			# Uncomment below for actual trading
			#buy_order = my_trader.place_market_buy_order(stock_instrument['url'], symbol, rh_config['time_in_force'], shares_per_trade)
		shares = shares_per_trade
		with open(f_name,'w') as f:
			f.write(str(shares))

	elif shares > 0 and labels_test[-1] == -1:
		print(f'SELLING ALL {symbol} ({shares} shares)...')
		if my_trader and rh_config['enable_live_trading']:
			stock_instrument = my_trader.instruments(symbol)[0]
			# Uncomment below for actual trading
			#sell_order = my_trader.place_market_sell_order(stock_instrument['url'], symbol, rh_config['time_in_force'], shares)
		shares = 0
		with open(f_name,'w') as f:
			f.write(str(shares))

	else: 
		print('DOING NOTHING')




