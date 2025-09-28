import datetime as dt  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
import pandas as pd 
import numpy as np 		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
import math
from marketsimcode import *
import matplotlib.pyplot as plt
import yfinance as yf
import yahoofinancials
from indicators import compute_indicators
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from config_manager import get_config


def get_stock_data(symbol,sd=dt.datetime(2018,1,1),ed=dt.datetime(2019,1,1)):
	stock_df = yf.download(symbol, start=sd, end=ed,progress=False)
	# Handle both single stock and multi-level column formats
	if isinstance(stock_df.columns, pd.MultiIndex):
		adj_close = stock_df[('Close', symbol)]  # yfinance now auto-adjusts by default
	else:
		adj_close = stock_df['Close']  # Use Close instead of Adj Close
	return adj_close.to_frame()

def get_indicator_data(data,symbol,window,train=True):
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
	
	indicator_data = indicator_data.values
	return indicator_data

def get_labels(prices, market_impact=None):
	config = get_config()
	if market_impact is None:
		market_impact = config.get_market_impact()
	
	labels = np.zeros(prices.shape[0]-3)
	for i in range(prices.shape[0]-3):
		ret = (prices.values[i+3] - prices.values[i])
		if ret / prices.values[i] < (-1*market_impact - 0.02):
			labels[i] = -1 # SELL
		elif ret / prices.values[i] > (market_impact + 0.02):
			labels[i] = 1 # BUY
	return labels

def create_trades_df(prices,symbol,Y_test):
	trades = pd.DataFrame(index = prices.index) 
	trades[symbol] = 0
	shares = 0
	config = get_config()
	shares_per_trade = config.get_shares_per_trade()
	
	for i in range(prices.shape[0]-3):
		if shares == 0 and Y_test[i] == 1:
			trades.iloc[i,0] = shares_per_trade
			shares = shares_per_trade
		elif shares > 0 and Y_test[i] == -1:
			trades.iloc[i,0] = -shares_per_trade
			shares = 0
	return trades



# Load configuration and get stocks and dates
config = get_config()
symbols = [config.get_default_symbol()]  # Use default symbol for simulation

# Get analysis dates from configuration
analysis_type = 'tesla_analysis'  # Can be made configurable
sd_train, ed_train, sd_test, ed_test = config.get_analysis_dates(analysis_type)

for symbol in symbols:
	stock_df_train = get_stock_data(symbol,sd_train,ed_train)
	stock_df_train.columns = [symbol]
	window_size = config.get_indicator_window()
	indicators_train = get_indicator_data(stock_df_train,symbol,window=window_size,train=True)
	labels_train = get_labels(stock_df_train)	

	stock_df_test = get_stock_data(symbol,sd_test,ed_test)
	stock_df_test.columns = [symbol]
	window_size = config.get_indicator_window()
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
	labels_test = clf.predict(indicators_test[:-3])
	labels_true = get_labels(stock_df_test)
	
	# Print technical indicators information
	print(f"\n=== TECHNICAL INDICATORS ANALYSIS FOR {symbol} ===")
	print(f"Training period: {sd_train.strftime('%Y-%m-%d')} to {ed_train.strftime('%Y-%m-%d')}")
	print(f"Testing period: {sd_test.strftime('%Y-%m-%d')} to {ed_test.strftime('%Y-%m-%d')}")
	
	# Get the full indicators dataframe for analysis
	indicators_full = compute_indicators(stock_df_test, symbol, window=window_size)
	indicators_full.fillna(0, inplace=True)
	
	print(f"\nAvailable Technical Indicators:")
	for col in indicators_full.columns:
		print(f"  - {col}")
	
	print(f"\nIndicators used by ML model (after dropping some):")
	indicators_used = compute_indicators(stock_df_test, symbol, window=5)
	indicators_used.fillna(0, inplace=True)
	indicators_used.drop('Upper_BB', axis=1, inplace=True)
	indicators_used.drop('Down_BB', axis=1, inplace=True)
	for col in indicators_used.columns:
		print(f"  - {col}")
	
	print(f"\nSample indicator values (first 10 days of test period):")
	print(indicators_used.head(10))
	
	print(f"\nIndicator Statistics for test period:")
	print(indicators_used.describe())
	
	print(f"\nLatest indicator values (last trading day):")
	latest_indicators = indicators_used.iloc[-1]
	for col, val in latest_indicators.items():
		print(f"  {col}: {val:.4f}")
	
	print(f"\nML Model Feature Importance (Decision Tree):")
	feature_names = indicators_used.columns
	importances = clf.feature_importances_
	for name, importance in zip(feature_names, importances):
		print(f"  {name}: {importance:.4f}")
	
	trades_df = create_trades_df(stock_df_test,symbol,labels_test)
	orders = trades2orders(trades_df,symbol)
	
	# Use portfolio configuration
	portfolio_config = config.get_portfolio_config()
	strategy_pval = compute_portvals(
		stock_df_test, orders, 
		start_val=portfolio_config['starting_value'],
		commission=portfolio_config['commission'],
		impact=portfolio_config['impact']
	)
	crb,adrb,sddrb,srb = compute_stats(strategy_pval)
	strategy_pval = strategy_pval / strategy_pval[0]

	print(labels_test)

	print(symbol)  
	print(orders) 		     			  		 			     			  	  		 	  	 		 			  		  			
	print("Cumulative Return: " + str(crb))
	print("Stdev of daily returns: " + str(sddrb))
	print("Average Daily Return: " + str(adrb))
	print("Sharpe Ratio: " + str(srb))	
	print()
	print()

	plt.figure(0)
	plt.plot(strategy_pval,'r',label='StrategyLearner')
	plt.grid()
	plt.legend()
	plt.title(symbol)
	plt.show()








