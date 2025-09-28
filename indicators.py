import numpy as np 
import pandas as pd 
import util 
import datetime as dt
import matplotlib.pyplot as plt
import talib

def compute_indicators(prices_all,syms=['JPM'],window=20):
	prices = prices_all

	# Compute SMA 
	SMA = prices.rolling(window=window).mean()
	price_SMA = prices / SMA

	# Compute Boellinger Bands 
	std = prices.rolling(window=window).std()
	upper_bb = SMA + (2 * std)
	down_bb  = SMA - (2 * std)
	bb_val = (prices - SMA)/(2 * std)

	# Compute Momentum
	Momentum = (prices/prices.shift(window)) - 1

	# Compute Volatility 
	Volatility = prices.rolling(window=window).std()

	# Compute MACD
	EMA_26 = prices.ewm(span=26).mean()
	EMA_12 = prices.ewm(span=12).mean()
	MACD = EMA_12 - EMA_26

	indicators = pd.DataFrame(index=prices.index)

	indicators['price_SMA'] = price_SMA
	indicators['Upper_BB'] = upper_bb
	indicators['Down_BB'] = down_bb
	indicators['BB_val'] = bb_val
	indicators['Momentum'] = Momentum
	indicators['Volatility'] = Volatility
	indicators['MACD'] = MACD
	#indicators = indicators.dropna()

	return indicators

def detect_golden_cross(prices, sma_short_window=50, sma_long_window=200, lookback_days=5):
	"""
	Detect golden cross pattern using TA-Lib - when shorter SMA crosses above longer SMA
	Args:
		prices: Price data (pandas Series or DataFrame)
		sma_short_window: Short-term SMA period (default 50)
		sma_long_window: Long-term SMA period (default 200) 
		lookback_days: Days to look back for crossover confirmation (default 5)
	Returns:
		dict: Golden cross analysis including signals and strength
	"""
	if isinstance(prices, pd.DataFrame):
		price_series = prices.iloc[:, 0]  # Use first column if DataFrame
		close_prices = price_series.values
	else:
		price_series = prices
		close_prices = prices.values
	
	# Calculate SMAs using TA-Lib (professional library)
	sma_short = talib.SMA(close_prices, timeperiod=sma_short_window)
	sma_long = talib.SMA(close_prices, timeperiod=sma_long_window)
	
	# Convert back to pandas Series with proper index
	sma_short_series = pd.Series(sma_short, index=price_series.index)
	sma_long_series = pd.Series(sma_long, index=price_series.index)
	
	# Detect crossovers using TA-Lib calculated SMAs
	golden_cross_signals = detect_sma_crossovers_talib(sma_short_series, sma_long_series)
	
	# Calculate pattern strength
	pattern_strength = calculate_golden_cross_strength_talib(price_series, sma_short_series, sma_long_series)
	
	# Get current status
	current_status = get_current_cross_status(sma_short_series, sma_long_series, lookback_days)
	
	return {
		'sma_short': sma_short_series,
		'sma_long': sma_long_series,
		'golden_cross_signals': golden_cross_signals,
		'death_cross_signals': golden_cross_signals * -1,  # Opposite signals
		'pattern_strength': pattern_strength,
		'current_status': current_status,
		'latest_golden_cross': get_latest_cross_date(golden_cross_signals, cross_type='golden'),
		'latest_death_cross': get_latest_cross_date(golden_cross_signals, cross_type='death')
	}

def detect_sma_crossovers_talib(sma_short, sma_long):
	"""
	Detect when short SMA crosses above (golden) or below (death) long SMA
	Returns 1 for golden cross, -1 for death cross, 0 otherwise
	Uses TA-Lib calculated moving averages for professional accuracy
	"""
	# Initialize signal series
	cross_signals = pd.Series(0, index=sma_short.index)
	
	# Find valid data points (non-NaN)
	valid_mask = ~(pd.isna(sma_short) | pd.isna(sma_long))
	valid_indices = sma_short.index[valid_mask]
	
	if len(valid_indices) < 2:
		return cross_signals
	
	# Detect crossovers by checking consecutive valid points
	for i in range(1, len(valid_indices)):
		curr_idx = valid_indices[i]
		prev_idx = valid_indices[i-1]
		
		curr_short = sma_short[curr_idx]
		curr_long = sma_long[curr_idx]
		prev_short = sma_short[prev_idx]
		prev_long = sma_long[prev_idx]
		
		# Golden cross: short MA crosses above long MA
		if prev_short <= prev_long and curr_short > curr_long:
			cross_signals[curr_idx] = 1
		
		# Death cross: short MA crosses below long MA
		elif prev_short >= prev_long and curr_short < curr_long:
			cross_signals[curr_idx] = -1
	
	return cross_signals

def calculate_golden_cross_strength_talib(prices, sma_short, sma_long):
	"""
	Calculate the strength of golden cross patterns using TA-Lib indicators:
	- SMA separation distance
	- Price momentum using TA-Lib ROC (Rate of Change)
	- Trend consistency
	"""
	strength_scores = pd.Series(0.0, index=prices.index)
	
	# Calculate SMA separation (normalized)
	sma_separation = (sma_short - sma_long) / sma_long
	sma_separation = sma_separation.fillna(0)
	
	# Calculate price momentum using TA-Lib ROC (Rate of Change)
	try:
		roc_values = talib.ROC(prices.values, timeperiod=20)
		price_momentum = pd.Series(roc_values, index=prices.index)
	except:
		# Fallback to pandas calculation if TA-Lib fails
		price_momentum = prices.pct_change(20)
	
	price_momentum = price_momentum.fillna(0)
	
	# Calculate trend consistency (how long short > long)
	trend_consistency = calculate_trend_consistency(sma_short, sma_long)
	
	# Combine factors for strength score with proper handling of NaN values
	try:
		norm_separation = normalize_indicator(sma_separation.dropna())
		norm_momentum = normalize_indicator(price_momentum.dropna())
		norm_consistency = normalize_indicator(trend_consistency.dropna())
		
		# Reindex to original index, filling with 0
		norm_separation = norm_separation.reindex(prices.index, fill_value=0)
		norm_momentum = norm_momentum.reindex(prices.index, fill_value=0)
		norm_consistency = norm_consistency.reindex(prices.index, fill_value=0)
		
		strength_scores = (
			0.4 * norm_separation +
			0.3 * norm_momentum +
			0.3 * norm_consistency
		)
	except:
		# If normalization fails, return zeros
		strength_scores = pd.Series(0.0, index=prices.index)
	
	return strength_scores.fillna(0)

def calculate_trend_consistency(sma_short, sma_long, window=20):
	"""
	Calculate how consistently the short SMA has been above the long SMA
	"""
	above_long = (sma_short > sma_long).astype(int)
	consistency = above_long.rolling(window=window).mean()
	return consistency

def normalize_indicator(series, method='minmax'):
	"""
	Normalize indicator values to 0-1 range with better error handling
	"""
	if len(series) == 0 or series.isna().all():
		return series
	
	series_clean = series.dropna()
	if len(series_clean) == 0:
		return pd.Series(0.0, index=series.index)
	
	if method == 'minmax':
		min_val = series_clean.min()
		max_val = series_clean.max()
		if max_val == min_val:
			return pd.Series(0.5, index=series.index)  # Return middle value if no variation
		return (series - min_val) / (max_val - min_val)
	elif method == 'zscore':
		mean_val = series_clean.mean()
		std_val = series_clean.std()
		if std_val == 0:
			return pd.Series(0.0, index=series.index)  # Return zeros if no variation
		return (series - mean_val) / std_val
	else:
		return series

def get_current_cross_status(sma_short, sma_long, lookback_days=5):
	"""
	Determine current golden/death cross status
	"""
	if len(sma_short) < lookback_days or len(sma_long) < lookback_days:
		return 'insufficient_data'
	
	# Check recent crossovers
	recent_short = sma_short.iloc[-lookback_days:]
	recent_long = sma_long.iloc[-lookback_days:]
	
	current_short = sma_short.iloc[-1]
	current_long = sma_long.iloc[-1]
	
	if current_short > current_long:
		# Check if this is a recent golden cross
		if any(recent_short.iloc[:-1] <= recent_long.iloc[:-1]):
			return 'recent_golden_cross'
		else:
			return 'established_golden_trend'
	else:
		# Check if this is a recent death cross
		if any(recent_short.iloc[:-1] >= recent_long.iloc[:-1]):
			return 'recent_death_cross'
		else:
			return 'established_death_trend'

def get_latest_cross_date(cross_signals, cross_type='golden'):
	"""
	Get the date of the most recent golden or death cross
	"""
	if cross_type == 'golden':
		cross_dates = cross_signals[cross_signals > 0].index
	elif cross_type == 'death':
		cross_dates = cross_signals[cross_signals < 0].index
	else:
		return None
	
	if len(cross_dates) > 0:
		return cross_dates[-1]
	else:
		return None

def generate_golden_cross_signals(prices, sma_short_window=50, sma_long_window=200):
	"""
	Generate trading signals based on golden cross patterns using TA-Lib
	Returns: DataFrame with buy/sell signals and confidence scores
	"""
	gc_analysis = detect_golden_cross(prices, sma_short_window, sma_long_window)
	
	signals_df = pd.DataFrame(index=prices.index)
	signals_df['sma_short'] = gc_analysis['sma_short']
	signals_df['sma_long'] = gc_analysis['sma_long']
	signals_df['golden_cross'] = gc_analysis['golden_cross_signals']
	signals_df['death_cross'] = gc_analysis['death_cross_signals']
	signals_df['pattern_strength'] = gc_analysis['pattern_strength']
	
	# Generate trading signals
	signals_df['buy_signal'] = (gc_analysis['golden_cross_signals'] > 0).astype(int)
	signals_df['sell_signal'] = (gc_analysis['golden_cross_signals'] < 0).astype(int)
	
	# Add confidence based on pattern strength
	signals_df['buy_confidence'] = signals_df['buy_signal'] * gc_analysis['pattern_strength']
	signals_df['sell_confidence'] = signals_df['sell_signal'] * gc_analysis['pattern_strength']
	
	# Fill NaN values with 0
	signals_df = signals_df.fillna(0)
	
	return signals_df

def detect_short_term_patterns(prices, high=None, low=None, volume=None, rsi_period=14, bb_period=20):
	"""
	Detect multiple short-term trading patterns using TA-Lib
	Args:
		prices: Close price data (pandas Series or DataFrame)
		high: High price data (optional, for candlestick patterns)
		low: Low price data (optional, for candlestick patterns)
		volume: Volume data (optional, for volume indicators)
		rsi_period: RSI calculation period (default 14)
		bb_period: Bollinger Bands period (default 20)
	Returns:
		dict: Comprehensive short-term pattern analysis
	"""
	if isinstance(prices, pd.DataFrame):
		price_series = prices.iloc[:, 0]  # Use first column if DataFrame
		close_prices = price_series.values.astype(np.float64)
	else:
		price_series = prices
		close_prices = prices.values.astype(np.float64)
	
	results = {}
	
	# 1. RSI Analysis (Momentum)
	rsi_values = talib.RSI(close_prices, timeperiod=rsi_period)
	rsi_series = pd.Series(rsi_values, index=price_series.index)
	results['rsi'] = rsi_series
	results['rsi_overbought'] = rsi_series > 70
	results['rsi_oversold'] = rsi_series < 30
	results['rsi_signals'] = generate_rsi_signals(rsi_series)
	
	# 2. Bollinger Bands (Volatility + Mean Reversion)
	upper_bb, middle_bb, lower_bb = talib.BBANDS(close_prices, timeperiod=bb_period)
	results['bb_upper'] = pd.Series(upper_bb, index=price_series.index)
	results['bb_middle'] = pd.Series(middle_bb, index=price_series.index)
	results['bb_lower'] = pd.Series(lower_bb, index=price_series.index)
	results['bb_signals'] = generate_bb_signals(price_series, results['bb_upper'], results['bb_lower'])
	
	# 3. Stochastic Oscillator (Momentum)
	if high is not None and low is not None:
		if isinstance(high, pd.Series):
			high_prices = high.values.astype(np.float64)
		elif isinstance(high, pd.DataFrame):
			high_prices = high.iloc[:, 0].values.astype(np.float64)
		else:
			high_prices = np.array(high).astype(np.float64)
			
		if isinstance(low, pd.Series):
			low_prices = low.values.astype(np.float64)
		elif isinstance(low, pd.DataFrame):
			low_prices = low.iloc[:, 0].values.astype(np.float64)
		else:
			low_prices = np.array(low).astype(np.float64)
			
		stoch_k, stoch_d = talib.STOCH(high_prices, low_prices, close_prices)
		results['stoch_k'] = pd.Series(stoch_k, index=price_series.index)
		results['stoch_d'] = pd.Series(stoch_d, index=price_series.index)
		results['stoch_signals'] = generate_stoch_signals(results['stoch_k'], results['stoch_d'])
	
	# 4. Volume Analysis (if available)
	if volume is not None:
		if isinstance(volume, pd.Series):
			volume_data = volume.values.astype(np.float64)
		elif isinstance(volume, pd.DataFrame):
			volume_data = volume.iloc[:, 0].values.astype(np.float64)
		else:
			volume_data = np.array(volume).astype(np.float64)
			
		obv_values = talib.OBV(close_prices, volume_data)
		results['obv'] = pd.Series(obv_values, index=price_series.index)
		mfi_values = talib.MFI(high_prices, low_prices, close_prices, volume_data) if high is not None and low is not None else None
		if mfi_values is not None:
			results['mfi'] = pd.Series(mfi_values, index=price_series.index)
	
	# 5. Candlestick Patterns (if OHLC available)
	if high is not None and low is not None:
		open_prices = price_series.shift(1).values.astype(np.float64)  # Approximate open as previous close
		results['candlestick_patterns'] = detect_candlestick_patterns(
			open_prices, high_prices, low_prices, close_prices, price_series.index
		)
	
	return results

def generate_rsi_signals(rsi_series):
	"""
	Generate trading signals from RSI
	Returns 1 for buy (oversold recovery), -1 for sell (overbought decline), 0 otherwise
	"""
	signals = pd.Series(0, index=rsi_series.index)
	
	# RSI signals with confirmation
	for i in range(1, len(rsi_series)):
		curr_rsi = rsi_series.iloc[i]
		prev_rsi = rsi_series.iloc[i-1]
		
		# Buy signal: RSI was oversold and now recovering above 30
		if prev_rsi < 30 and curr_rsi >= 30:
			signals.iloc[i] = 1
		# Sell signal: RSI was overbought and now declining below 70
		elif prev_rsi > 70 and curr_rsi <= 70:
			signals.iloc[i] = -1
	
	return signals

def generate_bb_signals(prices, bb_upper, bb_lower):
	"""
	Generate trading signals from Bollinger Bands
	Returns 1 for buy (bounce from lower band), -1 for sell (rejection at upper band)
	"""
	signals = pd.Series(0, index=prices.index)
	
	for i in range(1, len(prices)):
		curr_price = prices.iloc[i]
		prev_price = prices.iloc[i-1]
		curr_lower = bb_lower.iloc[i]
		curr_upper = bb_upper.iloc[i]
		
		if pd.notna(curr_lower) and pd.notna(curr_upper):
			# Buy signal: Price bounces off lower Bollinger Band
			if prev_price <= bb_lower.iloc[i-1] and curr_price > curr_lower:
				signals.iloc[i] = 1
			# Sell signal: Price rejected at upper Bollinger Band
			elif prev_price >= bb_upper.iloc[i-1] and curr_price < curr_upper:
				signals.iloc[i] = -1
	
	return signals

def generate_stoch_signals(stoch_k, stoch_d):
	"""
	Generate trading signals from Stochastic Oscillator
	Returns 1 for buy (%K crosses above %D in oversold), -1 for sell (%K crosses below %D in overbought)
	"""
	signals = pd.Series(0, index=stoch_k.index)
	
	for i in range(1, len(stoch_k)):
		curr_k = stoch_k.iloc[i]
		curr_d = stoch_d.iloc[i]
		prev_k = stoch_k.iloc[i-1]
		prev_d = stoch_d.iloc[i-1]
		
		if pd.notna(curr_k) and pd.notna(curr_d) and pd.notna(prev_k) and pd.notna(prev_d):
			# Buy signal: %K crosses above %D in oversold region (below 20)
			if prev_k <= prev_d and curr_k > curr_d and curr_k < 20:
				signals.iloc[i] = 1
			# Sell signal: %K crosses below %D in overbought region (above 80)
			elif prev_k >= prev_d and curr_k < curr_d and curr_k > 80:
				signals.iloc[i] = -1
	
	return signals

def detect_candlestick_patterns(open_prices, high_prices, low_prices, close_prices, index):
	"""
	Detect key candlestick patterns using TA-Lib
	Returns dict with pattern signals
	"""
	patterns = {}
	
	try:
		# Key reversal patterns
		patterns['doji'] = pd.Series(talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices), index=index)
		patterns['hammer'] = pd.Series(talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices), index=index)
		patterns['hanging_man'] = pd.Series(talib.CDLHANGINGMAN(open_prices, high_prices, low_prices, close_prices), index=index)
		patterns['engulfing'] = pd.Series(talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices), index=index)
		patterns['morning_star'] = pd.Series(talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices), index=index)
		patterns['evening_star'] = pd.Series(talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices), index=index)
		patterns['shooting_star'] = pd.Series(talib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices), index=index)
		
		# Combine all bullish patterns
		bullish_patterns = patterns['hammer'] + patterns['morning_star'] + (patterns['engulfing'] > 0).astype(int)
		patterns['bullish_signal'] = (bullish_patterns > 0).astype(int)
		
		# Combine all bearish patterns
		bearish_patterns = patterns['hanging_man'] + patterns['evening_star'] + patterns['shooting_star'] + (patterns['engulfing'] < 0).astype(int)
		patterns['bearish_signal'] = (bearish_patterns > 0).astype(int)
		
	except Exception as e:
		print(f"Warning: Candlestick pattern detection error: {e}")
		for pattern_name in ['doji', 'hammer', 'hanging_man', 'engulfing', 'morning_star', 'evening_star', 'shooting_star', 'bullish_signal', 'bearish_signal']:
			patterns[pattern_name] = pd.Series(0, index=index)
	
	return patterns

def generate_combined_short_term_signals(short_term_analysis):
	"""
	Generate combined short-term trading signals from multiple indicators
	Returns comprehensive buy/sell signals with confidence scores
	"""
	signals_df = pd.DataFrame(index=short_term_analysis['rsi'].index)
	
	# Individual signals
	signals_df['rsi_signal'] = short_term_analysis.get('rsi_signals', pd.Series(0, index=signals_df.index))
	signals_df['bb_signal'] = short_term_analysis.get('bb_signals', pd.Series(0, index=signals_df.index))
	signals_df['stoch_signal'] = short_term_analysis.get('stoch_signals', pd.Series(0, index=signals_df.index))
	
	# Candlestick signals if available
	if 'candlestick_patterns' in short_term_analysis:
		patterns = short_term_analysis['candlestick_patterns']
		signals_df['bullish_candle'] = patterns.get('bullish_signal', pd.Series(0, index=signals_df.index))
		signals_df['bearish_candle'] = patterns.get('bearish_signal', pd.Series(0, index=signals_df.index))
	else:
		signals_df['bullish_candle'] = pd.Series(0, index=signals_df.index)
		signals_df['bearish_candle'] = pd.Series(0, index=signals_df.index)
	
	# Combined signals with confidence scoring
	buy_signals = (signals_df['rsi_signal'] > 0).astype(int) + \
	             (signals_df['bb_signal'] > 0).astype(int) + \
	             (signals_df['stoch_signal'] > 0).astype(int) + \
	             signals_df['bullish_candle']
	
	sell_signals = (signals_df['rsi_signal'] < 0).astype(int) + \
	              (signals_df['bb_signal'] < 0).astype(int) + \
	              (signals_df['stoch_signal'] < 0).astype(int) + \
	              signals_df['bearish_candle']
	
	# Final signals with confidence thresholds
	signals_df['buy_signal'] = (buy_signals >= 2).astype(int)  # Require at least 2 confirming signals
	signals_df['sell_signal'] = (sell_signals >= 2).astype(int)
	signals_df['buy_confidence'] = buy_signals / 4  # Normalize to 0-1
	signals_df['sell_confidence'] = sell_signals / 4
	
	return signals_df

def author():
	return 'htekgul3'


def test_code():
	start_date = dt.datetime(2008,1,1)
	end_date = dt.datetime(2009,12,31)
	syms = ['JPM']
	dates = pd.date_range(start_date,end_date)
	prices_all = util.get_data(syms, dates)

	plt.figure(0)
	indicators = compute_indicators(prices_all,syms)
	indicators['prices'].plot(label='JPM')
	indicators['SMA'].plot(label='SMA')
	plt.title('Simple Moving Average (SMA)')
	plt.ylabel('Normalized Price')
	plt.legend()
	plt.grid()
	plt.savefig('SMA.png')

	plt.figure(1)
	indicators['Upper_BB'].plot(label='Upper BB')
	indicators['Down_BB'].plot(label='Down BB')
	indicators['prices'].plot(label='JPM')
	indicators['SMA'].plot(label='SMA')
	plt.title('Boellinger Bands')
	plt.legend()
	plt.grid()
	plt.savefig('BB.png')

	plt.figure(2)
	indicators['BB_val'].plot(label='BB Value')
	plt.title('BB Value')
	plt.ylabel('Normalized Price')
	plt.legend()
	plt.grid()
	plt.savefig('BB_Val.png')

	plt.figure(3)
	indicators['Momentum'].plot(label='Momentum')
	indicators['prices'].plot(label='prices')
	plt.title('Momentum')
	plt.legend()
	plt.grid()
	plt.savefig('momentum.png')

	plt.figure(4)
	indicators['Volatility'].plot(label='Volatility')
	indicators['prices'].plot(label='JPM')
	plt.title('Volatility')
	plt.legend()
	plt.grid()
	plt.savefig('volatility.png')
	
	plt.figure(5)
	indicators['MACD'].plot(label='MACD')
	indicators['prices'].plot(label='JPM')
	plt.title('Moving Average Convergence Divergence')
	plt.legend()
	plt.grid()
	plt.savefig('MACD.png')
	return

if __name__ == "__main__":
	test_code()