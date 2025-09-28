import numpy as np 
import pandas as pd 
import util 
import datetime as dt
import matplotlib.pyplot as plt

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
	Detect golden cross pattern - when shorter SMA crosses above longer SMA
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
	else:
		price_series = prices
	
	# Calculate SMAs
	sma_short = price_series.rolling(window=sma_short_window).mean()
	sma_long = price_series.rolling(window=sma_long_window).mean()
	
	# Detect crossovers
	golden_cross_signals = detect_sma_crossovers(sma_short, sma_long)
	
	# Calculate pattern strength
	pattern_strength = calculate_golden_cross_strength(price_series, sma_short, sma_long)
	
	# Get current status
	current_status = get_current_cross_status(sma_short, sma_long, lookback_days)
	
	return {
		'sma_short': sma_short,
		'sma_long': sma_long,
		'golden_cross_signals': golden_cross_signals,
		'death_cross_signals': golden_cross_signals * -1,  # Opposite signals
		'pattern_strength': pattern_strength,
		'current_status': current_status,
		'latest_golden_cross': get_latest_cross_date(golden_cross_signals, cross_type='golden'),
		'latest_death_cross': get_latest_cross_date(golden_cross_signals, cross_type='death')
	}

def detect_sma_crossovers(sma_short, sma_long):
	"""
	Detect when short SMA crosses above (golden) or below (death) long SMA
	Returns 1 for golden cross, -1 for death cross, 0 otherwise
	"""
	# Calculate the difference between SMAs
	sma_diff = sma_short - sma_long
	
	# Detect sign changes (crossovers)
	sign_changes = np.sign(sma_diff).diff()
	
	# Golden cross: short SMA crosses above long SMA (sign change from negative to positive)
	golden_cross = (sign_changes > 0).astype(int)
	
	# Death cross: short SMA crosses below long SMA (sign change from positive to negative)
	death_cross = (sign_changes < 0).astype(int) * -1
	
	# Combine signals
	cross_signals = golden_cross + death_cross
	
	return cross_signals

def calculate_golden_cross_strength(prices, sma_short, sma_long):
	"""
	Calculate the strength of golden cross patterns based on:
	- Volume confirmation (if available)
	- Price momentum
	- SMA separation distance
	- Trend consistency
	"""
	strength_scores = pd.Series(0.0, index=prices.index)
	
	# Calculate SMA separation (normalized)
	sma_separation = (sma_short - sma_long) / sma_long
	
	# Calculate price momentum (20-day)
	price_momentum = prices.pct_change(20)
	
	# Calculate trend consistency (how long short > long)
	trend_consistency = calculate_trend_consistency(sma_short, sma_long)
	
	# Combine factors for strength score
	strength_scores = (
		0.4 * normalize_indicator(sma_separation) +
		0.3 * normalize_indicator(price_momentum) +
		0.3 * normalize_indicator(trend_consistency)
	)
	
	return strength_scores

def calculate_trend_consistency(sma_short, sma_long, window=20):
	"""
	Calculate how consistently the short SMA has been above the long SMA
	"""
	above_long = (sma_short > sma_long).astype(int)
	consistency = above_long.rolling(window=window).mean()
	return consistency

def normalize_indicator(series, method='minmax'):
	"""
	Normalize indicator values to 0-1 range
	"""
	if method == 'minmax':
		return (series - series.min()) / (series.max() - series.min())
	elif method == 'zscore':
		return (series - series.mean()) / series.std()
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
	Generate trading signals based on golden cross patterns
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