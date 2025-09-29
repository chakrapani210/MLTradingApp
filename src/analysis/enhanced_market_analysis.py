"""
Enhanced Market Analysis Implementation
Implements comprehensive market context analysis, correlation studies, and enhanced feature engineering
"""

import datetime as dt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings
from dataclasses import dataclass
from scipy import stats
from sklearn.preprocessing import StandardScaler

from ..interfaces.data_provider import DataProvider
from ..data.providers import YFinanceProvider
import sys
import os

# Add project root to path for config_manager import
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config_manager import get_config_manager

warnings.filterwarnings('ignore')

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("[WARNING] TA-Lib not available. Using simplified technical indicators.")


@dataclass
class MarketContext:
    """Market context information"""
    spy_correlation: float
    qqq_correlation: float
    spy_beta: float
    qqq_beta: float
    market_regime: str  # 'bull', 'bear', 'sideways'
    volatility_regime: str  # 'low', 'normal', 'high'
    sector_strength: Dict[str, float]
    market_indicators: Dict[str, float]
    analysis_period: str


@dataclass
class EnhancedFeatures:
    """Enhanced features for ML training"""
    features: np.ndarray
    feature_names: List[str]
    target_labels: np.ndarray
    metadata: Dict[str, Any]


class MarketContextAnalyzer:
    """
    Comprehensive Market Context Analysis
    
    Analyzes:
    - Market correlations (SPY, QQQ, sector ETFs)
    - Beta calculations
    - Market regime detection
    - Volatility analysis
    - Sector rotation patterns
    """
    
    def __init__(self, data_provider: DataProvider):
        """Initialize Market Context Analyzer"""
        self.data_provider = data_provider
        self.market_etfs = ['SPY', 'QQQ', 'IWM', '^VIX']  # Core market indicators
        self.sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV', 
            'Financial': 'XLF',
            'Consumer Discretionary': 'XLY',
            'Consumer Staples': 'XLP',
            'Energy': 'XLE',
            'Utilities': 'XLU',
            'Industrials': 'XLI',
            'Materials': 'XLB',
            'Real Estate': 'XLRE',
            'Communication': 'XLC'
        }
        
        print(f"[MARKET_ANALYZER] Initialized with {len(self.sector_etfs)} sector ETFs")
    
    def analyze_market_context(self, symbol: str, start_date: dt.datetime, 
                             end_date: dt.datetime, window: int = 60) -> MarketContext:
        """
        Analyze comprehensive market context for a symbol
        
        Args:
            symbol: Target symbol for analysis
            start_date: Analysis start date
            end_date: Analysis end date
            window: Rolling window for calculations
            
        Returns:
            MarketContext with comprehensive analysis
        """
        print(f"[MARKET_CONTEXT] Analyzing market context for {symbol}")
        print(f"                 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Get market data
        symbols_to_fetch = [symbol] + self.market_etfs + list(self.sector_etfs.values())
        market_data = self.data_provider.get_market_data(
            symbols=symbols_to_fetch,
            start_date=start_date,
            end_date=end_date
        )
        
        if symbol not in market_data:
            raise ValueError(f"No data available for {symbol}")
        
        symbol_data = market_data[symbol]
        spy_data = market_data.get('SPY')
        qqq_data = market_data.get('QQQ')
        
        # Calculate correlations
        correlations = self._calculate_correlations(symbol_data, market_data, window)
        
        # Calculate betas
        spy_beta = self._calculate_beta(symbol_data, spy_data) if spy_data is not None else 0.0
        qqq_beta = self._calculate_beta(symbol_data, qqq_data) if qqq_data is not None else 0.0
        
        # Detect market regime
        market_regime = self._detect_market_regime(spy_data, window) if spy_data is not None else 'unknown'
        
        # Analyze volatility regime
        volatility_regime = self._analyze_volatility_regime(symbol_data, window)
        
        # Analyze sector strength
        sector_strength = self._analyze_sector_strength(market_data, window)
        
        # Calculate market indicators
        market_indicators = self._calculate_market_indicators(market_data, window)
        
        context = MarketContext(
            spy_correlation=correlations.get('SPY', 0.0),
            qqq_correlation=correlations.get('QQQ', 0.0),
            spy_beta=spy_beta,
            qqq_beta=qqq_beta,
            market_regime=market_regime,
            volatility_regime=volatility_regime,
            sector_strength=sector_strength,
            market_indicators=market_indicators,
            analysis_period=f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )
        
        print(f"                 [OK] SPY Correlation: {correlations.get('SPY', 0.0):.3f}")
        print(f"                 [OK] QQQ Correlation: {correlations.get('QQQ', 0.0):.3f}")
        print(f"                 [OK] SPY Beta: {spy_beta:.3f}")
        print(f"                 [OK] Market Regime: {market_regime}")
        print(f"                 [OK] Volatility Regime: {volatility_regime}")
        
        return context
    
    def _calculate_correlations(self, symbol_data: pd.DataFrame, market_data: Dict[str, pd.DataFrame], 
                               window: int) -> Dict[str, float]:
        """Calculate rolling correlations with market indices"""
        correlations = {}
        
        # Access close price if symbol_data is a DataFrame
        if hasattr(symbol_data, 'columns') and 'close' in symbol_data.columns:
            symbol_price_data = symbol_data['close']
        else:
            symbol_price_data = symbol_data
        symbol_returns = symbol_price_data.pct_change().dropna()
        
        for etf, etf_data in market_data.items():
            if etf_data is not None and len(etf_data) > window:
                # Access close price if etf_data is a DataFrame
                if hasattr(etf_data, 'columns') and 'close' in etf_data.columns:
                    etf_price_data = etf_data['close']
                else:
                    etf_price_data = etf_data
                etf_returns = etf_price_data.pct_change().dropna()
                
                # Align data
                aligned_symbol, aligned_etf = symbol_returns.align(etf_returns, join='inner')
                
                if len(aligned_symbol) > window:
                    # Calculate rolling correlation and take the mean
                    rolling_corr = aligned_symbol.rolling(window).corr(aligned_etf)
                    correlations[etf] = rolling_corr.mean()
        
        return correlations
    
    def _calculate_beta(self, symbol_data: pd.DataFrame, market_data: pd.DataFrame) -> float:
        """Calculate beta (systematic risk) relative to market"""
        if market_data is None or len(market_data) < 30:
            return 0.0
        
        # Use close prices
        if 'close' not in symbol_data.columns or 'close' not in market_data.columns:
            return 0.0
            
        symbol_returns = symbol_data['close'].pct_change().dropna()
        market_returns = market_data['close'].pct_change().dropna()
        
        # Align data
        aligned_symbol, aligned_market = symbol_returns.align(market_returns, join='inner')
        
        if len(aligned_symbol) < 30:
            return 0.0
        
        # Calculate beta using linear regression
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                aligned_market.values.flatten(), 
                aligned_symbol.values.flatten()
            )
            return slope
        except:
            return 0.0
    
    def _detect_market_regime(self, spy_data: pd.DataFrame, window: int) -> str:
        """Detect market regime (bull, bear, sideways)"""
        if spy_data is None or len(spy_data) < window:
            return 'unknown'
        
        # Use close price for calculations
        if 'close' not in spy_data.columns:
            return 'unknown'
            
        spy_close = spy_data['close']
        
        # Calculate trend indicators
        sma_short = spy_close.rolling(window//3).mean()
        sma_long = spy_close.rolling(window).mean()
        
        # Recent trend direction
        recent_trend = (sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
        
        # Volatility
        returns = spy_close.pct_change().dropna()
        recent_vol = returns.tail(window).std()
        
        # Market regime classification
        if recent_trend > 0.02 and recent_vol < 0.015:
            return 'bull'
        elif recent_trend < -0.02 or recent_vol > 0.025:
            return 'bear'
        else:
            return 'sideways'
    
    def _analyze_volatility_regime(self, symbol_data: pd.DataFrame, window: int) -> str:
        """Analyze volatility regime"""
        returns = symbol_data.pct_change().dropna()
        if len(returns) < window:
            return 'unknown'
        
        # Calculate recent volatility
        recent_vol = returns.tail(window).std()
        
        # Calculate historical volatility percentiles
        historical_vol = returns.rolling(window).std().dropna()
        vol_percentile = (historical_vol < recent_vol).sum() / len(historical_vol)
        
        if vol_percentile > 0.8:
            return 'high'
        elif vol_percentile < 0.2:
            return 'low'
        else:
            return 'normal'
    
    def _analyze_sector_strength(self, market_data: Dict[str, pd.DataFrame], 
                                window: int) -> Dict[str, float]:
        """Analyze sector strength using sector ETF performance"""
        sector_strength = {}
        
        for sector_name, etf_symbol in self.sector_etfs.items():
            if etf_symbol in market_data and market_data[etf_symbol] is not None:
                etf_data = market_data[etf_symbol]
                if len(etf_data) >= window and 'close' in etf_data.columns:
                    # Calculate recent performance using close prices
                    etf_close = etf_data['close']
                    recent_return = (etf_close.iloc[-1] - etf_close.iloc[-window]) / etf_close.iloc[-window]
                    sector_strength[sector_name] = float(recent_return)
        
        return sector_strength
    
    def _calculate_market_indicators(self, market_data: Dict[str, pd.DataFrame], 
                                   window: int) -> Dict[str, float]:
        """Calculate comprehensive market indicators"""
        indicators = {}
        
        # VIX analysis (fear index)
        if '^VIX' in market_data and market_data['^VIX'] is not None:
            vix_data = market_data['^VIX']
            if len(vix_data) >= window:
                # Use close price for VIX
                if 'close' in vix_data.columns:
                    vix_close = vix_data['close']
                    indicators['vix_level'] = float(vix_close.iloc[-1])
                    vix_window_data = vix_close.tail(window*3)
                    indicators['vix_percentile'] = float(
                        (vix_window_data < vix_close.iloc[-1]).sum() / len(vix_window_data)
                    )
        
        # Market breadth (using IWM vs SPY)
        if 'IWM' in market_data and 'SPY' in market_data:
            iwm_data = market_data['IWM']
            spy_data = market_data['SPY']
            if iwm_data is not None and spy_data is not None and len(iwm_data) >= window:
                # Use close prices for calculations
                if 'close' in iwm_data.columns and 'close' in spy_data.columns:
                    iwm_close = iwm_data['close']
                    spy_close = spy_data['close']
                    iwm_return = (iwm_close.iloc[-1] - iwm_close.iloc[-window]) / iwm_close.iloc[-window]
                    spy_return = (spy_close.iloc[-1] - spy_close.iloc[-window]) / spy_close.iloc[-window]
                    indicators['market_breadth'] = float(iwm_return - spy_return)
        
        # Tech vs Market (QQQ vs SPY)
        if 'QQQ' in market_data and 'SPY' in market_data:
            qqq_data = market_data['QQQ']
            spy_data = market_data['SPY']
            if qqq_data is not None and spy_data is not None and len(qqq_data) >= window:
                # Use close prices for calculations
                if 'close' in qqq_data.columns and 'close' in spy_data.columns:
                    qqq_close = qqq_data['close']
                    spy_close = spy_data['close']
                    qqq_return = (qqq_close.iloc[-1] - qqq_close.iloc[-window]) / qqq_close.iloc[-window]
                    spy_return = (spy_close.iloc[-1] - spy_close.iloc[-window]) / spy_close.iloc[-window]
                    indicators['tech_leadership'] = float(qqq_return - spy_return)
        
        return indicators


class EnhancedFeatureEngineer:
    """
    Enhanced Feature Engineering for ML Models
    
    Creates comprehensive feature sets including:
    - Technical indicators (40+ indicators using TA-Lib)
    - Market context features
    - Volatility features
    - Momentum features
    - Mean reversion features
    """
    
    def __init__(self, market_analyzer: MarketContextAnalyzer):
        """Initialize Enhanced Feature Engineer"""
        self.market_analyzer = market_analyzer
        self.scaler = StandardScaler()
        
        print(f"[FEATURE_ENG] Enhanced Feature Engineer initialized")
        print(f"              TA-Lib Available: {TALIB_AVAILABLE}")
    
    def create_enhanced_features(self, symbol: str, start_date: dt.datetime, 
                               end_date: dt.datetime, window: int = 60,
                               include_market_context: bool = True,
                               normalize_features: bool = True) -> EnhancedFeatures:
        """
        Create enhanced feature set for ML training
        
        Args:
            symbol: Target symbol
            start_date: Feature calculation start date
            end_date: Feature calculation end date  
            window: Window for technical indicators
            include_market_context: Include market context features
            normalize_features: Normalize features using StandardScaler
            
        Returns:
            EnhancedFeatures object with features, labels, and metadata
        """
        print(f"[FEATURE_CREATE] Creating enhanced features for {symbol}")
        print(f"                 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Get price data
        symbols_for_features = [symbol]
        if include_market_context:
            symbols_for_features.extend(['SPY', 'QQQ', '^VIX'])
        
        data = self.market_analyzer.data_provider.get_market_data(
            symbols=symbols_for_features,
            start_date=start_date,
            end_date=end_date
        )
        
        if symbol not in data:
            raise ValueError(f"No data available for {symbol}")
        
        symbol_data = data[symbol]
        
        # Create base technical features
        technical_features, technical_names = self._create_technical_features(symbol_data, window)
        
        # Create market context features
        if include_market_context:
            market_features, market_names = self._create_market_context_features(
                symbol_data, data, window
            )
        else:
            market_features, market_names = np.array([]), []
        
        # Combine all features
        if market_features.size > 0:
            all_features = np.hstack([technical_features, market_features])
            all_feature_names = technical_names + market_names
        else:
            all_features = technical_features
            all_feature_names = technical_names
        
        # Create labels
        labels = self._create_labels(symbol_data)
        
        # Align features and labels
        min_length = min(len(all_features), len(labels))
        all_features = all_features[:min_length]
        labels = labels[:min_length]
        
        # Remove rows with NaN values
        valid_rows = ~np.isnan(all_features).any(axis=1) & ~np.isnan(labels)
        all_features = all_features[valid_rows]
        labels = labels[valid_rows]
        
        # Normalize features if requested
        if normalize_features and len(all_features) > 0:
            all_features = self.scaler.fit_transform(all_features)
        
        enhanced_features = EnhancedFeatures(
            features=all_features,
            feature_names=all_feature_names,
            target_labels=labels,
            metadata={
                'symbol': symbol,
                'feature_count': len(all_feature_names),
                'sample_count': len(all_features),
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'window': window,
                'include_market_context': include_market_context,
                'normalized': normalize_features,
                'talib_available': TALIB_AVAILABLE
            }
        )
        
        print(f"                 [OK] Features: {len(all_feature_names)}")
        print(f"                 [OK] Samples: {len(all_features)}")
        print(f"                 [OK] Labels: BUY={np.sum(labels == 1)}, SELL={np.sum(labels == -1)}, HOLD={np.sum(labels == 0)}")
        
        return enhanced_features
    
    def _create_technical_features(self, data: pd.DataFrame, window: int) -> Tuple[np.ndarray, List[str]]:
        """Create comprehensive technical indicator features"""
        price_data = data.values.flatten()
        features = []
        feature_names = []
        
        if TALIB_AVAILABLE:
            # TA-Lib indicators
            features_dict = self._calculate_talib_features(price_data, data.index)
            for name, values in features_dict.items():
                if values is not None and len(values) == len(data):
                    features.append(values)
                    feature_names.append(name)
        else:
            # Simplified indicators if TA-Lib not available
            features_dict = self._calculate_simple_features(data, window)
            for name, values in features_dict.items():
                if values is not None and len(values) == len(data):
                    features.append(values)
                    feature_names.append(name)
        
        if features:
            return np.column_stack(features), feature_names
        else:
            return np.array([]).reshape(len(data), 0), []
    
    def _calculate_talib_features(self, price_data: np.ndarray, index: pd.DatetimeIndex) -> Dict[str, np.ndarray]:
        """Calculate features using TA-Lib with centralized configuration"""
        features = {}
        
        # Ensure we have enough data
        if len(price_data) < 50:
            return features
        
        try:
            # Get centralized configuration
            config_manager = get_config_manager()
            tech_config = config_manager.get_technical_indicators_config()
            
            # Trend Indicators
            features['SMA_10'] = talib.SMA(price_data, timeperiod=tech_config.get('sma_periods', [10])[0] if isinstance(tech_config.get('sma_periods', [10]), list) else 10)
            features['SMA_20'] = talib.SMA(price_data, timeperiod=tech_config.get('sma_short_period', 20))
            features['SMA_50'] = talib.SMA(price_data, timeperiod=tech_config.get('sma_long_period', 50))
            features['EMA_12'] = talib.EMA(price_data, timeperiod=tech_config.get('ema_short_period', 12))
            features['EMA_26'] = talib.EMA(price_data, timeperiod=tech_config.get('ema_long_period', 26))
            
            # Momentum Indicators
            features['RSI'] = talib.RSI(price_data, timeperiod=tech_config.get('rsi_period', 14))
            features['MOM'] = talib.MOM(price_data, timeperiod=tech_config.get('momentum_periods', [10])[0] if isinstance(tech_config.get('momentum_periods', [10]), list) else 10)
            features['ROC'] = talib.ROC(price_data, timeperiod=tech_config.get('roc_period', 10))
            features['CCI'] = talib.CCI(price_data, price_data, price_data, timeperiod=tech_config.get('cci_period', 14))
            features['WILLR'] = talib.WILLR(price_data, price_data, price_data, timeperiod=tech_config.get('williams_r_period', 14))
            
            # MACD
            macd_fast = tech_config.get('macd_fast_period', 12)
            macd_slow = tech_config.get('macd_slow_period', 26)
            macd_signal = tech_config.get('macd_signal_period', 9)
            macd, macdsignal, macdhist = talib.MACD(price_data, fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal)
            features['MACD'] = macd
            features['MACD_SIGNAL'] = macdsignal
            features['MACD_HIST'] = macdhist
            
            # Bollinger Bands
            bb_period = tech_config.get('bollinger_bands_period', 20)
            bb_std_dev = tech_config.get('bollinger_bands_std_dev', 2)
            bb_upper, bb_middle, bb_lower = talib.BBANDS(price_data, timeperiod=bb_period, nbdevup=bb_std_dev, nbdevdn=bb_std_dev)
            features['BB_UPPER'] = bb_upper
            features['BB_MIDDLE'] = bb_middle
            features['BB_LOWER'] = bb_lower
            features['BB_WIDTH'] = (bb_upper - bb_lower) / bb_middle
            features['BB_POSITION'] = (price_data - bb_lower) / (bb_upper - bb_lower)
            
            # Volatility Indicators
            atr_period = tech_config.get('atr_period', 14)
            features['ATR'] = talib.ATR(price_data, price_data, price_data, timeperiod=atr_period)
            
            # Overlap Studies
            dema_period = tech_config.get('dema_period', 30)
            tema_period = tech_config.get('tema_period', 30)
            trima_period = tech_config.get('trima_period', 30)
            features['DEMA'] = talib.DEMA(price_data, timeperiod=dema_period)
            features['TEMA'] = talib.TEMA(price_data, timeperiod=tema_period)
            features['TRIMA'] = talib.TRIMA(price_data, timeperiod=trima_period)
            
            # Pattern Recognition (comprehensive set matching original enhanced_strategy.py)
            features['CDL_DOJI'] = talib.CDLDOJI(price_data, price_data, price_data, price_data).astype(float)
            features['CDL_HAMMER'] = talib.CDLHAMMER(price_data, price_data, price_data, price_data).astype(float)
            features['CDL_HANGINGMAN'] = talib.CDLHANGINGMAN(price_data, price_data, price_data, price_data).astype(float)
            features['CDL_ENGULFING'] = talib.CDLENGULFING(price_data, price_data, price_data, price_data).astype(float)
            features['CDL_MORNINGSTAR'] = talib.CDLMORNINGSTAR(price_data, price_data, price_data, price_data).astype(float)
            features['CDL_EVENINGSTAR'] = talib.CDLEVENINGSTAR(price_data, price_data, price_data, price_data).astype(float)
            features['CDL_SHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(price_data, price_data, price_data, price_data).astype(float)
            
            # Additional key patterns
            features['CDL_HARAMI'] = talib.CDLHARAMI(price_data, price_data, price_data, price_data).astype(float)
            features['CDL_PIERCING'] = talib.CDLPIERCING(price_data, price_data, price_data, price_data).astype(float)
            features['CDL_DARKCLOUD'] = talib.CDLDARKCLOUDCOVER(price_data, price_data, price_data, price_data).astype(float)
            
        except Exception as e:
            print(f"[WARNING] TA-Lib feature calculation failed: {e}")
        
        return features
    
    def _calculate_simple_features(self, data: pd.DataFrame, window: int) -> Dict[str, np.ndarray]:
        """Calculate simplified features when TA-Lib is not available"""
        features = {}
        price_data = data.iloc[:, 0]  # Assume first column is price
        
        # Moving averages
        features['SMA_10'] = price_data.rolling(10).mean().values
        features['SMA_20'] = price_data.rolling(20).mean().values
        features['SMA_50'] = price_data.rolling(50).mean().values
        
        # Simple RSI
        delta = price_data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['RSI'] = (100 - (100 / (1 + rs))).values
        
        # Price momentum
        features['MOM_10'] = (price_data / price_data.shift(10) - 1).values
        features['MOM_20'] = (price_data / price_data.shift(20) - 1).values
        
        # Volatility
        features['VOLATILITY'] = price_data.rolling(20).std().values
        
        # Bollinger Bands
        sma20 = price_data.rolling(20).mean()
        std20 = price_data.rolling(20).std()
        features['BB_UPPER'] = (sma20 + 2*std20).values
        features['BB_LOWER'] = (sma20 - 2*std20).values
        features['BB_POSITION'] = ((price_data - (sma20 - 2*std20)) / (4*std20)).values
        
        return features
    
    def _create_market_context_features(self, symbol_data: pd.DataFrame, 
                                      all_data: Dict[str, pd.DataFrame], 
                                      window: int) -> Tuple[np.ndarray, List[str]]:
        """Create market context features"""
        features = []
        feature_names = []
        
        # SPY correlation and beta
        if 'SPY' in all_data and all_data['SPY'] is not None:
            spy_data = all_data['SPY']
            spy_corr = self._calculate_rolling_correlation(symbol_data, spy_data, window)
            spy_beta = self._calculate_rolling_beta(symbol_data, spy_data, window)
            
            features.extend([spy_corr, spy_beta])
            feature_names.extend(['SPY_CORRELATION', 'SPY_BETA'])
        
        # QQQ correlation and beta
        if 'QQQ' in all_data and all_data['QQQ'] is not None:
            qqq_data = all_data['QQQ']
            qqq_corr = self._calculate_rolling_correlation(symbol_data, qqq_data, window)
            qqq_beta = self._calculate_rolling_beta(symbol_data, qqq_data, window)
            
            features.extend([qqq_corr, qqq_beta])
            feature_names.extend(['QQQ_CORRELATION', 'QQQ_BETA'])
        
        # VIX features
        if '^VIX' in all_data and all_data['^VIX'] is not None:
            vix_data = all_data['^VIX']
            # Access close price properly
            if hasattr(vix_data, 'columns') and 'close' in vix_data.columns:
                vix_close = vix_data['close']
            else:
                vix_close = vix_data
            
            # Ensure proper length alignment
            vix_level = self._align_series_to_data(vix_close, symbol_data)
            vix_change = self._align_series_to_data(vix_close.pct_change().fillna(0), symbol_data)
            
            features.extend([vix_level, vix_change])
            feature_names.extend(['VIX_LEVEL', 'VIX_CHANGE'])
        
        if features:
            return np.column_stack(features), feature_names
        else:
            return np.array([]).reshape(len(symbol_data), 0), []
    
    def _calculate_rolling_correlation(self, data1: pd.DataFrame, data2: pd.DataFrame, 
                                     window: int) -> np.ndarray:
        """Calculate rolling correlation between two time series"""
        # Extract close prices
        if hasattr(data1, 'columns') and 'close' in data1.columns:
            price1 = data1['close']
        else:
            price1 = data1
            
        if hasattr(data2, 'columns') and 'close' in data2.columns:
            price2 = data2['close']
        else:
            price2 = data2
            
        returns1 = price1.pct_change()
        returns2 = price2.pct_change()
        
        # Align data
        aligned1, aligned2 = returns1.align(returns2, join='inner')
        
        # Calculate rolling correlation
        rolling_corr = aligned1.rolling(window).corr(aligned2).fillna(0)
        
        # Align back to original data length
        result = np.zeros(len(data1))
        if len(rolling_corr) > 0:
            result[-len(rolling_corr):] = rolling_corr.values.flatten()
        
        return result
    
    def _calculate_rolling_beta(self, data1: pd.DataFrame, data2: pd.DataFrame, 
                              window: int) -> np.ndarray:
        """Calculate rolling beta between two time series"""
        # Extract close prices
        if hasattr(data1, 'columns') and 'close' in data1.columns:
            price1 = data1['close']
        else:
            price1 = data1
            
        if hasattr(data2, 'columns') and 'close' in data2.columns:
            price2 = data2['close']
        else:
            price2 = data2
            
        returns1 = price1.pct_change()
        returns2 = price2.pct_change()
        
        # Align data
        aligned1, aligned2 = returns1.align(returns2, join='inner')
        
        def rolling_beta(x, y):
            try:
                slope, _, _, _, _ = stats.linregress(y, x)
                return slope
            except:
                return 0.0
        
        # Calculate rolling beta
        rolling_beta_values = []
        for i in range(len(aligned1)):
            if i < window:
                rolling_beta_values.append(0.0)
            else:
                x_window = aligned1.iloc[i-window:i]
                y_window = aligned2.iloc[i-window:i]
                beta = rolling_beta(x_window.values.flatten(), y_window.values.flatten())
                rolling_beta_values.append(beta)
        
        # Align back to original data length
        result = np.zeros(len(data1))
        if rolling_beta_values:
            result[-len(rolling_beta_values):] = rolling_beta_values
        
        return result
    
    def _create_labels(self, data: pd.DataFrame, future_periods: int = 3, 
                      buy_threshold: float = 0.02, sell_threshold: float = -0.02) -> np.ndarray:
        """Create trading labels based on future returns"""
        price_data = data.iloc[:, 0]  # Assume first column is price
        labels = np.zeros(len(data))
        
        for i in range(len(data) - future_periods):
            current_price = price_data.iloc[i]
            future_price = price_data.iloc[i + future_periods]
            future_return = (future_price - current_price) / current_price
            
            if future_return > buy_threshold:
                labels[i] = 1  # BUY
            elif future_return < sell_threshold:
                labels[i] = -1  # SELL
            else:
                labels[i] = 0  # HOLD
        
        return labels
    
    def _align_series_to_data(self, series: pd.Series, target_data: pd.DataFrame) -> np.ndarray:
        """Align a series to match the length of target data"""
        target_length = len(target_data)
        
        if len(series) == target_length:
            return series.values
        elif len(series) > target_length:
            # Take the last N values to match target length
            return series.tail(target_length).values
        else:
            # Pad with NaN if series is shorter
            result = np.full(target_length, np.nan)
            result[-len(series):] = series.values
            return result