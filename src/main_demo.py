"""
Modular Trading System Main Application
Orchestrates the entire trading system using dependency injection and factory patterns
"""

import logging
import datetime as dt
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Import interfaces
from src.interfaces.data_provider import DataProvider
from src.interfaces.signal_generator import SignalGenerator, TradingSignal
from src.interfaces.trading_strategy import TradingStrategy
from src.interfaces.backtester import Backtester, BacktestConfig
from src.interfaces.risk_manager import RiskManager

# Import concrete implementations
from src.data.providers import YFinanceProvider
from src.signals.technical import RSISignalGenerator, MACDSignalGenerator, BollingerBandsSignalGenerator
from src.data.preprocessors import DataPreprocessor, TechnicalIndicatorCalculator

# Import existing components (bridge to legacy)
from config_manager import get_config
from model_management import ModelManager


class TradingSystemOrchestrator:
    """
    Main orchestrator for the modular trading system
    
    Implements Dependency Injection and Factory patterns to create
    a loosely coupled, highly configurable trading system
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the trading system orchestrator
        
        Args:
            config_path: Path to configuration file
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = get_config()
        
        # Initialize components
        self.data_provider: Optional[DataProvider] = None
        self.preprocessor: Optional[DataPreprocessor] = None
        self.indicator_calculator: Optional[TechnicalIndicatorCalculator] = None
        self.signal_generators: List[SignalGenerator] = []
        self.trading_strategy: Optional[TradingStrategy] = None
        self.risk_manager: Optional[RiskManager] = None
        self.backtester: Optional[Backtester] = None
        
        # Legacy components (bridge to existing system)
        self.model_manager = ModelManager()
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self) -> None:
        """Initialize all system components based on configuration"""
        try:
            self.logger.info("🚀 Initializing Modular Trading System")
            
            # Initialize data provider
            self._initialize_data_provider()
            
            # Initialize preprocessing components
            self._initialize_preprocessors()
            
            # Initialize signal generators
            self._initialize_signal_generators()
            
            # Initialize trading components (placeholder for now)
            self._initialize_trading_components()
            
            self.logger.info("✅ Modular Trading System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize trading system: {str(e)}")
            raise
    
    def _initialize_data_provider(self) -> None:
        """Initialize data provider based on configuration"""
        provider_config = self.config.config.get('data_provider', {})
        provider_type = provider_config.get('type', 'yfinance')
        
        if provider_type.lower() == 'yfinance':
            self.data_provider = YFinanceProvider(provider_config)
            self.logger.info("📊 YFinance data provider initialized")
        else:
            raise ValueError(f"Unsupported data provider type: {provider_type}")
    
    def _initialize_preprocessors(self) -> None:
        """Initialize data preprocessing components"""
        preprocess_config = self.config.config.get('preprocessing', {})
        indicator_config = self.config.config.get('technical_indicators', {})
        
        self.preprocessor = DataPreprocessor(preprocess_config)
        self.indicator_calculator = TechnicalIndicatorCalculator(indicator_config)
        
        self.logger.info("🔧 Data preprocessing components initialized")
    
    def _initialize_signal_generators(self) -> None:
        """Initialize signal generators based on configuration"""
        signals_config = self.config.config.get('signals', {})
        
        # RSI Signal Generator
        if signals_config.get('rsi', {}).get('enabled', True):
            rsi_config = signals_config.get('rsi', {})
            self.signal_generators.append(RSISignalGenerator(rsi_config))
            self.logger.info("📈 RSI signal generator initialized")
        
        # MACD Signal Generator
        if signals_config.get('macd', {}).get('enabled', True):
            macd_config = signals_config.get('macd', {})
            self.signal_generators.append(MACDSignalGenerator(macd_config))
            self.logger.info("📊 MACD signal generator initialized")
        
        # Bollinger Bands Signal Generator
        if signals_config.get('bollinger_bands', {}).get('enabled', True):
            bb_config = signals_config.get('bollinger_bands', {})
            self.signal_generators.append(BollingerBandsSignalGenerator(bb_config))
            self.logger.info("📉 Bollinger Bands signal generator initialized")
        
        self.logger.info(f"🎯 {len(self.signal_generators)} signal generators initialized")
    
    def _initialize_trading_components(self) -> None:
        """Initialize trading strategy, risk manager, and backtester (placeholder)"""
        # Placeholder - these would be implemented with concrete classes
        self.logger.info("⚠️ Trading components initialization (placeholder)")
    
    def run_analysis(self, symbol: str, start_date: dt.datetime, 
                    end_date: dt.datetime) -> Dict[str, Any]:
        """
        Run complete analysis for a symbol
        
        Args:
            symbol: Trading symbol
            start_date: Analysis start date
            end_date: Analysis end date
            
        Returns:
            Complete analysis results
        """
        try:
            self.logger.info(f"🔍 Starting analysis for {symbol} ({start_date.date()} to {end_date.date()})")
            
            # 1. Get market data
            market_data = self._get_market_data(symbol, start_date, end_date)
            if market_data.empty:
                raise ValueError(f"No market data available for {symbol}")
            
            # 2. Preprocess data
            cleaned_data = self._preprocess_data(market_data, symbol)
            
            # 3. Calculate technical indicators
            indicators = self._calculate_indicators(cleaned_data)
            
            # 4. Generate signals
            all_signals = self._generate_signals(cleaned_data, symbol)
            
            # 5. Analyze results
            analysis_results = self._analyze_results(symbol, cleaned_data, indicators, all_signals)
            
            self.logger.info(f"✅ Analysis completed for {symbol}")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"❌ Analysis failed for {symbol}: {str(e)}")
            return {'error': str(e)}
    
    def _get_market_data(self, symbol: str, start_date: dt.datetime, 
                        end_date: dt.datetime) -> pd.DataFrame:
        """Get and validate market data"""
        if not self.data_provider:
            raise RuntimeError("Data provider not initialized")
        
        # Validate symbol
        if not self.data_provider.validate_symbol(symbol):
            raise ValueError(f"Invalid or non-tradeable symbol: {symbol}")
        
        # Get historical data
        data = self.data_provider.get_historical_data(symbol, start_date, end_date)
        
        if data.empty:
            raise ValueError(f"No market data returned for {symbol}")
        
        self.logger.info(f"📈 Retrieved {len(data)} days of market data for {symbol}")
        return data
    
    def _preprocess_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean and preprocess market data"""
        if not self.preprocessor:
            raise RuntimeError("Data preprocessor not initialized")
        
        cleaned_data = self.preprocessor.clean_data(data, symbol)
        
        if cleaned_data.empty:
            raise ValueError(f"Data preprocessing failed for {symbol}")
        
        self.logger.info(f"🧹 Data preprocessing completed for {symbol}")
        return cleaned_data
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        if not self.indicator_calculator:
            raise RuntimeError("Indicator calculator not initialized")
        
        indicators = self.indicator_calculator.calculate_all_indicators(data)
        
        self.logger.info(f"📊 Calculated {len(indicators.columns) if not indicators.empty else 0} technical indicators")
        return indicators
    
    def _generate_signals(self, data: pd.DataFrame, symbol: str) -> List[TradingSignal]:
        """Generate signals from all signal generators"""
        all_signals = []
        
        for generator in self.signal_generators:
            if not generator.is_enabled():
                continue
            
            try:
                signals = generator.generate_signals(data, symbol)
                all_signals.extend(signals)
                self.logger.info(f"📡 Generated {len(signals)} signals from {generator.get_name()}")
                
            except Exception as e:
                self.logger.error(f"❌ Signal generation failed for {generator.get_name()}: {str(e)}")
                continue
        
        self.logger.info(f"🎯 Generated {len(all_signals)} total signals for {symbol}")
        return all_signals
    
    def _analyze_results(self, symbol: str, data: pd.DataFrame, 
                        indicators: pd.DataFrame, signals: List[TradingSignal]) -> Dict[str, Any]:
        """Analyze and summarize results"""
        # Signal analysis
        buy_signals = [s for s in signals if s.signal_type.value == 1]
        sell_signals = [s for s in signals if s.signal_type.value == -1]
        
        # Signal breakdown by source
        signal_sources = {}
        for signal in signals:
            source = signal.source
            if source not in signal_sources:
                signal_sources[source] = {'buy': 0, 'sell': 0, 'total': 0}
            
            if signal.signal_type.value == 1:
                signal_sources[source]['buy'] += 1
            elif signal.signal_type.value == -1:
                signal_sources[source]['sell'] += 1
            signal_sources[source]['total'] += 1
        
        # Calculate average confidence
        avg_confidence = np.mean([s.confidence for s in signals]) if signals else 0
        
        # Price statistics
        price_stats = {
            'start_price': data['close'].iloc[0],
            'end_price': data['close'].iloc[-1],
            'min_price': data['close'].min(),
            'max_price': data['close'].max(),
            'price_change': (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100,
            'volatility': data['close'].pct_change().std() * np.sqrt(252) * 100
        }
        
        results = {
            'symbol': symbol,
            'analysis_period': {
                'start': data.index[0].isoformat(),
                'end': data.index[-1].isoformat(),
                'days': len(data)
            },
            'price_statistics': price_stats,
            'signal_summary': {
                'total_signals': len(signals),
                'buy_signals': len(buy_signals),
                'sell_signals': len(sell_signals),
                'average_confidence': avg_confidence,
                'signal_sources': signal_sources
            },
            'technical_indicators': {
                'indicators_calculated': len(indicators.columns) if not indicators.empty else 0,
                'data_coverage': len(indicators) if not indicators.empty else 0
            }
        }
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'data_provider': {
                'type': type(self.data_provider).__name__ if self.data_provider else None,
                'available': self.data_provider.is_available() if self.data_provider else False
            },
            'signal_generators': [
                {
                    'name': gen.get_name(),
                    'enabled': gen.is_enabled(),
                    'config': gen.get_config()
                }
                for gen in self.signal_generators
            ],
            'components_initialized': {
                'data_provider': self.data_provider is not None,
                'preprocessor': self.preprocessor is not None,
                'indicator_calculator': self.indicator_calculator is not None,
                'signal_generators': len(self.signal_generators) > 0
            }
        }


def main():
    """Main function to demonstrate the modular trading system"""
    try:
        # Initialize the trading system
        trading_system = TradingSystemOrchestrator()
        
        # Check system status
        status = trading_system.get_system_status()
        print("\n🔍 System Status:")
        print(f"Data Provider Available: {status['data_provider']['available']}")
        print(f"Signal Generators: {len(status['signal_generators'])}")
        
        # Run analysis on a sample symbol
        symbol = "AAPL"
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=90)  # 3 months of data
        
        print(f"\n🚀 Running analysis for {symbol}...")
        results = trading_system.run_analysis(symbol, start_date, end_date)
        
        if 'error' not in results:
            print("\n📊 Analysis Results:")
            print(f"Symbol: {results['symbol']}")
            print(f"Period: {results['analysis_period']['days']} days")
            print(f"Price Change: {results['price_statistics']['price_change']:.2f}%")
            print(f"Volatility: {results['price_statistics']['volatility']:.2f}%")
            print(f"Total Signals: {results['signal_summary']['total_signals']}")
            print(f"Buy Signals: {results['signal_summary']['buy_signals']}")
            print(f"Sell Signals: {results['signal_summary']['sell_signals']}")
            print(f"Average Confidence: {results['signal_summary']['average_confidence']:.3f}")
            
            print("\n📡 Signal Sources:")
            for source, counts in results['signal_summary']['signal_sources'].items():
                print(f"  {source}: {counts['total']} signals (Buy: {counts['buy']}, Sell: {counts['sell']})")
        else:
            print(f"❌ Analysis failed: {results['error']}")
        
    except Exception as e:
        print(f"❌ System initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()