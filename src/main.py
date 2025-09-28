"""
Modular Trading System Main Application
Orchestrates the entire trading system using dependency injection and factory patterns
"""

import logging
import datetime as dt
from typing import Dict, Any, List, Optional
import pandas as pd

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
        self.logger.info("⚠️ Trading components initialization (placeholder)")\n    \n    def run_analysis(self, symbol: str, start_date: dt.datetime, \n                    end_date: dt.datetime) -> Dict[str, Any]:\n        \"\"\"\n        Run complete analysis for a symbol\n        \n        Args:\n            symbol: Trading symbol\n            start_date: Analysis start date\n            end_date: Analysis end date\n            \n        Returns:\n            Complete analysis results\n        \"\"\"\n        try:\n            self.logger.info(f\"🔍 Starting analysis for {symbol} ({start_date.date()} to {end_date.date()})\")\n            \n            # 1. Get market data\n            market_data = self._get_market_data(symbol, start_date, end_date)\n            if market_data.empty:\n                raise ValueError(f\"No market data available for {symbol}\")\n            \n            # 2. Preprocess data\n            cleaned_data = self._preprocess_data(market_data, symbol)\n            \n            # 3. Calculate technical indicators\n            indicators = self._calculate_indicators(cleaned_data)\n            \n            # 4. Generate signals\n            all_signals = self._generate_signals(cleaned_data, symbol)\n            \n            # 5. Analyze results\n            analysis_results = self._analyze_results(symbol, cleaned_data, indicators, all_signals)\n            \n            self.logger.info(f\"✅ Analysis completed for {symbol}\")\n            return analysis_results\n            \n        except Exception as e:\n            self.logger.error(f\"❌ Analysis failed for {symbol}: {str(e)}\")\n            return {'error': str(e)}\n    \n    def _get_market_data(self, symbol: str, start_date: dt.datetime, \n                        end_date: dt.datetime) -> pd.DataFrame:\n        \"\"\"Get and validate market data\"\"\"\n        if not self.data_provider:\n            raise RuntimeError(\"Data provider not initialized\")\n        \n        # Validate symbol\n        if not self.data_provider.validate_symbol(symbol):\n            raise ValueError(f\"Invalid or non-tradeable symbol: {symbol}\")\n        \n        # Get historical data\n        data = self.data_provider.get_historical_data(symbol, start_date, end_date)\n        \n        if data.empty:\n            raise ValueError(f\"No market data returned for {symbol}\")\n        \n        self.logger.info(f\"📈 Retrieved {len(data)} days of market data for {symbol}\")\n        return data\n    \n    def _preprocess_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:\n        \"\"\"Clean and preprocess market data\"\"\"\n        if not self.preprocessor:\n            raise RuntimeError(\"Data preprocessor not initialized\")\n        \n        cleaned_data = self.preprocessor.clean_data(data, symbol)\n        \n        if cleaned_data.empty:\n            raise ValueError(f\"Data preprocessing failed for {symbol}\")\n        \n        self.logger.info(f\"🧹 Data preprocessing completed for {symbol}\")\n        return cleaned_data\n    \n    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:\n        \"\"\"Calculate technical indicators\"\"\"\n        if not self.indicator_calculator:\n            raise RuntimeError(\"Indicator calculator not initialized\")\n        \n        indicators = self.indicator_calculator.calculate_all_indicators(data)\n        \n        self.logger.info(f\"📊 Calculated {len(indicators.columns) if not indicators.empty else 0} technical indicators\")\n        return indicators\n    \n    def _generate_signals(self, data: pd.DataFrame, symbol: str) -> List[TradingSignal]:\n        \"\"\"Generate signals from all signal generators\"\"\"\n        all_signals = []\n        \n        for generator in self.signal_generators:\n            if not generator.is_enabled():\n                continue\n            \n            try:\n                signals = generator.generate_signals(data, symbol)\n                all_signals.extend(signals)\n                self.logger.info(f\"📡 Generated {len(signals)} signals from {generator.get_name()}\")\n                \n            except Exception as e:\n                self.logger.error(f\"❌ Signal generation failed for {generator.get_name()}: {str(e)}\")\n                continue\n        \n        self.logger.info(f\"🎯 Generated {len(all_signals)} total signals for {symbol}\")\n        return all_signals\n    \n    def _analyze_results(self, symbol: str, data: pd.DataFrame, \n                        indicators: pd.DataFrame, signals: List[TradingSignal]) -> Dict[str, Any]:\n        \"\"\"Analyze and summarize results\"\"\"\n        # Signal analysis\n        buy_signals = [s for s in signals if s.signal_type.value == 1]\n        sell_signals = [s for s in signals if s.signal_type.value == -1]\n        \n        # Signal breakdown by source\n        signal_sources = {}\n        for signal in signals:\n            source = signal.source\n            if source not in signal_sources:\n                signal_sources[source] = {'buy': 0, 'sell': 0, 'total': 0}\n            \n            if signal.signal_type.value == 1:\n                signal_sources[source]['buy'] += 1\n            elif signal.signal_type.value == -1:\n                signal_sources[source]['sell'] += 1\n            signal_sources[source]['total'] += 1\n        \n        # Calculate average confidence\n        avg_confidence = np.mean([s.confidence for s in signals]) if signals else 0\n        \n        # Price statistics\n        price_stats = {\n            'start_price': data['close'].iloc[0],\n            'end_price': data['close'].iloc[-1],\n            'min_price': data['close'].min(),\n            'max_price': data['close'].max(),\n            'price_change': (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100,\n            'volatility': data['close'].pct_change().std() * np.sqrt(252) * 100\n        }\n        \n        results = {\n            'symbol': symbol,\n            'analysis_period': {\n                'start': data.index[0].isoformat(),\n                'end': data.index[-1].isoformat(),\n                'days': len(data)\n            },\n            'price_statistics': price_stats,\n            'signal_summary': {\n                'total_signals': len(signals),\n                'buy_signals': len(buy_signals),\n                'sell_signals': len(sell_signals),\n                'average_confidence': avg_confidence,\n                'signal_sources': signal_sources\n            },\n            'technical_indicators': {\n                'indicators_calculated': len(indicators.columns) if not indicators.empty else 0,\n                'data_coverage': len(indicators) if not indicators.empty else 0\n            }\n        }\n        \n        return results\n    \n    def get_system_status(self) -> Dict[str, Any]:\n        \"\"\"Get current system status\"\"\"\n        return {\n            'data_provider': {\n                'type': type(self.data_provider).__name__ if self.data_provider else None,\n                'available': self.data_provider.is_available() if self.data_provider else False\n            },\n            'signal_generators': [\n                {\n                    'name': gen.get_name(),\n                    'enabled': gen.is_enabled(),\n                    'config': gen.get_config()\n                }\n                for gen in self.signal_generators\n            ],\n            'components_initialized': {\n                'data_provider': self.data_provider is not None,\n                'preprocessor': self.preprocessor is not None,\n                'indicator_calculator': self.indicator_calculator is not None,\n                'signal_generators': len(self.signal_generators) > 0\n            }\n        }\n\n\ndef main():\n    \"\"\"Main function to demonstrate the modular trading system\"\"\"\n    try:\n        # Initialize the trading system\n        trading_system = TradingSystemOrchestrator()\n        \n        # Check system status\n        status = trading_system.get_system_status()\n        print(\"\\n🔍 System Status:\")\n        print(f\"Data Provider Available: {status['data_provider']['available']}\")\n        print(f\"Signal Generators: {len(status['signal_generators'])}\")\n        \n        # Run analysis on a sample symbol\n        symbol = \"AAPL\"\n        end_date = dt.datetime.now()\n        start_date = end_date - dt.timedelta(days=90)  # 3 months of data\n        \n        print(f\"\\n🚀 Running analysis for {symbol}...\")\n        results = trading_system.run_analysis(symbol, start_date, end_date)\n        \n        if 'error' not in results:\n            print(\"\\n📊 Analysis Results:\")\n            print(f\"Symbol: {results['symbol']}\")\n            print(f\"Period: {results['analysis_period']['days']} days\")\n            print(f\"Price Change: {results['price_statistics']['price_change']:.2f}%\")\n            print(f\"Volatility: {results['price_statistics']['volatility']:.2f}%\")\n            print(f\"Total Signals: {results['signal_summary']['total_signals']}\")\n            print(f\"Buy Signals: {results['signal_summary']['buy_signals']}\")\n            print(f\"Sell Signals: {results['signal_summary']['sell_signals']}\")\n            print(f\"Average Confidence: {results['signal_summary']['average_confidence']:.3f}\")\n            \n            print(\"\\n📡 Signal Sources:\")\n            for source, counts in results['signal_summary']['signal_sources'].items():\n                print(f\"  {source}: {counts['total']} signals (Buy: {counts['buy']}, Sell: {counts['sell']})\")\n        else:\n            print(f\"❌ Analysis failed: {results['error']}\")\n        \n    except Exception as e:\n        print(f\"❌ System initialization failed: {str(e)}\")\n        import traceback\n        traceback.print_exc()\n\n\nif __name__ == \"__main__\":\n    main()