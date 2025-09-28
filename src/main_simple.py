"""
Modular Trading System Main Application
Orchestrates the entire trading system using dependency injection and factory patterns
"""

import logging
import datetime as dt
import numpy as np
import sys
import os
from typing import Dict, Any, List, Optional
import pandas as pd
import yaml

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our TradingView chart generator
from tradingview_charts import TradingViewChartGenerator

# Simple config manager inline
class ConfigManager:
    def __init__(self):
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

def get_config():
    return ConfigManager()


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
        
        # Initialize chart generator
        self.chart_generator = TradingViewChartGenerator()
        
        self.logger.info("🚀 Trading System Orchestrator initialized")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'configuration_loaded': True,
            'primary_algorithm': self.config.get('machine_learning.primary_algorithm', 'Unknown'),
            'default_symbol': self.config.get('trading.default_symbol', 'Unknown'),
            'supported_assets': len(self.config.get('trading.assets.stocks', [])),
            'environment': self.config.get('application.environment', 'Unknown')
        }
    
    def run_simple_analysis(self, symbol: str = None) -> Dict[str, Any]:
        """
        Run a simple analysis demonstration
        
        Args:
            symbol: Trading symbol to analyze (defaults to config default)
            
        Returns:
            Analysis results
        """
        if symbol is None:
            symbol = self.config.get('trading.default_symbol', 'AAPL')
        
        try:
            self.logger.info(f"🔍 Running simple analysis for {symbol}")
            
            # Get configuration details for the symbol
            assets = self.config.get('trading.assets.stocks', [])
            asset_info = None
            for asset in assets:
                if asset.get('symbol') == symbol:
                    asset_info = asset
                    break
            
            if not asset_info:
                return {'error': f'Symbol {symbol} not configured in assets'}
            
            # Get ML configuration
            ml_config = self.config.get('machine_learning', {})
            
            results = {
                'symbol': symbol,
                'asset_info': asset_info,
                'ml_configuration': {
                    'primary_algorithm': ml_config.get('primary_algorithm', 'Unknown'),
                    'retraining_frequency': ml_config.get('retraining', {}).get('frequency_days', 'Unknown'),
                    'confidence_threshold': ml_config.get('prediction', {}).get('confidence_threshold', 'Unknown')
                },
                'trading_config': {
                    'position_size': self.config.get('trading.position_sizing.default_percentage', 'Unknown'),
                    'stop_loss': self.config.get('trading.risk_management.stop_loss_percentage', 'Unknown'),
                    'take_profit': self.config.get('trading.risk_management.take_profit_percentage', 'Unknown')
                },
                'analysis_timestamp': dt.datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Analysis completed for {symbol}")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Analysis failed for {symbol}: {str(e)}")
            return {'error': str(e)}
    
    def create_trading_chart(self, symbol: str = None, period: str = "6mo") -> Optional[str]:
        """
        Create a comprehensive TradingView-style chart for a symbol
        
        Args:
            symbol: Trading symbol (defaults to config default)
            period: Time period for analysis
            
        Returns:
            Path to the saved chart file or None if failed
        """
        if symbol is None:
            symbol = self.config.get('trading.default_symbol', 'AAPL')
        
        try:
            self.logger.info(f"📊 Creating TradingView chart for {symbol}")
            
            # Create comprehensive chart
            chart_path = self.chart_generator.create_comprehensive_chart(symbol, period)
            
            self.logger.info(f"✅ Chart created successfully: {chart_path}")
            return chart_path
            
        except Exception as e:
            self.logger.error(f"❌ Chart creation failed for {symbol}: {str(e)}")
            return None


def main():
    """Main function to demonstrate the modular trading system"""
    try:
        print("🚀 Starting ML Trading System...")
        
        # Initialize the trading system
        trading_system = TradingSystemOrchestrator()
        
        # Check system status
        status = trading_system.get_system_status()
        print("\n🔍 System Status:")
        print(f"Configuration Loaded: {status['configuration_loaded']}")
        print(f"Primary Algorithm: {status['primary_algorithm']}")
        print(f"Default Symbol: {status['default_symbol']}")
        print(f"Supported Assets: {status['supported_assets']}")
        print(f"Environment: {status['environment']}")
        
        # Run analysis on default symbol
        print(f"\n🚀 Running analysis for default symbol...")
        results = trading_system.run_simple_analysis()
        
        if 'error' not in results:
            print("\n📊 Analysis Results:")
            print(f"Symbol: {results['symbol']}")
            print(f"Asset Name: {results['asset_info']['name']}")
            print(f"Sector: {results['asset_info']['sector']}")
            print(f"Volatility Class: {results['asset_info']['volatility_class']}")
            print(f"Primary Algorithm: {results['ml_configuration']['primary_algorithm']}")
            print(f"Retraining Frequency: {results['ml_configuration']['retraining_frequency']} days")
            print(f"Confidence Threshold: {results['ml_configuration']['confidence_threshold']}")
            print(f"Position Size: {results['trading_config']['position_size'] * 100}%")
            print(f"Stop Loss: {results['trading_config']['stop_loss'] * 100}%")
            print(f"Take Profit: {results['trading_config']['take_profit'] * 100}%")
        else:
            print(f"❌ Analysis failed: {results['error']}")
        
        # Test all configured assets
        print(f"\n📋 Testing all configured assets:")
        assets = trading_system.config.get('trading.assets.stocks', [])
        for asset in assets:
            symbol = asset['symbol']
            result = trading_system.run_simple_analysis(symbol)
            if 'error' not in result:
                print(f"✅ {symbol} ({asset['name']}): {asset['volatility_class']} volatility")
            else:
                print(f"❌ {symbol}: {result['error']}")
        
        # Create comprehensive charts for key assets
        print(f"\n📊 Creating TradingView-style charts...")
        
        # Create chart for default symbol
        default_symbol = trading_system.config.get('trading.default_symbol', 'AAPL')
        print(f"📈 Creating chart for {default_symbol}...")
        chart_path = trading_system.create_trading_chart(default_symbol)
        
        if chart_path:
            print(f"✅ Chart created: {chart_path}")
            print(f"🔗 Open the chart in your browser to view the TradingView-style analysis!")
        
        # Create charts for other high-volatility assets
        high_vol_assets = [asset['symbol'] for asset in assets if asset.get('volatility_class') == 'high']
        for symbol in high_vol_assets[:2]:  # Limit to 2 additional charts
            print(f"📈 Creating chart for {symbol}...")
            chart_path = trading_system.create_trading_chart(symbol)
            if chart_path:
                print(f"✅ Chart created for {symbol}")
        
        print(f"\n🎉 System demonstration completed successfully!")
        print(f"📁 Check the 'results' folder for all generated charts!")
        
    except Exception as e:
        print(f"❌ System initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()