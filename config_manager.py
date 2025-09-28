"""
Configuration Manager for ML Trading Application
Handles loading and accessing configuration from YAML file
"""

import yaml
import os
import datetime as dt
from typing import Dict, List, Any, Optional

class ConfigManager:
    """Manages configuration settings from YAML file"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration manager
        
        Args:
            config_path (str): Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
    
    def reload_config(self):
        """Reload configuration from file"""
        self.config = self._load_config()
    
    # Trading Configuration
    def get_symbols(self) -> List[str]:
        """Get list of trading symbols"""
        return self.config.get('trading', {}).get('symbols', ['TSLA'])
    
    def get_default_symbol(self) -> str:
        """Get default symbol for analysis"""
        return self.config.get('trading', {}).get('default_symbol', 'TSLA')
    
    def get_shares_per_trade(self) -> int:
        """Get number of shares per trade"""
        return self.config.get('trading', {}).get('shares_per_trade', 20)
    
    # Model Management Configuration
    def get_model_management_config(self) -> Dict[str, Any]:
        """Get model management configuration"""
        return self.config.get('model_management', {})
    
    def get_models_base_path(self) -> str:
        """Get base path for model storage"""
        return self.config.get('model_management', {}).get('models_base_path', 'models')
    
    def get_max_versions_per_symbol(self) -> int:
        """Get maximum versions to keep per symbol"""
        return self.config.get('model_management', {}).get('max_versions_per_symbol', 5)
    
    def should_auto_save_models(self) -> bool:
        """Check if models should be automatically saved"""
        return self.config.get('model_management', {}).get('auto_save_models', True)
    
    def should_use_model_cache(self) -> bool:
        """Check if model cache should be used"""
        return self.config.get('model_management', {}).get('use_model_cache', True)
    
    def should_auto_cleanup(self) -> bool:
        """Check if old model versions should be automatically cleaned up"""
        return self.config.get('model_management', {}).get('auto_cleanup', True)
    
    def get_prediction_service_config(self) -> Dict[str, Any]:
        """Get prediction service configuration"""
        return self.config.get('model_management', {}).get('prediction_service', {})
    
    def is_prediction_service_enabled(self) -> bool:
        """Check if prediction service is enabled"""
        return self.config.get('model_management', {}).get('prediction_service', {}).get('enabled', True)
    
    def get_confidence_threshold(self) -> float:
        """Get minimum confidence threshold for predictions"""
        return self.config.get('model_management', {}).get('prediction_service', {}).get('confidence_threshold', 0.7)
    
    def get_retraining_config(self) -> Dict[str, Any]:
        """Get model retraining configuration"""
        return self.config.get('model_management', {}).get('retraining', {})
    
    def get_market_impact(self) -> float:
        """Get market impact percentage"""
        return self.config.get('trading', {}).get('market_impact', 0.005)
    
    # Data Configuration
    def get_training_period_months(self) -> int:
        """Get training period in months"""
        return self.config.get('data', {}).get('training_period_months', 12)
    
    def get_simulation_dates(self) -> tuple:
        """Get simulation start and end dates"""
        sim_config = self.config.get('data', {}).get('simulation', {})
        start_str = sim_config.get('start_date', '2025-07-01')
        end_str = sim_config.get('end_date', '2025-12-31')
        
        start_date = dt.datetime.strptime(start_str, '%Y-%m-%d')
        end_date = dt.datetime.strptime(end_str, '%Y-%m-%d')
        
        return start_date, end_date
    
    def get_analysis_dates(self, analysis_type: str) -> tuple:
        """
        Get analysis dates for specific analysis type
        
        Args:
            analysis_type (str): Type of analysis (tesla_analysis, tqqq_analysis, etc.)
        
        Returns:
            tuple: (train_start, train_end, test_start, test_end)
        """
        analysis_config = self.config.get('data', {}).get('analysis', {}).get(analysis_type, {})
        
        train_start = dt.datetime.strptime(analysis_config.get('train_start', '2020-01-01'), '%Y-%m-%d')
        train_end = dt.datetime.strptime(analysis_config.get('train_end', '2025-07-01'), '%Y-%m-%d') 
        test_start = dt.datetime.strptime(analysis_config.get('test_start', '2025-07-01'), '%Y-%m-%d')
        test_end = dt.datetime.strptime(analysis_config.get('test_end', '2025-12-31'), '%Y-%m-%d')
        
        return train_start, train_end, test_start, test_end
    
    def get_lookback_days(self) -> int:
        """Get lookback days for live trading data"""
        return self.config.get('data', {}).get('live_trading', {}).get('lookback_days', 30)
    
    # Technical Indicators Configuration
    def get_indicator_window(self) -> int:
        """Get window size for technical indicators"""
        return self.config.get('indicators', {}).get('window', 5)
    
    def get_indicator_settings(self) -> Dict[str, bool]:
        """Get which indicators to use"""
        indicators_config = self.config.get('indicators', {})
        return {
            'use_sma': indicators_config.get('use_sma', True),
            'use_bollinger_bands': indicators_config.get('use_bollinger_bands', True),
            'use_momentum': indicators_config.get('use_momentum', True),
            'use_volatility': indicators_config.get('use_volatility', True),
            'use_macd': indicators_config.get('use_macd', True),
            'drop_upper_bb': indicators_config.get('drop_upper_bb', True),
            'drop_lower_bb': indicators_config.get('drop_lower_bb', True),
            'keep_bb_value': indicators_config.get('keep_bb_value', True),
            # Market context indicators
            'use_market_context': indicators_config.get('use_market_context', False),
            'market_indices': indicators_config.get('market_indices', ['SPY', 'QQQ']),
            'use_spy_trend': indicators_config.get('use_spy_trend', True),
            'use_qqq_trend': indicators_config.get('use_qqq_trend', True),
            'use_relative_strength': indicators_config.get('use_relative_strength', True),
            'use_beta_features': indicators_config.get('use_beta_features', True),
            'use_market_volatility_regime': indicators_config.get('use_market_volatility_regime', True),
            'use_sector_rotation': indicators_config.get('use_sector_rotation', True),
            'use_market_breadth': indicators_config.get('use_market_breadth', True)
        }
    
    # ML Model Configuration
    def get_ml_config(self) -> Dict[str, Any]:
        """Get ML model configuration"""
        ml_config = self.config.get('ml_model', {})
        return {
            'algorithm': ml_config.get('algorithm', 'DecisionTree'),
            'max_depth': ml_config.get('max_depth', 5),
            'random_state': ml_config.get('random_state', 42),
            'n_estimators': ml_config.get('n_estimators', 100),
            'min_training_samples': ml_config.get('min_training_samples', 100),
            'min_labels_required': ml_config.get('min_labels_required', 20)
        }
    
    def get_retraining_config(self) -> Dict[str, Any]:
        """Get retraining configuration"""
        retraining_config = self.config.get('ml_model', {}).get('retraining', {})
        return {
            'default_frequency': retraining_config.get('default_frequency', 10),
            'high_volatility_frequency': retraining_config.get('high_volatility_frequency', 5),
            'medium_volatility_frequency': retraining_config.get('medium_volatility_frequency', 10),
            'low_volatility_frequency': retraining_config.get('low_volatility_frequency', 22),
            'high_volatility_threshold': retraining_config.get('high_volatility_threshold', 0.8),
            'medium_volatility_threshold': retraining_config.get('medium_volatility_threshold', 0.4),
            'accuracy_threshold': retraining_config.get('accuracy_threshold', 0.30)
        }
    
    # Portfolio Configuration
    def get_portfolio_config(self) -> Dict[str, Any]:
        """Get portfolio configuration"""
        portfolio_config = self.config.get('portfolio', {})
        return {
            'starting_value': portfolio_config.get('starting_value', 3000),
            'commission': portfolio_config.get('commission', 0.0),
            'impact': portfolio_config.get('impact', 0.005),
            'max_position_size': portfolio_config.get('max_position_size', 20)
        }
    
    # Robinhood Configuration
    def get_robinhood_config(self) -> Dict[str, Any]:
        """Get Robinhood configuration"""
        rh_config = self.config.get('robinhood', {})
        return {
            'username': rh_config.get('username', ''),
            'password': rh_config.get('password', ''),
            'qr_code': rh_config.get('qr_code', ''),
            'order_type': rh_config.get('order_type', 'market'),
            'time_in_force': rh_config.get('time_in_force', 'GFD'),
            'enable_live_trading': rh_config.get('enable_live_trading', False),
            'paper_trading': rh_config.get('paper_trading', True)
        }
    
    # Analysis Configuration
    def get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis configuration"""
        analysis_config = self.config.get('analysis', {})
        return {
            'calculate_sharpe': analysis_config.get('calculate_sharpe', True),
            'calculate_returns': analysis_config.get('calculate_returns', True),
            'calculate_volatility': analysis_config.get('calculate_volatility', True),
            'calculate_max_drawdown': analysis_config.get('calculate_max_drawdown', True),
            'generate_plots': analysis_config.get('generate_plots', True),
            'plot_strategy_performance': analysis_config.get('plot_strategy_performance', True),
            'save_plots': analysis_config.get('save_plots', True),
            'plot_directory': analysis_config.get('plot_directory', 'plots'),
            'generate_reports': analysis_config.get('generate_reports', True),
            'report_directory': analysis_config.get('report_directory', 'reports'),
            'log_level': analysis_config.get('log_level', 'INFO'),
            'log_file': analysis_config.get('log_file', 'trading.log')
        }
    
    # Utility Methods
    def get_position_filename(self, symbol: str) -> str:
        """Get position tracking filename for a symbol"""
        extension = self.config.get('trading', {}).get('position_file_extension', '.txt')
        return f"{symbol}{extension}"
    
    def get_training_dates_for_symbol(self, symbol: str, test_start_date: dt.datetime) -> tuple:
        """
        Calculate training start and end dates based on training period
        
        Args:
            symbol (str): Trading symbol
            test_start_date (dt.datetime): When testing/live trading starts
            
        Returns:
            tuple: (train_start_date, train_end_date)
        """
        training_months = self.get_training_period_months()
        train_start = test_start_date - dt.timedelta(days=training_months * 30)
        train_end = test_start_date
        
        return train_start, train_end
    
    def get_retraining_frequency_for_symbol(self, symbol: str, current_volatility: Optional[float] = None) -> int:
        """
        Get retraining frequency for a specific symbol based on volatility
        
        Args:
            symbol (str): Trading symbol
            current_volatility (float, optional): Current volatility measure
            
        Returns:
            int: Retraining frequency in days
        """
        retraining_config = self.get_retraining_config()
        
        if current_volatility is None:
            return retraining_config['default_frequency']
        
        if current_volatility > retraining_config['high_volatility_threshold']:
            return retraining_config['high_volatility_frequency']
        elif current_volatility > retraining_config['medium_volatility_threshold']:
            return retraining_config['medium_volatility_frequency']
        else:
            return retraining_config['low_volatility_frequency']
    
    def is_experimental_feature_enabled(self, feature_name: str) -> bool:
        """Check if an experimental feature is enabled"""
        experimental_config = self.config.get('experimental', {})
        return experimental_config.get(feature_name, False)
    
    def update_config(self, section: str, key: str, value: Any):
        """
        Update a configuration value and save to file
        
        Args:
            section (str): Configuration section
            key (str): Configuration key
            value (Any): New value
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
        
        # Save back to file
        with open(self.config_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get the complete configuration dictionary"""
        return self.config.copy()

# Global configuration instance
config = ConfigManager()

def get_config() -> ConfigManager:
    """Get the global configuration instance"""
    return config