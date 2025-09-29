# ML Automated Trading System - Class Diagram

```mermaid
classDiagram
    %% ====================================
    %% INTERFACES LAYER (Abstract Base Classes)
    %% ====================================
    
    class DataProvider {
        <<abstract>>
        +get_historical_data(symbol, start_date, end_date)* pd.DataFrame
        +get_market_data(symbols, start_date, end_date)* Dict[str, DataFrame]
        +get_current_price(symbol)* float
        +is_available()* bool
        +validate_symbol(symbol)* bool
    }

    class SignalGenerator {
        <<abstract>>
        -name: str
        -config: Dict[str, Any]
        -enabled: bool
        +generate_signals(data, symbol)* List[TradingSignal]
        +get_required_columns()* List[str]
        +validate_data(data)* bool
        +is_enabled() bool
        +get_name() str
        +get_config() Dict[str, Any]
    }

    class TradingStrategy {
        <<abstract>>
        -name: str
        -config: Dict[str, Any]
        -positions: Dict[str, Position]
        -orders: List[Order]
        +generate_orders(signals, market_data, positions)* List[Order]
        +calculate_position_size(signal, price, account_value)* int
        +validate_order(order, market_data, positions, account_value)* bool
        +get_name() str
        +get_config() Dict[str, Any]
    }

    class ModelManagerInterface {
        <<abstract>>
        +train_model(features, labels, model_type)* BaseEstimator
        +predict(model, features)* np.ndarray
        +save_model(model, model_name, metadata)*
        +load_model(model_name)* BaseEstimator
        +evaluate_model(model, test_features, test_labels)* ModelMetadata
        +get_model_metadata(model_name)* ModelMetadata
        +list_models()* List[str]
    }

    class RiskManager {
        <<abstract>>
        -name: str
        -config: Dict[str, Any]
        +assess_portfolio_risk(positions, portfolio_value, market_data)* RiskMetrics
        +validate_orders(orders, positions, portfolio_value, market_data)* RiskAssessment
        +calculate_position_size_limit(symbol, price, portfolio_value, positions)* int
        +check_concentration_risk(positions, portfolio_value)* float
    }

    class Backtester {
        <<abstract>>
        -name: str
        -config: BacktestConfig
        -results: BacktestResults
        +run_backtest(strategy, market_data)* BacktestResults
        +simulate_order_execution(order, market_data, timestamp)* Dict[str, Any]
        +calculate_portfolio_value(positions, cash, market_data, timestamp)* float
        +calculate_performance_metrics(portfolio_values, benchmark_data)* StrategyPerformance
    }

    %% ====================================
    %% DATA CLASSES & ENUMS
    %% ====================================
    
    class TradingSignal {
        <<dataclass>>
        +symbol: str
        +timestamp: pd.Timestamp
        +signal_type: SignalType
        +confidence: float
        +strength: float
        +price: float
        +source: str
        +metadata: Dict[str, Any]
    }

    class SignalType {
        <<enumeration>>
        BUY = 1
        SELL = -1
        HOLD = 0
    }

    class Order {
        <<dataclass>>
        +symbol: str
        +side: OrderSide
        +quantity: int
        +order_type: OrderType
        +price: Optional[float]
        +timestamp: pd.Timestamp
        +metadata: Dict[str, Any]
    }

    class Position {
        <<dataclass>>
        +symbol: str
        +quantity: int
        +average_cost: float
        +current_price: float
        +market_value: float
        +unrealized_pnl: float
        +last_updated: pd.Timestamp
    }

    class OrderSizingStrategy {
        <<enumeration>>
        FIXED = "fixed"
        PERCENTAGE = "percentage"
        VOLATILITY_ADJUSTED = "volatility_adjusted"
        KELLY_CRITERION = "kelly_criterion"
        RISK_PARITY = "risk_parity"
    }

    class OrderSizingConfig {
        <<dataclass>>
        +strategy: OrderSizingStrategy
        +fixed_shares: int
        +portfolio_pct: float
        +min_shares: int
        +max_shares: int
        +volatility_window: int
        +volatility_target: float
        +adjustment_factor: float
        +base_shares: int
    }

    %% ====================================
    %% DATA PROVIDERS (Concrete Implementations)
    %% ====================================
    
    class YFinanceProvider {
        -config: Dict[str, Any]
        -session: Optional[Session]
        +get_historical_data(symbol, start_date, end_date) pd.DataFrame
        +get_market_data(symbols, start_date, end_date) Dict[str, DataFrame]
        +get_current_price(symbol) float
        +is_available() bool
        +validate_symbol(symbol) bool
    }

    class RobinhoodProvider {
        -config: Dict[str, Any]
        -api_key: str
        +get_historical_data(symbol, start_date, end_date) pd.DataFrame
        +get_market_data(symbols, start_date, end_date) Dict[str, DataFrame]
        +get_current_price(symbol) float
        +is_available() bool
        +validate_symbol(symbol) bool
    }

    %% ====================================
    %% SIGNAL GENERATORS (Concrete Implementations)
    %% ====================================
    
    class RSISignalGenerator {
        -period: int
        -overbought_threshold: float
        -oversold_threshold: float
        +generate_signals(data, symbol) List[TradingSignal]
        +get_required_columns() List[str]
        +validate_data(data) bool
    }

    class MACDSignalGenerator {
        -fast_period: int
        -slow_period: int
        -signal_period: int
        +generate_signals(data, symbol) List[TradingSignal]
        +get_required_columns() List[str]
        +validate_data(data) bool
    }

    class BollingerBandsSignalGenerator {
        -period: int
        -std_dev: float
        +generate_signals(data, symbol) List[TradingSignal]
        +get_required_columns() List[str]
        +validate_data(data) bool
    }

    class GoldenCrossSignalGenerator {
        -short_window: int
        -long_window: int
        -strength_threshold: float
        -confirmation_days: int
        +generate_signals(data, symbol) List[TradingSignal]
        +get_required_columns() List[str]
        +validate_data(data) bool
    }

    class ShortTermPatternSignalGenerator {
        -rsi_period: int
        -bb_period: int
        -bb_std: float
        +generate_signals(data, symbol) List[TradingSignal]
        +get_required_columns() List[str]
        +validate_data(data) bool
    }

    class MLSignalGenerator {
        -model_manager: EnhancedModelManager
        -model_name: str
        -feature_columns: List[str]
        +generate_signals(data, symbol) List[TradingSignal]
        +get_required_columns() List[str]
        +validate_data(data) bool
    }

    %% ====================================
    %% TRADING STRATEGIES (Concrete Implementations)
    %% ====================================
    
    class AutoOrderSizeManager {
        -config: OrderSizingConfig
        -starting_portfolio_value: float
        -current_portfolio_value: float
        -positions: Dict[str, Position]
        +calculate_order_size(symbol, signal, market_data) int
        +add_position(symbol, position)
        +get_position_size(symbol) int
        +update_portfolio_value(new_value)
        -_get_current_price(symbol) float
        -_detect_market_condition(data) str
    }

    class EnhancedMLTradingStrategy {
        -symbol: str
        -data_provider: YFinanceProvider
        -model_manager: ModelManagerInterface
        -order_size_manager: AutoOrderSizeManager
        -signal_generators: List[SignalGenerator]
        -starting_portfolio_value: float
        +generate_orders(signals, market_data, positions) List[Order]
        +calculate_position_size(signal, price, account_value) int
        +validate_order(order, market_data, positions, account_value) bool
        +generate_signal(data, timestamp) TradingSignal
        -_initialize_signal_generators(golden_cross_config, short_term_config)
    }

    %% ====================================
    %% MODEL MANAGEMENT
    %% ====================================
    
    class EnhancedModelManager {
        -base_path: str
        -models: Dict[str, BaseEstimator]
        -metadata: Dict[str, ModelMetadata]
        +train_model(features, labels, model_type) BaseEstimator
        +predict(model, features) np.ndarray
        +save_model(model, model_name, metadata)
        +load_model(model_name) BaseEstimator
        +evaluate_model(model, test_features, test_labels) ModelMetadata
        +get_model_metadata(model_name) ModelMetadata
        +list_models() List[str]
    }

    class ModelTrainingService {
        -model_manager: EnhancedModelManager
        -data_provider: DataProvider
        +train_and_evaluate_models(symbol, start_date, end_date) Dict[str, ModelMetadata]
        +retrain_model(model_name, symbol, start_date, end_date) ModelMetadata
        -_prepare_features_and_labels(data, symbol) Tuple[DataFrame, Series]
    }

    %% ====================================
    %% BACKTESTING
    %% ====================================
    
    class EnhancedBacktester {
        -data_provider: DataProvider
        -initial_cash: float
        -commission_rate: float
        -slippage_rate: float
        -risk_manager: Optional[RiskManager]
        -portfolio: Portfolio
        +run_backtest(strategy, symbols, start_date, end_date) BacktestResult
        +simulate_order_execution(order, market_data, timestamp) Dict[str, Any]
        +calculate_portfolio_value(positions, cash, market_data, timestamp) float
        +calculate_performance_metrics(portfolio_values, benchmark_data) StrategyPerformance
        -_execute_strategy_signal(symbol, signal, data) Optional[Dict[str, Any]]
    }

    class Portfolio {
        -cash: float
        -positions: Dict[str, Position]
        -transaction_history: List[Dict[str, Any]]
        +add_position(symbol, quantity, price)
        +remove_position(symbol, quantity, price)
        +get_total_value(market_data) float
        +get_position(symbol) Position
        +get_cash() float
    }

    %% ====================================
    %% ANALYSIS & FEATURES
    %% ====================================
    
    class MarketContextAnalyzer {
        -data_provider: DataProvider
        +analyze_market_regime(spy_data, qqq_data) Dict[str, Any]
        +calculate_market_correlation(symbol_data, market_data) float
        +detect_volatility_regime(data) str
        +get_sector_rotation_signals(data) Dict[str, Any]
    }

    class EnhancedFeatureEngineer {
        -market_analyzer: MarketContextAnalyzer
        +create_features(data, symbol) pd.DataFrame
        +add_technical_indicators(data) pd.DataFrame
        +add_market_context_features(data, symbol) pd.DataFrame
        +add_volatility_features(data) pd.DataFrame
    }

    class DataPreprocessor {
        +clean_data(data, symbol) pd.DataFrame
        +handle_missing_values(data) pd.DataFrame
        +normalize_prices(data) pd.DataFrame
        +resample_data(data, frequency) pd.DataFrame
    }

    class TechnicalIndicatorCalculator {
        +calculate_all_indicators(data) pd.DataFrame
        +calculate_rsi(data, period) pd.Series
        +calculate_macd(data) Tuple[pd.Series, pd.Series, pd.Series]
        +calculate_bollinger_bands(data, period, std_dev) Tuple[pd.Series, pd.Series, pd.Series]
        +calculate_moving_averages(data, periods) pd.DataFrame
    }

    %% ====================================
    %% ORCHESTRATORS (Main System Controllers)
    %% ====================================
    
    class TradingSystemOrchestrator {
        -logger: logging.Logger
        -config: Dict[str, Any]
        -data_provider: DataProvider
        -preprocessor: DataPreprocessor
        -indicator_calculator: TechnicalIndicatorCalculator
        -signal_generators: List[SignalGenerator]
        +run_analysis(symbol, start_date, end_date) Dict[str, Any]
        +get_system_status() Dict[str, Any]
        -_initialize_components()
        -_create_data_provider() DataProvider
        -_create_signal_generators() List[SignalGenerator]
        -_get_market_data(symbol, start_date, end_date) pd.DataFrame
        -_preprocess_data(data, symbol) pd.DataFrame
        -_calculate_indicators(data) pd.DataFrame
        -_generate_signals(data, symbol) List[TradingSignal]
        -_analyze_results(symbol, data, indicators, signals) Dict[str, Any]
    }

    class EnhancedTradingSystemOrchestrator {
        -data_provider: YFinanceProvider
        -model_manager: EnhancedModelManager
        -model_training_service: ModelTrainingService
        -market_analyzer: MarketContextAnalyzer
        -feature_engineer: EnhancedFeatureEngineer
        -backtester: EnhancedBacktester
        -starting_capital: float
        -commission_rate: float
        -slippage_rate: float
        +run_enhanced_analysis(symbols, start_date, end_date) Dict[str, Any]
        +run_backtest_with_ml(symbols, start_date, end_date) BacktestResult
        +train_models_for_symbols(symbols, start_date, end_date) Dict[str, ModelMetadata]
        -_initialize_components(models_path)
        -_create_enhanced_strategy(symbol) EnhancedMLTradingStrategy
    }

    %% ====================================
    %% FACTORY PATTERN
    %% ====================================
    
    class TradingSystemFactory {
        -data_provider_factory: DataProviderFactory
        -signal_generator_factory: SignalGeneratorFactory
        -trading_strategy_factory: TradingStrategyFactory
        -backtester_factory: BacktesterFactory
        -risk_manager_factory: RiskManagerFactory
        +create_data_provider(provider_type, config) DataProvider
        +create_signal_generator(generator_type, config) SignalGenerator
        +create_trading_strategy(strategy_type, config) TradingStrategy
        +create_backtester(backtester_type, config) Backtester
        +create_risk_manager(manager_type, config) RiskManager
        +get_available_components() Dict[str, List[str]]
    }

    class ComponentFactory {
        <<abstract>>
        +create(component_type, config)* object
        +get_supported_types()* List[str]
        +get_default_config(component_type)* Dict[str, Any]
    }

    class DataProviderFactory {
        SUPPORTED_PROVIDERS: Dict[str, type]
        +create(provider_type, config) DataProvider
        +get_supported_types() List[str]
        +get_default_config(provider_type) Dict[str, Any]
    }

    class SignalGeneratorFactory {
        SUPPORTED_GENERATORS: Dict[str, type]
        +create(generator_type, config) SignalGenerator
        +create_multiple(generators_config) List[SignalGenerator]
        +get_supported_types() List[str]
        +get_default_config(generator_type) Dict[str, Any]
    }

    %% ====================================
    %% RELATIONSHIPS
    %% ====================================
    
    %% Interface Implementations
    DataProvider <|-- YFinanceProvider
    DataProvider <|-- RobinhoodProvider
    
    SignalGenerator <|-- RSISignalGenerator
    SignalGenerator <|-- MACDSignalGenerator  
    SignalGenerator <|-- BollingerBandsSignalGenerator
    SignalGenerator <|-- GoldenCrossSignalGenerator
    SignalGenerator <|-- ShortTermPatternSignalGenerator
    SignalGenerator <|-- MLSignalGenerator
    
    TradingStrategy <|-- EnhancedMLTradingStrategy
    ModelManagerInterface <|-- EnhancedModelManager
    Backtester <|-- EnhancedBacktester
    
    ComponentFactory <|-- DataProviderFactory
    ComponentFactory <|-- SignalGeneratorFactory
    
    %% Data Class Relationships
    SignalGenerator --> TradingSignal : creates
    TradingSignal --> SignalType : uses
    TradingStrategy --> Order : creates
    Order --> TradingSignal : based_on
    
    AutoOrderSizeManager --> OrderSizingStrategy : uses
    AutoOrderSizeManager --> OrderSizingConfig : uses
    AutoOrderSizeManager --> Position : manages
    
    %% Component Dependencies
    TradingSystemOrchestrator --> DataProvider : uses
    TradingSystemOrchestrator --> SignalGenerator : uses
    TradingSystemOrchestrator --> DataPreprocessor : uses
    TradingSystemOrchestrator --> TechnicalIndicatorCalculator : uses
    
    EnhancedTradingSystemOrchestrator --> YFinanceProvider : uses
    EnhancedTradingSystemOrchestrator --> EnhancedModelManager : uses
    EnhancedTradingSystemOrchestrator --> ModelTrainingService : uses
    EnhancedTradingSystemOrchestrator --> MarketContextAnalyzer : uses
    EnhancedTradingSystemOrchestrator --> EnhancedFeatureEngineer : uses
    EnhancedTradingSystemOrchestrator --> EnhancedBacktester : uses
    
    EnhancedMLTradingStrategy --> AutoOrderSizeManager : uses
    EnhancedMLTradingStrategy --> SignalGenerator : uses
    EnhancedMLTradingStrategy --> ModelManagerInterface : uses
    EnhancedMLTradingStrategy --> YFinanceProvider : uses
    
    EnhancedBacktester --> Portfolio : manages
    EnhancedBacktester --> DataProvider : uses
    EnhancedBacktester --> RiskManager : uses
    
    ModelTrainingService --> EnhancedModelManager : uses
    ModelTrainingService --> DataProvider : uses
    
    EnhancedFeatureEngineer --> MarketContextAnalyzer : uses
    MarketContextAnalyzer --> DataProvider : uses
    
    MLSignalGenerator --> EnhancedModelManager : uses
    
    %% Factory Relationships
    TradingSystemFactory --> DataProviderFactory : contains
    TradingSystemFactory --> SignalGeneratorFactory : contains
    DataProviderFactory --> YFinanceProvider : creates
    DataProviderFactory --> RobinhoodProvider : creates
    SignalGeneratorFactory --> RSISignalGenerator : creates
    SignalGeneratorFactory --> MACDSignalGenerator : creates
    SignalGeneratorFactory --> BollingerBandsSignalGenerator : creates

    %% Styling
    classDef interface fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef concrete fill:#f3e5f5,stroke:#4a148c,stroke-width:1px
    classDef dataclass fill:#e8f5e8,stroke:#2e7d32,stroke-width:1px
    classDef enum fill:#fff3e0,stroke:#ef6c00,stroke-width:1px
    classDef orchestrator fill:#fff8e1,stroke:#f57f17,stroke-width:2px
    classDef factory fill:#fce4ec,stroke:#c2185b,stroke-width:2px

    class DataProvider,SignalGenerator,TradingStrategy,ModelManagerInterface,RiskManager,Backtester,ComponentFactory interface
    class YFinanceProvider,RSISignalGenerator,MACDSignalGenerator,BollingerBandsSignalGenerator,GoldenCrossSignalGenerator,ShortTermPatternSignalGenerator,MLSignalGenerator,EnhancedMLTradingStrategy,EnhancedModelManager,EnhancedBacktester,AutoOrderSizeManager,MarketContextAnalyzer,EnhancedFeatureEngineer,DataPreprocessor,TechnicalIndicatorCalculator,Portfolio,ModelTrainingService concrete
    class TradingSignal,Order,Position,OrderSizingConfig dataclass
    class SignalType,OrderSizingStrategy enum
    class TradingSystemOrchestrator,EnhancedTradingSystemOrchestrator orchestrator
    class TradingSystemFactory,DataProviderFactory,SignalGeneratorFactory factory
```

## Architecture Overview

### **1. Interface Layer (Abstract Base Classes)**
- **DataProvider**: Abstract interface for market data sources
- **SignalGenerator**: Abstract interface for generating trading signals
- **TradingStrategy**: Abstract interface for trading strategies
- **ModelManagerInterface**: Abstract interface for ML model management
- **RiskManager**: Abstract interface for risk management
- **Backtester**: Abstract interface for backtesting engines

### **2. Implementation Layer**
- **Data Providers**: YFinanceProvider, RobinhoodProvider
- **Signal Generators**: RSI, MACD, Bollinger Bands, Golden Cross, Short-term Pattern, ML-based
- **Trading Strategies**: EnhancedMLTradingStrategy with intelligent order sizing
- **Model Management**: EnhancedModelManager, ModelTrainingService
- **Analysis**: MarketContextAnalyzer, EnhancedFeatureEngineer
- **Backtesting**: EnhancedBacktester with comprehensive portfolio management

### **3. Orchestration Layer**
- **TradingSystemOrchestrator**: Basic system coordination
- **EnhancedTradingSystemOrchestrator**: Advanced system with ML and market analysis

### **4. Factory Pattern**
- **TradingSystemFactory**: Master factory for all components
- **Specialized Factories**: DataProviderFactory, SignalGeneratorFactory, etc.

### **5. Key Design Patterns**
- **Strategy Pattern**: Interchangeable algorithms (SignalGenerator, TradingStrategy)
- **Factory Pattern**: Component creation with proper configuration
- **Dependency Injection**: Orchestrators inject dependencies into components
- **Observer Pattern**: Signal generation and consumption
- **Template Method**: Abstract base classes define workflows

### **6. Data Flow**
1. **DataProvider** → Market Data → **DataPreprocessor**
2. **SignalGenerator** → Trading Signals → **TradingStrategy**
3. **TradingStrategy** → Orders → **Backtester**
4. **ModelManager** → ML Predictions → **MLSignalGenerator**
5. **MarketAnalyzer** → Context → **EnhancedFeatureEngineer**

This architecture provides a highly modular, extensible, and testable trading system with clear separation of concerns and proper abstraction layers.