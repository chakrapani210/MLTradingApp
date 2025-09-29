"""
Generate Visual Architecture Diagram for ML Trading System
Creates a comprehensive visual representation of all component relationships
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Rectangle
import numpy as np

def create_architecture_diagram():
    """Create a comprehensive architecture diagram"""
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'interface': '#E3F2FD',      # Light Blue
        'concrete': '#F3E5F5',       # Light Purple
        'dataclass': '#E8F5E8',      # Light Green
        'enum': '#FFF3E0',           # Light Orange
        'orchestrator': '#FFF8E1',   # Light Yellow
        'factory': '#FCE4EC'         # Light Pink
    }
    
    edge_colors = {
        'interface': '#1976D2',
        'concrete': '#7B1FA2',
        'dataclass': '#388E3C',
        'enum': '#F57C00',
        'orchestrator': '#F9A825',
        'factory': '#C2185B'
    }
    
    def draw_component(x, y, width, height, text, comp_type, fontsize=8):
        """Draw a component box with text"""
        # Draw rectangle
        rect = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.1",
            facecolor=colors[comp_type],
            edgecolor=edge_colors[comp_type],
            linewidth=1.5
        )
        ax.add_patch(rect)
        
        # Add text
        ax.text(x + width/2, y + height/2, text, 
                ha='center', va='center', fontsize=fontsize, fontweight='bold')
        
        return x + width/2, y + height/2  # Return center point
    
    def draw_arrow(start_x, start_y, end_x, end_y, style='->', color='black', alpha=0.7):
        """Draw arrow between components"""
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle=style, color=color, alpha=alpha, lw=1))
    
    # Title
    ax.text(10, 15.5, 'ML Automated Trading System - Architecture Overview', 
            ha='center', va='center', fontsize=20, fontweight='bold')
    
    # =============================================
    # INTERFACE LAYER (Top)
    # =============================================
    interface_y = 13.5
    
    # Core Interfaces
    dp_center = draw_component(1, interface_y, 2.5, 1, 'DataProvider\n<<interface>>', 'interface')
    sg_center = draw_component(4, interface_y, 2.5, 1, 'SignalGenerator\n<<interface>>', 'interface')
    ts_center = draw_component(7, interface_y, 2.5, 1, 'TradingStrategy\n<<interface>>', 'interface')
    mm_center = draw_component(10, interface_y, 2.5, 1, 'ModelManager\n<<interface>>', 'interface')
    rm_center = draw_component(13, interface_y, 2.5, 1, 'RiskManager\n<<interface>>', 'interface')
    bt_center = draw_component(16, interface_y, 2.5, 1, 'Backtester\n<<interface>>', 'interface')
    
    # =============================================
    # DATA CLASSES & ENUMS
    # =============================================
    data_y = 12
    
    # Data classes
    signal_center = draw_component(1, data_y, 2, 0.8, 'TradingSignal', 'dataclass', 7)
    order_center = draw_component(3.5, data_y, 1.8, 0.8, 'Order', 'dataclass', 7)
    position_center = draw_component(5.5, data_y, 1.8, 0.8, 'Position', 'dataclass', 7)
    config_center = draw_component(7.5, data_y, 2.2, 0.8, 'OrderSizingConfig', 'dataclass', 7)
    
    # Enums
    signal_type_center = draw_component(10.5, data_y, 1.8, 0.8, 'SignalType', 'enum', 7)
    order_strategy_center = draw_component(12.5, data_y, 2.2, 0.8, 'OrderSizingStrategy', 'enum', 7)
    
    # =============================================
    # DATA PROVIDERS LAYER
    # =============================================
    provider_y = 10.5
    
    yf_center = draw_component(0.5, provider_y, 2.2, 0.8, 'YFinanceProvider', 'concrete', 7)
    rh_center = draw_component(3, provider_y, 2.2, 0.8, 'RobinhoodProvider', 'concrete', 7)
    
    # Data processing
    preproc_center = draw_component(6, provider_y, 2.2, 0.8, 'DataPreprocessor', 'concrete', 7)
    tech_calc_center = draw_component(8.5, provider_y, 2.5, 0.8, 'TechnicalIndicator\nCalculator', 'concrete', 7)
    
    # =============================================
    # SIGNAL GENERATORS LAYER
    # =============================================
    signal_gen_y = 9
    
    rsi_center = draw_component(0.5, signal_gen_y, 1.8, 0.8, 'RSI\nGenerator', 'concrete', 7)
    macd_center = draw_component(2.5, signal_gen_y, 1.8, 0.8, 'MACD\nGenerator', 'concrete', 7)
    bb_center = draw_component(4.5, signal_gen_y, 1.8, 0.8, 'Bollinger\nBands', 'concrete', 7)
    gc_center = draw_component(6.5, signal_gen_y, 1.8, 0.8, 'Golden\nCross', 'concrete', 7)
    st_center = draw_component(8.5, signal_gen_y, 1.8, 0.8, 'Short-term\nPattern', 'concrete', 7)
    ml_signal_center = draw_component(10.5, signal_gen_y, 1.8, 0.8, 'ML Signal\nGenerator', 'concrete', 7)
    
    # =============================================
    # TRADING STRATEGIES & MANAGEMENT
    # =============================================
    strategy_y = 7.5
    
    order_mgr_center = draw_component(1, strategy_y, 2.5, 0.8, 'AutoOrderSize\nManager', 'concrete', 7)
    enhanced_strategy_center = draw_component(4, strategy_y, 2.8, 0.8, 'EnhancedML\nTradingStrategy', 'concrete', 7)
    
    # Model Management
    model_mgr_center = draw_component(7.5, strategy_y, 2.2, 0.8, 'Enhanced\nModelManager', 'concrete', 7)
    train_service_center = draw_component(10, strategy_y, 2.2, 0.8, 'ModelTraining\nService', 'concrete', 7)
    
    # Analysis
    market_analyzer_center = draw_component(12.5, strategy_y, 2.2, 0.8, 'MarketContext\nAnalyzer', 'concrete', 7)
    feature_eng_center = draw_component(15, strategy_y, 2.2, 0.8, 'Enhanced\nFeatureEngineer', 'concrete', 7)
    
    # =============================================
    # BACKTESTING LAYER
    # =============================================
    backtest_y = 6
    
    enhanced_bt_center = draw_component(2, backtest_y, 2.5, 0.8, 'Enhanced\nBacktester', 'concrete', 7)
    portfolio_center = draw_component(5, backtest_y, 2, 0.8, 'Portfolio', 'concrete', 7)
    
    # =============================================
    # ORCHESTRATORS LAYER
    # =============================================
    orchestrator_y = 4.5
    
    basic_orch_center = draw_component(2, orchestrator_y, 3, 1, 'TradingSystem\nOrchestrator', 'orchestrator', 8)
    enhanced_orch_center = draw_component(6, orchestrator_y, 3.5, 1, 'EnhancedTradingSystem\nOrchestrator', 'orchestrator', 8)
    
    # =============================================
    # FACTORY LAYER
    # =============================================
    factory_y = 2.5
    
    master_factory_center = draw_component(4, factory_y, 3, 0.8, 'TradingSystemFactory', 'factory', 8)
    dp_factory_center = draw_component(0.5, factory_y, 2.2, 0.8, 'DataProvider\nFactory', 'factory', 7)
    sg_factory_center = draw_component(8, factory_y, 2.2, 0.8, 'SignalGenerator\nFactory', 'factory', 7)
    
    # =============================================
    # DRAW RELATIONSHIPS
    # =============================================
    
    # Interface implementations (inheritance)
    inheritance_color = '#1976D2'
    
    # DataProvider implementations
    draw_arrow(yf_center[0], yf_center[1] + 0.4, dp_center[0] - 0.8, dp_center[1] - 0.5, 
               style='-|>', color=inheritance_color)
    draw_arrow(rh_center[0], rh_center[1] + 0.4, dp_center[0] - 0.3, dp_center[1] - 0.5, 
               style='-|>', color=inheritance_color)
    
    # SignalGenerator implementations
    for i, gen_center in enumerate([rsi_center, macd_center, bb_center, gc_center, st_center, ml_signal_center]):
        offset_x = -1 + i * 0.4
        draw_arrow(gen_center[0], gen_center[1] + 0.4, sg_center[0] + offset_x, sg_center[1] - 0.5, 
                   style='-|>', color=inheritance_color)
    
    # TradingStrategy implementation
    draw_arrow(enhanced_strategy_center[0], enhanced_strategy_center[1] + 0.8, 
               ts_center[0], ts_center[1] - 0.5, style='-|>', color=inheritance_color)
    
    # ModelManager implementation
    draw_arrow(model_mgr_center[0], model_mgr_center[1] + 0.8, 
               mm_center[0], mm_center[1] - 0.5, style='-|>', color=inheritance_color)
    
    # Backtester implementation
    draw_arrow(enhanced_bt_center[0], enhanced_bt_center[1] + 0.8, 
               bt_center[0] - 1, bt_center[1] - 0.5, style='-|>', color=inheritance_color)
    
    # Composition relationships (uses/contains)
    composition_color = '#7B1FA2'
    
    # Enhanced Strategy uses multiple components
    draw_arrow(enhanced_strategy_center[0] - 0.5, enhanced_strategy_center[1], 
               order_mgr_center[0] + 1, order_mgr_center[1], color=composition_color)
    
    # Model Training Service uses Model Manager
    draw_arrow(train_service_center[0] - 0.5, train_service_center[1], 
               model_mgr_center[0] + 1, model_mgr_center[1], color=composition_color)
    
    # Enhanced Backtester uses Portfolio
    draw_arrow(enhanced_bt_center[0] + 1, enhanced_bt_center[1], 
               portfolio_center[0] - 0.5, portfolio_center[1], color=composition_color)
    
    # Orchestrator relationships
    orchestrator_color = '#F9A825'
    
    # Basic Orchestrator uses components
    draw_arrow(basic_orch_center[0], basic_orch_center[1] + 0.5, 
               yf_center[0], yf_center[1] - 0.4, color=orchestrator_color)
    draw_arrow(basic_orch_center[0] + 0.5, basic_orch_center[1] + 0.5, 
               preproc_center[0], preproc_center[1] - 0.4, color=orchestrator_color)
    
    # Enhanced Orchestrator uses advanced components
    draw_arrow(enhanced_orch_center[0], enhanced_orch_center[1] + 0.5, 
               enhanced_strategy_center[0], enhanced_strategy_center[1] - 0.4, color=orchestrator_color)
    draw_arrow(enhanced_orch_center[0] + 1, enhanced_orch_center[1] + 0.5, 
               model_mgr_center[0], model_mgr_center[1] - 0.4, color=orchestrator_color)
    draw_arrow(enhanced_orch_center[0] - 1, enhanced_orch_center[1] + 0.5, 
               enhanced_bt_center[0], enhanced_bt_center[1] - 0.4, color=orchestrator_color)
    
    # Factory relationships
    factory_color = '#C2185B'
    
    # Master factory contains sub-factories
    draw_arrow(master_factory_center[0] - 1, master_factory_center[1], 
               dp_factory_center[0] + 1, dp_factory_center[1], color=factory_color)
    draw_arrow(master_factory_center[0] + 1, master_factory_center[1], 
               sg_factory_center[0] - 1, sg_factory_center[1], color=factory_color)
    
    # Data flow arrows (signal flow)
    data_flow_color = '#388E3C'
    
    # Signal generation flow
    draw_arrow(signal_center[0], signal_center[1] - 0.4, 
               rsi_center[0], rsi_center[1] + 0.4, style='<-', color=data_flow_color)
    
    # Order flow
    draw_arrow(order_center[0], order_center[1] - 0.4, 
               enhanced_strategy_center[0], enhanced_strategy_center[1] + 0.4, style='<-', color=data_flow_color)
    
    # =============================================
    # LEGEND
    # =============================================
    legend_y = 1
    
    # Component types legend
    ax.text(1, legend_y + 0.8, 'LEGEND:', fontsize=12, fontweight='bold')
    
    legend_items = [
        ('Interface', 'interface', 0.5),
        ('Concrete Implementation', 'concrete', 3),
        ('Data Class', 'dataclass', 6.5),
        ('Enum', 'enum', 8.5),
        ('Orchestrator', 'orchestrator', 10.5),
        ('Factory', 'factory', 13)
    ]
    
    for name, comp_type, x_pos in legend_items:
        draw_component(x_pos, legend_y, 1.8, 0.4, name, comp_type, 7)
    
    # Relationship types legend
    ax.text(1, legend_y - 0.5, 'RELATIONSHIPS:', fontsize=10, fontweight='bold')
    
    # Inheritance arrow
    ax.annotate('', xy=(3, legend_y - 0.7), xytext=(2, legend_y - 0.7),
               arrowprops=dict(arrowstyle='-|>', color=inheritance_color, lw=2))
    ax.text(3.2, legend_y - 0.7, 'Inheritance', fontsize=9)
    
    # Composition arrow
    ax.annotate('', xy=(6, legend_y - 0.7), xytext=(5, legend_y - 0.7),
               arrowprops=dict(arrowstyle='->', color=composition_color, lw=2))
    ax.text(6.2, legend_y - 0.7, 'Uses/Contains', fontsize=9)
    
    # Orchestration arrow
    ax.annotate('', xy=(9, legend_y - 0.7), xytext=(8, legend_y - 0.7),
               arrowprops=dict(arrowstyle='->', color=orchestrator_color, lw=2))
    ax.text(9.2, legend_y - 0.7, 'Orchestrates', fontsize=9)
    
    # Data flow arrow
    ax.annotate('', xy=(12, legend_y - 0.7), xytext=(11, legend_y - 0.7),
               arrowprops=dict(arrowstyle='<-', color=data_flow_color, lw=2))
    ax.text(12.2, legend_y - 0.7, 'Data Flow', fontsize=9)
    
    # Factory arrow
    ax.annotate('', xy=(15, legend_y - 0.7), xytext=(14, legend_y - 0.7),
               arrowprops=dict(arrowstyle='->', color=factory_color, lw=2))
    ax.text(15.2, legend_y - 0.7, 'Creates', fontsize=9)
    
    # Add layer labels
    ax.text(19, interface_y + 0.5, 'INTERFACE\nLAYER', ha='center', va='center', 
            fontsize=10, fontweight='bold', rotation=90)
    ax.text(19, 10, 'IMPLEMENTATION\nLAYER', ha='center', va='center', 
            fontsize=10, fontweight='bold', rotation=90)
    ax.text(19, 5, 'ORCHESTRATION\nLAYER', ha='center', va='center', 
            fontsize=10, fontweight='bold', rotation=90)
    ax.text(19, 2.5, 'FACTORY\nLAYER', ha='center', va='center', 
            fontsize=10, fontweight='bold', rotation=90)
    
    plt.tight_layout()
    plt.savefig('ml_trading_system_architecture.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('ml_trading_system_architecture.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print("Architecture diagram saved as:")
    print("- ml_trading_system_architecture.png (high resolution)")
    print("- ml_trading_system_architecture.pdf (vector format)")
    
    plt.show()

if __name__ == "__main__":
    create_architecture_diagram()