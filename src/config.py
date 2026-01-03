# Configuration file
from dataclasses import dataclass

@dataclass
class Config:
    # Data parameters
    LOOKBACK_YEARS: int = 8
    TOP_N_STOCKS: int = 150
    
    # Feature parameters
    RSI_PERIOD: int = 20
    BB_PERIOD: int = 20
    ATR_PERIOD: int = 14
    MACD_PERIOD: int = 20
    
    # Clustering parameters
    N_CLUSTERS: int = 4
    MIN_CLUSTER_SIZE: int = 10
    
    # Portfolio parameters
    MAX_POSITION_SIZE: float = 0.10
    MIN_POSITION_SIZE: float = 0.01
    REBALANCE_FREQUENCY: str = 'M'
    
    # Risk parameters
    MAX_PORTFOLIO_VOLATILITY: float = 0.20
    TRANSACTION_COST_BPS: float = 10  # 10 basis points
    
    # Backtest parameters
    INITIAL_CAPITAL: float = 100000
    BENCHMARK: str = 'SPY'