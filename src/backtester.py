import numpy as np
import pandas as pd
import logging
from typing import Dict
from config import Config

# backtester.py
class Backtester:
    """Backtest the strategy with proper accounting for costs and slippage."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def run(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        transaction_cost_bps: float = 10
    ) -> pd.DataFrame:
        """
        Run backtest with transaction costs.
        
        Args:
            signals: DataFrame with columns ['date', 'ticker', 'weight']
            prices: Price data
            transaction_cost_bps: Transaction costs in basis points
        
        Returns:
            DataFrame with portfolio returns and metrics
        """
        results = {
            'date': [],
            'gross_return': [],
            'transaction_costs': [],
            'net_return': []
        }
        
        prev_weights = {}
        
        for date in signals.index.get_level_values(0).unique():
            # Get current weights
            curr_weights = signals.loc[date].to_dict()
            
            # Calculate turnover
            turnover = self._calculate_turnover(prev_weights, curr_weights)
            
            # Calculate transaction costs
            tc = turnover * (transaction_cost_bps / 10000)
            
            # Get returns for this period
            period_return = self._calculate_period_return(
                date, prices, curr_weights
            )
            
            results['date'].append(date)
            results['gross_return'].append(period_return)
            results['transaction_costs'].append(tc)
            results['net_return'].append(period_return - tc)
            
            prev_weights = curr_weights
        
        return pd.DataFrame(results).set_index('date')
    
    def _calculate_turnover(
        self,
        prev_weights: Dict[str, float],
        curr_weights: Dict[str, float]
    ) -> float:
        """Calculate portfolio turnover."""
        all_tickers = set(prev_weights.keys()) | set(curr_weights.keys())
        
        turnover = sum(
            abs(curr_weights.get(t, 0) - prev_weights.get(t, 0))
            for t in all_tickers
        )
        
        return turnover / 2  # Divide by 2 as we count both buys and sells
    
    def calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics."""
        total_return = (1 + returns).prod() - 1
        ann_return = (1 + total_return) ** (252 / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        # Maximum drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_dd = drawdown.min()
        
        return {
            'total_return': total_return,
            'annual_return': ann_return,
            'annual_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'calmar_ratio': ann_return / abs(max_dd) if max_dd != 0 else 0
        }