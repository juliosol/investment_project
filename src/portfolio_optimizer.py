import numpy as np
import pandas as pd
import logging
from typing import Dict
from config import Config
from pypfopt import EfficientFrontier, risk_models, expected_returns

# portfolio_optimizer.py
class PortfolioOptimizer:
    """Portfolio optimization with robust error handling."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def optimize(
        self,
        prices: pd.DataFrame,
        method: str = 'max_sharpe'
    ) -> Dict[str, float]:
        """
        Optimize portfolio weights.
        
        Args:
            prices: Historical prices for optimization
            method: 'max_sharpe', 'min_volatility', or 'equal_weight'
        
        Returns:
            Dictionary of {ticker: weight}
        """
        try:
            if method == 'equal_weight':
                return self._equal_weights(prices)
            
            returns = expected_returns.mean_historical_return(prices, frequency=252)
            cov = risk_models.sample_cov(prices, frequency=252)
            
            # Check for issues
            if not self._validate_inputs(returns, cov):
                self.logger.warning("Invalid inputs, falling back to equal weights")
                return self._equal_weights(prices)
            
            ef = EfficientFrontier(
                returns,
                cov,
                weight_bounds=(
                    self.config.MIN_POSITION_SIZE,
                    self.config.MAX_POSITION_SIZE
                )
            )
            
            if method == 'max_sharpe':
                weights = ef.max_sharpe()
            elif method == 'min_volatility':
                weights = ef.min_volatility()
            else:
                raise ValueError(f"Unknown method: {method}")
            
            return ef.clean_weights()
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}", exc_info=True)
            return self._equal_weights(prices)
    
    def _validate_inputs(self, returns: pd.Series, cov: pd.DataFrame) -> bool:
        """Validate optimization inputs."""
        if returns.isna().any():
            self.logger.warning("NaN values in returns")
            return False
        
        if not np.all(np.linalg.eigvals(cov) > 0):
            self.logger.warning("Covariance matrix is not positive definite")
            return False
        
        return True
    
    def _equal_weights(self, prices: pd.DataFrame) -> Dict[str, float]:
        """Return equal weights as fallback."""
        n = len(prices.columns)
        return {col: 1/n for col in prices.columns}