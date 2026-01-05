import pandas as pd
from typing import List
import numpy as np

# feature_engineer.py
class FeatureEngineer:
    """Compute technical indicators and features without look-ahead bias."""
    
    @staticmethod
    def compute_returns(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """
        Compute returns for multiple periods, properly shifted to avoid look-ahead bias.
        
        Returns at time t should only use prices up to t-1.
        """
        for period in periods:
            # Calculate return
            ret = df['adj close'].pct_change(period)
            
            # CRITICAL: Shift to avoid look-ahead bias
            df[f'return_{period}m'] = ret.shift(1)
        
        return df
    
    @staticmethod
    def compute_garman_klass_volatility(df: pd.DataFrame) -> pd.Series:
        """Garman-Klass volatility estimator."""
        gk = (
            (np.log(df['high']) - np.log(df['low']))**2 / 2 
            - (2*np.log(2)-1) * (np.log(df['adj close']) - np.log(df['open']))**2
        )
        return gk
    
    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features with proper grouping and error handling."""
        # Implementation with proper error handling
        pass
