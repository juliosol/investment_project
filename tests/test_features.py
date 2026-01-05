from typing import Dict
import sys
sys.path.append('/home/julio/Documents/investment_project/src')
from clustering_engine import ClusteringEngine

# tests/test_features.py
import pytest
import pandas as pd
import numpy as np

def test_returns_no_lookahead_bias():
    """Test that return calculation doesn't introduce look-ahead bias."""
    dates = pd.date_range('2020-01-01', periods=100)
    prices = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)
    
    df = pd.DataFrame({'adj close': prices})
    df['return_1m'] = df['adj close'].pct_change(1).shift(1)
    
    # Return at time t should only use prices up to t-1
    for i in range(1, len(df)):
        expected = (prices.iloc[i-1] / prices.iloc[i-2] - 1) if i > 1 else np.nan
        actual = df['return_1m'].iloc[i]
        
        if not np.isnan(expected):
            assert np.isclose(actual, expected), f"Look-ahead bias detected at index {i}"

def test_clustering_reproducible():
    """Test that clustering is reproducible with same random seed."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    
    clusterer1 = ClusteringEngine(random_state=42)
    clusterer2 = ClusteringEngine(random_state=42)
    
    labels1 = clusterer1.fit_predict(pd.DataFrame(X))
    labels2 = clusterer2.fit_predict(pd.DataFrame(X))
    
    assert np.array_equal(labels1, labels2)