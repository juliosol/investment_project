import logging
from typing import Optional, List
import pandas as pd
from pathlib import Path
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class DataManager:
    """Handles all data acquisition and caching with proper error handling."""
    
    def __init__(self, cache_dir: str = './cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def get_sp500_constituents(self, date: Optional[pd.Timestamp] = None) -> List[str]:
        """
        Get S&P 500 constituents for a specific date.
        
        TODO: Use historical constituents to avoid survivorship bias
        Currently uses current constituents - this creates survivorship bias!
        """
        self.logger.warning("Using current S&P 500 constituents - survivorship bias present")
        # Implementation...
    
    def download_price_data(
        self, 
        symbols: List[str], 
        start: str, 
        end: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Download price data with caching and retry logic."""
        cache_file = self.cache_dir / f"prices_{start}_{end}.pkl"
        
        if use_cache and cache_file.exists():
            self.logger.info(f"Loading cached data from {cache_file}")
            return pd.read_pickle(cache_file)
        
        try:
            df = self._download_with_retry(symbols, start, end)
            df.to_pickle(cache_file)
            return df
        except Exception as e:
            self.logger.error(f"Failed to download data: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _download_with_retry(self, symbols, start, end):
        """Download data with exponential backoff retry."""
        return yf.download(symbols, start=start, end=end, auto_adjust=False)
