from config import Config
from data_manager import DataManager
from feature_engineer import FeatureEngineer
from clustering_engine import ClusteringEngine
from portfolio_optimizer import PortfolioOptimizer
from backtester import Backtester

# main.py
import logging

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trading_bot.log'),
            logging.StreamHandler()
        ]
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    config = Config()
    
    # Initialize components
    data_mgr = DataManager()
    feature_eng = FeatureEngineer()
    clusterer = ClusteringEngine(n_clusters=config.N_CLUSTERS)
    optimizer = PortfolioOptimizer(config)
    backtester = Backtester(config)
    
    logger.info("Starting trading strategy...")
    
    # Run strategy
    # ... implementation
    
    logger.info("Strategy complete")

if __name__ == "__main__":
    main()# main.py
