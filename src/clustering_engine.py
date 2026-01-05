import numpy as np
import pandas as pd
import logging
from sklearn.cluster import KMeans

# clustering_engine.py
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

class ClusteringEngine:
    """Handle clustering with validation and stability checks."""
    
    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
    
    def fit_predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Fit KMeans and return cluster labels.
        
        WARNING: This implementation doesn't check cluster stability!
        Consider using consensus clustering or validation.
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(features)
        
        # Fit KMeans
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10  # Multiple initializations
        )
        labels = kmeans.fit_predict(X_scaled)
        
        # Validate clustering quality
        silhouette = silhouette_score(X_scaled, labels)
        self.logger.info(f"Silhouette score: {silhouette:.3f}")
        
        if silhouette < 0.2:
            self.logger.warning("Low silhouette score - clusters may not be well-separated")
        
        return labels
    
    def validate_cluster_stability(
        self, 
        features: pd.DataFrame,
        n_iterations: int = 10
    ) -> float:
        """
        Check if clusters are stable across multiple runs.
        Returns: average adjusted Rand index (ARI) across runs.
        """
        from sklearn.metrics import adjusted_rand_score
        
        labels_list = []
        for i in range(n_iterations):
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=i)
            labels = kmeans.fit_predict(self.scaler.fit_transform(features))
            labels_list.append(labels)
        
        # Compare all pairs
        ari_scores = []
        for i in range(len(labels_list)):
            for j in range(i+1, len(labels_list)):
                ari = adjusted_rand_score(labels_list[i], labels_list[j])
                ari_scores.append(ari)
        
        avg_ari = np.mean(ari_scores)
        self.logger.info(f"Average ARI (cluster stability): {avg_ari:.3f}")
        
        return avg_ari