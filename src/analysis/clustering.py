"""
Anomaly Clustering Analysis
============================

Cluster anomalies in latent space to discover unknown universality classes.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class ClusterResult:
    """Container for clustering results."""
    labels: np.ndarray  # Cluster assignment for each sample (-1 = noise)
    n_clusters: int
    cluster_centers: np.ndarray  # Centroid of each cluster
    cluster_sizes: np.ndarray
    silhouette_score: float  # Clustering quality metric
    probabilities: Optional[np.ndarray] = None  # Soft cluster membership


class AnomalyClusterer:
    """
    Cluster anomalous samples in latent space.
    
    Key insight: Samples flagged as anomalous (not matching training distribution)
    may cluster together - indicating shared unknown dynamics.
    
    Usage:
        clusterer = AnomalyClusterer(min_cluster_size=10)
        
        # Get latent representations of anomalies
        anomaly_latents = encoder(anomaly_surfaces)
        
        # Cluster
        result = clusterer.fit(anomaly_latents)
        
        # Analyze clusters
        for i in range(result.n_clusters):
            cluster_samples = anomaly_surfaces[result.labels == i]
            # What do these share?
    """
    
    def __init__(
        self,
        min_cluster_size: int = 10,
        min_samples: int = 5,
        metric: str = 'euclidean',
        cluster_selection_method: str = 'eom'
    ):
        """
        Args:
            min_cluster_size: Minimum points to form a cluster
            min_samples: Core sample neighborhood size
            metric: Distance metric for clustering
            cluster_selection_method: 'eom' (excess of mass) or 'leaf'
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method
    
    def fit(self, latent_vectors: np.ndarray) -> ClusterResult:
        """
        Cluster latent vectors using HDBSCAN.
        
        HDBSCAN is ideal because:
        - Doesn't require specifying number of clusters
        - Handles varying density
        - Identifies noise points (true outliers)
        
        Args:
            latent_vectors: (n_samples, latent_dim) array
        
        Returns:
            ClusterResult with labels, centers, etc.
        """
        try:
            import hdbscan
        except ImportError:
            raise ImportError("Install hdbscan: pip install hdbscan")
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method
        )
        
        labels = clusterer.fit_predict(latent_vectors)
        
        # Compute cluster statistics
        unique_labels = set(labels) - {-1}  # Exclude noise
        n_clusters = len(unique_labels)
        
        # Cluster centers and sizes
        cluster_centers = []
        cluster_sizes = []
        for label in sorted(unique_labels):
            mask = labels == label
            cluster_centers.append(latent_vectors[mask].mean(axis=0))
            cluster_sizes.append(mask.sum())
        
        cluster_centers = np.array(cluster_centers) if cluster_centers else np.array([])
        cluster_sizes = np.array(cluster_sizes) if cluster_sizes else np.array([])
        
        # Silhouette score (if we have clusters)
        if n_clusters >= 2:
            from sklearn.metrics import silhouette_score
            non_noise = labels != -1
            if non_noise.sum() > n_clusters:
                sil_score = silhouette_score(latent_vectors[non_noise], labels[non_noise])
            else:
                sil_score = 0.0
        else:
            sil_score = 0.0
        
        return ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            cluster_centers=cluster_centers,
            cluster_sizes=cluster_sizes,
            silhouette_score=sil_score,
            probabilities=clusterer.probabilities_ if hasattr(clusterer, 'probabilities_') else None
        )
    
    def fit_with_umap(
        self,
        latent_vectors: np.ndarray,
        n_components: int = 2,
        n_neighbors: int = 15
    ) -> Tuple[ClusterResult, np.ndarray]:
        """
        First reduce dimensionality with UMAP, then cluster.
        
        Useful when latent space is high-dimensional.
        
        Returns:
            (ClusterResult, umap_embedding)
        """
        try:
            import umap
        except ImportError:
            raise ImportError("Install umap: pip install umap-learn")
        
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            metric=self.metric,
            random_state=42
        )
        
        embedding = reducer.fit_transform(latent_vectors)
        result = self.fit(embedding)
        
        return result, embedding


def characterize_clusters(
    cluster_result: ClusterResult,
    surfaces: np.ndarray,
    generation_params: Optional[List[Dict]] = None
) -> List[Dict]:
    """
    Characterize each cluster by computing statistics on member surfaces.
    
    Args:
        cluster_result: Output from AnomalyClusterer.fit()
        surfaces: Original surface data (n_samples, width, time)
        generation_params: Optional list of param dicts for each surface
    
    Returns:
        List of dicts with cluster characteristics
    """
    cluster_info = []
    
    for label in range(cluster_result.n_clusters):
        mask = cluster_result.labels == label
        cluster_surfaces = surfaces[mask]
        
        info = {
            'label': label,
            'size': mask.sum(),
            'center': cluster_result.cluster_centers[label].tolist(),
        }
        
        # Compute surface statistics
        # Mean gradient
        gradients = np.diff(cluster_surfaces, axis=1)
        info['mean_abs_gradient'] = float(np.mean(np.abs(gradients)))
        info['std_gradient'] = float(np.std(gradients))
        
        # Surface width (roughness)
        widths = cluster_surfaces.std(axis=1)
        info['mean_width'] = float(np.mean(widths))
        info['final_width'] = float(np.mean(widths[:, -1]))
        
        # Simplified growth statistic: compare late width to an early-but-safe
        # reference frame. Short smoke-test trajectories may not have t=10.
        initial_idx = min(10, widths.shape[1] - 1)
        initial_width = np.mean(widths[:, initial_idx])
        final_width = np.mean(widths[:, -1])
        if initial_width > 0:
            info['width_growth_ratio'] = float(final_width / initial_width)
        
        # If we have generation parameters, summarize them
        if generation_params is not None:
            cluster_params = [generation_params[i] for i in np.where(mask)[0]]
            # Find common parameters
            param_names = set()
            for p in cluster_params:
                param_names.update(p.keys())
            
            param_summary = {}
            for name in param_names:
                values = [p.get(name) for p in cluster_params if name in p]
                if values and all(isinstance(v, (int, float)) for v in values):
                    param_summary[name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values))
                    }
            info['generation_params'] = param_summary
        
        cluster_info.append(info)
    
    return cluster_info


def compare_to_known_classes(
    cluster_result: ClusterResult,
    latent_vectors: np.ndarray,
    known_class_latents: Dict[str, np.ndarray]
) -> Dict[int, Dict[str, float]]:
    """
    Compare each cluster to known universality class representations.
    
    Args:
        cluster_result: Output from clustering anomalies
        latent_vectors: Latent vectors of anomalies
        known_class_latents: Dict mapping class name to latent vectors
    
    Returns:
        Dict mapping cluster label to distances from each known class
    """
    # Compute mean latent for each known class
    known_centers = {}
    for name, latents in known_class_latents.items():
        known_centers[name] = latents.mean(axis=0)
    
    cluster_distances = {}
    
    for label in range(cluster_result.n_clusters):
        center = cluster_result.cluster_centers[label]
        
        distances = {}
        for name, known_center in known_centers.items():
            dist = np.linalg.norm(center - known_center)
            distances[name] = float(dist)
        
        cluster_distances[label] = distances
    
    return cluster_distances


def plot_cluster_umap(
    embedding: np.ndarray,
    labels: np.ndarray,
    ax=None,
    show_noise: bool = True
):
    """
    Visualize clusters in UMAP embedding.
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_labels = set(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(sorted(unique_labels), colors):
        if label == -1:
            if show_noise:
                ax.scatter(
                    embedding[labels == label, 0],
                    embedding[labels == label, 1],
                    c='gray', alpha=0.3, s=10, label='Noise'
                )
        else:
            ax.scatter(
                embedding[labels == label, 0],
                embedding[labels == label, 1],
                c=[color], s=30, label=f'Cluster {label}'
            )
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Anomaly Clusters in Latent Space')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    return ax
