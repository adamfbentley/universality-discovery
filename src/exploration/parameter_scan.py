"""
Parameter Space Exploration
===========================

Scan continuous parameter spaces and map anomaly geometry.
Reveals RG basin boundaries and phase transition structure.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class ParameterRange:
    """Definition of a parameter to scan."""
    name: str
    min_val: float
    max_val: float
    n_points: int
    log_scale: bool = False
    
    def values(self) -> np.ndarray:
        """Generate parameter values."""
        if self.log_scale:
            return np.logspace(np.log10(self.min_val), np.log10(self.max_val), self.n_points)
        else:
            return np.linspace(self.min_val, self.max_val, self.n_points)


class ParameterSpaceScanner:
    """
    Scan parameter space and compute anomaly scores.
    
    Usage:
        scanner = ParameterSpaceScanner(
            simulator=my_simulator,
            anomaly_model=trained_autoencoder,
            params=[
                ParameterRange('lambda', 0.0, 2.0, 20),
                ParameterRange('kappa', 1e-4, 1.0, 20, log_scale=True)
            ]
        )
        
        results = scanner.scan()
        scanner.plot_anomaly_heatmap(results)
    """
    
    def __init__(
        self,
        simulator: Callable,
        anomaly_model: Callable,
        params: List[ParameterRange],
        n_samples_per_point: int = 10,
        random_seed: int = 42
    ):
        """
        Args:
            simulator: Function(params_dict, n_samples) -> surfaces array
            anomaly_model: Function(surfaces) -> anomaly scores
            params: List of ParameterRange objects defining scan
            n_samples_per_point: Number of surfaces per parameter combination
            random_seed: For reproducibility
        """
        self.simulator = simulator
        self.anomaly_model = anomaly_model
        self.params = params
        self.n_samples = n_samples_per_point
        self.rng = np.random.default_rng(random_seed)
    
    def scan(self, verbose: bool = True) -> Dict:
        """
        Execute full parameter scan.
        
        Returns dict with:
            - param_values: dict of parameter name -> values array
            - anomaly_scores: ndarray of shape (n_params[0], n_params[1], ...)
            - anomaly_std: standard deviation at each point
        """
        # Create parameter grids
        param_values = {p.name: p.values() for p in self.params}
        
        # Create meshgrid for multi-dimensional scan
        if len(self.params) == 1:
            grid_shape = (self.params[0].n_points,)
        elif len(self.params) == 2:
            grid_shape = (self.params[0].n_points, self.params[1].n_points)
        else:
            grid_shape = tuple(p.n_points for p in self.params)
        
        anomaly_scores = np.zeros(grid_shape)
        anomaly_std = np.zeros(grid_shape)
        
        total_points = np.prod(grid_shape)
        
        # Iterate over all parameter combinations
        for idx in np.ndindex(grid_shape):
            if verbose:
                flat_idx = np.ravel_multi_index(idx, grid_shape)
                print(f"Scanning point {flat_idx + 1}/{total_points}", end='\r')
            
            # Build parameter dict for this point
            params_dict = {}
            for i, p in enumerate(self.params):
                params_dict[p.name] = param_values[p.name][idx[i]]
            
            # Generate surfaces at this parameter point
            surfaces = self.simulator(params_dict, self.n_samples)
            
            # Compute anomaly scores
            scores = self.anomaly_model(surfaces)
            
            anomaly_scores[idx] = np.mean(scores)
            anomaly_std[idx] = np.std(scores)
        
        if verbose:
            print(f"\nScan complete: {total_points} points evaluated")
        
        return {
            'param_values': param_values,
            'anomaly_scores': anomaly_scores,
            'anomaly_std': anomaly_std,
            'param_names': [p.name for p in self.params]
        }
    
    def find_basin_boundaries(
        self,
        results: Dict,
        gradient_threshold: float = 0.5
    ) -> np.ndarray:
        """
        Identify basin boundaries from anomaly score gradients.
        
        High gradient magnitude = boundary between basins.
        
        Returns:
            Binary mask of boundary points
        """
        scores = results['anomaly_scores']
        
        # Compute gradient magnitude
        gradients = np.gradient(scores)
        if isinstance(gradients, np.ndarray):
            grad_magnitude = np.abs(gradients)
        else:
            grad_magnitude = np.sqrt(sum(g**2 for g in gradients))
        
        # Normalize
        grad_magnitude = grad_magnitude / grad_magnitude.max()
        
        # Threshold to find boundaries
        boundaries = grad_magnitude > gradient_threshold
        
        return boundaries
    
    def find_basins(
        self,
        results: Dict,
        anomaly_threshold: float = 0.5
    ) -> np.ndarray:
        """
        Identify basins of attraction (low anomaly regions).
        
        Returns:
            Integer labels for each basin (0 = high anomaly / unknown)
        """
        from scipy.ndimage import label
        
        scores = results['anomaly_scores']
        
        # Normalize scores to [0, 1]
        normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        # Low anomaly = inside basin
        low_anomaly = normalized < anomaly_threshold
        
        # Label connected regions
        basins, n_basins = label(low_anomaly)
        
        return basins, n_basins
    
    def extract_crossover_curve(
        self,
        results: Dict,
        param_name: str,
        fixed_params: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract 1D crossover curve along one parameter axis.
        
        Args:
            results: Output from scan()
            param_name: Parameter to vary
            fixed_params: Values for other parameters
        
        Returns:
            (parameter_values, anomaly_scores)
        """
        # Find index of varying parameter
        param_idx = results['param_names'].index(param_name)
        param_vals = results['param_values'][param_name]
        
        # Find indices for fixed parameters
        slice_idx = []
        for i, name in enumerate(results['param_names']):
            if name == param_name:
                slice_idx.append(slice(None))  # Vary this one
            else:
                # Find closest value
                vals = results['param_values'][name]
                idx = np.argmin(np.abs(vals - fixed_params[name]))
                slice_idx.append(idx)
        
        scores = results['anomaly_scores'][tuple(slice_idx)]
        
        return param_vals, scores
    
    def fit_crossover(
        self,
        param_vals: np.ndarray,
        scores: np.ndarray
    ) -> Dict:
        """
        Fit crossover curve to sigmoid/tanh form.
        
        D_ML(κ) = D_base + (D_max - D_base) * sigmoid((κ - κ_c) / w)
        
        Returns:
            Dict with fitted parameters: kappa_c, width, D_base, D_max, gamma
        """
        from scipy.optimize import curve_fit
        
        def sigmoid(x, x_c, w, y_min, y_max):
            return y_min + (y_max - y_min) / (1 + np.exp(-(x - x_c) / w))
        
        try:
            # Initial guesses
            p0 = [
                np.median(param_vals),  # x_c
                (param_vals.max() - param_vals.min()) / 4,  # width
                scores.min(),  # y_min
                scores.max()   # y_max
            ]
            
            popt, pcov = curve_fit(sigmoid, param_vals, scores, p0=p0, maxfev=5000)
            
            # Compute sharpness exponent γ = 1 / (w * slope_at_center)
            x_c, w, y_min, y_max = popt
            gamma = 1.0 / w if w > 0 else np.inf
            
            return {
                'kappa_c': x_c,
                'width': w,
                'D_base': y_min,
                'D_max': y_max,
                'gamma': gamma,
                'covariance': pcov.tolist()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def save_results(self, results: Dict, path: str):
        """Save scan results to JSON."""
        save_data = {
            'param_values': {k: v.tolist() for k, v in results['param_values'].items()},
            'anomaly_scores': results['anomaly_scores'].tolist(),
            'anomaly_std': results['anomaly_std'].tolist(),
            'param_names': results['param_names']
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2)
    
    @staticmethod
    def load_results(path: str) -> Dict:
        """Load scan results from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return {
            'param_values': {k: np.array(v) for k, v in data['param_values'].items()},
            'anomaly_scores': np.array(data['anomaly_scores']),
            'anomaly_std': np.array(data['anomaly_std']),
            'param_names': data['param_names']
        }


def plot_anomaly_heatmap(results: Dict, ax=None, cmap: str = 'viridis'):
    """
    Plot 2D anomaly score heatmap.
    
    Args:
        results: Output from ParameterSpaceScanner.scan()
        ax: Matplotlib axis (creates new figure if None)
        cmap: Colormap name
    """
    import matplotlib.pyplot as plt
    
    if len(results['param_names']) != 2:
        raise ValueError("Heatmap requires exactly 2 parameters")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    scores = results['anomaly_scores']
    p1_vals = results['param_values'][results['param_names'][0]]
    p2_vals = results['param_values'][results['param_names'][1]]
    
    im = ax.imshow(
        scores.T,
        origin='lower',
        aspect='auto',
        extent=[p1_vals.min(), p1_vals.max(), p2_vals.min(), p2_vals.max()],
        cmap=cmap
    )
    
    ax.set_xlabel(results['param_names'][0])
    ax.set_ylabel(results['param_names'][1])
    ax.set_title('Anomaly Score (distance from training distribution)')
    plt.colorbar(im, ax=ax, label='Anomaly Score')
    
    return ax
