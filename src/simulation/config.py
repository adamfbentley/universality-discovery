"""
Experiment Configuration
=======================
Central configuration file for the ML Universality Classification experiment.
All parameters, paths, and settings are defined here.
"""

import os
from pathlib import Path
import numpy as np

# ============================================================================
# EXPERIMENT METADATA
# ============================================================================

EXPERIMENT_NAME = "ML_Universality_Classification"
EXPERIMENT_VERSION = "1.0"
AUTHOR = "Physics ML Research"
DATE = "October 2025"

# ============================================================================
# DIRECTORY STRUCTURE
# ============================================================================

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = RESULTS_DIR / "models"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RESULTS_DIR, PLOTS_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# File paths
PHYSICS_DATA_PATH = DATA_DIR / "physics_trajectories.pkl"
FEATURES_DATA_PATH = DATA_DIR / "extracted_features.pkl"
ML_RESULTS_PATH = RESULTS_DIR / "ml_results.pkl"

# ============================================================================
# PHYSICS SIMULATION PARAMETERS
# ============================================================================

# Simulation grid parameters
SIMULATION_CONFIG = {
    # Grid dimensions
    'width': 512,                    # Lattice width (spatial sites) - research standard
    'height': 500,                   # Time steps - allows proper scaling regime
    'samples_per_class': 90,         # Number of samples to generate per universality class
    
    # Model parameters - Two genuinely different universality classes
    'edwards_wilkinson': {
        'diffusion_range': (0.8, 1.2),
        'noise_strength_range': (0.8, 1.2),
        'dt': 0.1,
        'description': 'Edwards-Wilkinson Universality Class (Linear Growth)'
    },
    
    'kpz_equation': {
        'diffusion_range': (0.5, 1.5),
        'nonlinearity_range': (0.8, 1.2),
        'noise_strength_range': (0.8, 1.2),
        'dt': 0.05,
        'description': 'KPZ Universality Class (Nonlinear Growth)'
    }
}

# Theoretical scaling exponents for validation (1D systems)
THEORETICAL_EXPONENTS = {
    'edwards_wilkinson': {'alpha': 0.5, 'beta': 0.25, 'z': 2.0},
    'kpz_equation': {'alpha': 0.5, 'beta': 0.33, 'z': 1.5}
}

# Class information - Two genuinely different universality classes
CLASS_NAMES = ['Edwards-Wilkinson', 'KPZ']
MODEL_TYPES = ['edwards_wilkinson', 'kpz_equation']

# ============================================================================
# FEATURE EXTRACTION PARAMETERS
# ============================================================================

FEATURE_CONFIG = {
    # Scaling exponent computation
    'alpha_computation': {
        'min_length_fraction': 0.125,    # Minimum segment length as fraction of width
        'max_length_fraction': 0.5,      # Maximum segment length as fraction of width
        'n_length_points': 8,            # Number of length scales to sample
        'n_segments_per_length': 15,     # Number of random segments per length scale
        'physical_bounds': (0.001, 2.5)  # Physical constraints for α (relaxed)
    },
    
    'beta_computation': {
        'transient_skip_fraction': 0.3,  # Skip first 30% of time evolution
        'sampling_step': 2,               # Use every nth time point
        'min_points': 8,                 # Minimum points for reliable fitting
        'physical_bounds': (0.001, 1.5)  # Physical constraints for β (relaxed)
    },
    
    # Spectral analysis
    'spectral_features': {
        'freq_cutoff': 0.4,              # Maximum frequency to consider (Nyquist limit)
        'n_freq_bins': 10                # Number of frequency bins
    },
    
    # Morphological analysis
    'morphology': {
        'gradient_window': 3,            # Window size for gradient computation
        'correlation_lags': [1, 2, 4, 8, 16]  # Lag values for autocorrelation
    }
}

# Feature names for interpretability
FEATURE_NAMES = [
    'alpha_roughness', 'beta_growth', 'total_power', 'peak_frequency', 
    'low_freq_power', 'high_freq_power', 'mean_height', 'std_height',
    'mean_gradient', 'gradient_variance', 'width_change', 'velocity_mean',
    'velocity_std', 'autocorr_lag1', 'autocorr_lag4', 'autocorr_lag16'
]

# ============================================================================
# MACHINE LEARNING PARAMETERS
# ============================================================================

ML_CONFIG = {
    # Data splitting
    'test_size': 0.25,
    'validation_size': 0.2,  # From training set
    'random_state': 42,
    'stratify': True,
    
    # Random Forest parameters
    'random_forest': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'bootstrap': True,
        'random_state': 42
    },
    
    # SVM parameters
    'svm': {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale',
        'random_state': 42
    },
    
    # Cross-validation
    'cv_folds': 5,
    'cv_scoring': 'accuracy',
    
    # Feature scaling
    'scale_features': True
}

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

PLOT_CONFIG = {
    # Figure settings
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    
    # Colors for each class
    'class_colors': ['#FF6B6B', '#4ECDC4', '#45B7D1'],
    'color_palette': 'Set2',
    
    # Plot types to generate
    'plots_to_generate': [
        'trajectory_samples',
        'feature_distributions', 
        'feature_importance',
        'confusion_matrices',
        'learning_curves',
        'pca_visualization',
        'classification_report'
    ],
    
    # Plot-specific settings
    'trajectory_plot': {
        'n_samples_per_class': 2,
        'time_range': (100, 150),  # Show last 50 time steps
        'spatial_range': (0, 64)   # Show first 64 spatial sites
    }
}

# ============================================================================
# QUALITY CONTROL PARAMETERS
# ============================================================================

QUALITY_CONFIG = {
    # Physics validation thresholds
    'min_alpha': 0.01,
    'max_alpha': 2.0,
    'min_beta': 0.01,
    'max_beta': 1.5,
    
    # Trajectory quality checks
    'min_interface_variance': 1e-6,  # Minimum variance to avoid flat interfaces
    'max_outlier_fraction': 0.1,     # Maximum fraction of outlier samples to discard
    
    # ML validation thresholds
    'min_cv_accuracy': 0.6,          # Minimum cross-validation accuracy
    'max_class_imbalance': 0.4       # Maximum allowed class imbalance
}

# ============================================================================
# ADVANCED FEATURES (OPTIONAL)
# ============================================================================

ADVANCED_CONFIG = {
    # Neural network architectures (if TensorFlow available)
    'neural_networks': {
        'enable': True,
        'architectures': ['1d_cnn', '2d_cnn', 'lstm'],
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.001
    },
    
    # Ensemble methods
    'ensemble_methods': {
        'enable': True,
        'methods': ['voting', 'bagging', 'stacking'],
        'n_estimators': 10
    },
    
    # Hyperparameter optimization
    'hyperparameter_tuning': {
        'enable': False,  # Set to True for extensive tuning
        'method': 'random_search',
        'n_iter': 50,
        'cv_folds': 3
    },
    
    # Interpretability analysis
    'interpretability': {
        'enable': True,
        'methods': ['permutation_importance', 'partial_dependence'],
        'shap_analysis': False  # Requires SHAP library
    }
}

# ============================================================================
# COMPUTATIONAL PARAMETERS
# ============================================================================

COMPUTE_CONFIG = {
    # Parallel processing
    'n_jobs': -1,  # Use all available cores
    'use_parallel': True,
    
    # Memory management
    'batch_processing': False,  # Set True for very large datasets
    'batch_size': 1000,
    
    # Optimization
    'use_numba': True,         # Use Numba acceleration where possible
    'precision': 'float32',    # Computational precision
    
    # Progress tracking
    'verbose': True,
    'progress_bars': True
}

# ============================================================================
# OUTPUT FORMAT SETTINGS
# ============================================================================

OUTPUT_CONFIG = {
    # File formats
    'save_formats': ['pkl', 'csv'],  # Data formats
    'plot_formats': ['png', 'pdf'],  # Plot formats
    
    # Reporting
    'generate_report': True,
    'report_format': 'markdown',
    
    # Logging
    'log_level': 'INFO',
    'log_to_file': True,
    'log_file': RESULTS_DIR / 'experiment.log'
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_config(model_type: str) -> dict:
    """Get configuration for a specific model type."""
    return SIMULATION_CONFIG.get(model_type, {})

def get_theoretical_exponents(model_type: str) -> dict:
    """Get theoretical scaling exponents for a model type."""
    return THEORETICAL_EXPONENTS.get(model_type, {})

def get_class_index(model_type: str) -> int:
    """Get class index for a model type."""
    try:
        return MODEL_TYPES.index(model_type)
    except ValueError:
        return -1

def print_config_summary():
    """Print a summary of the experiment configuration."""
    print(f"🔬 {EXPERIMENT_NAME} v{EXPERIMENT_VERSION}")
    print(f"📅 {DATE} | 👤 {AUTHOR}")
    print("\n📊 Experiment Configuration Summary:")
    print(f"  • Physics Models: {len(MODEL_TYPES)} classes")
    print(f"  • Samples per Class: {SIMULATION_CONFIG['samples_per_class']}")
    print(f"  • Grid Size: {SIMULATION_CONFIG['width']}×{SIMULATION_CONFIG['height']}")
    print(f"  • Features: {len(FEATURE_NAMES)} total")
    print(f"  • ML Models: Random Forest, SVM")
    print(f"  • Test Split: {ML_CONFIG['test_size']:.0%}")
    print(f"  • Cross-Validation: {ML_CONFIG['cv_folds']}-fold")
    print(f"  • Output Directory: {RESULTS_DIR}")

if __name__ == "__main__":
    print_config_summary()