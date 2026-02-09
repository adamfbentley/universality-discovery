"""
Physics Simulations Module
==========================
Implementation of three surface growth models for universality classification:
1. Ballistic Deposition (KPZ universality class)
2. Edwards-Wilkinson Model (Linear growth with diffusion)
3. KPZ Equation (Nonlinear growth equation)

This module generates realistic growth trajectories with proper physics
and parameter variations for machine learning training.
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from typing import Tuple, List, Dict, Optional, Any
import pickle
import time
from pathlib import Path

# Import configuration
from .config import (
    SIMULATION_CONFIG, THEORETICAL_EXPONENTS, MODEL_TYPES, CLASS_NAMES,
    PHYSICS_DATA_PATH, get_model_config, get_theoretical_exponents, 
    print_config_summary
)

# ============================================================================
# CORE GROWTH MODEL SIMULATOR
# ============================================================================

class GrowthModelSimulator:
    """
    Simulate three different universality classes of surface growth.
    
    Models implemented:
    - Ballistic Deposition: KPZ universality class
    - Edwards-Wilkinson: Linear diffusive growth  
    - KPZ Equation: Nonlinear growth equation
    """
    
    def __init__(self, width: int = None, height: int = None, random_state: int = 42):
        """
        Initialize the growth simulator.
        
        Parameters:
        -----------
        width : int
            Lattice width (number of spatial sites)
        height : int  
            Number of time steps to simulate
        random_state : int
            Random seed for reproducibility
        """
        # Use config defaults if not specified
        self.width = width or SIMULATION_CONFIG['width']
        self.height = height or SIMULATION_CONFIG['height']
        self.random_state = random_state
        
        # Set random seed
        np.random.seed(random_state)
        
        print(f"Growth Simulator initialized: {self.width}x{self.height} grid")
    
    # ========================================================================
    # BALLISTIC DEPOSITION (KPZ Universality Class)
    # ========================================================================
    
    @staticmethod
    @jit(nopython=True)
    def _ballistic_deposition_step(interface: np.ndarray, 
                                 noise_strength: float = 0.2) -> np.ndarray:
        """
        Single time step of ballistic deposition.
        
        Rule: Particles land at random x positions and stick to the first
        surface they encounter (highest neighbor + 1).
        
        Parameters:
        -----------
        interface : np.ndarray
            Current interface heights
        noise_strength : float
            Strength of thermal noise
            
        Returns:
        --------
        new_interface : np.ndarray
            Updated interface after one time step
        """
        L = len(interface)
        new_interface = interface.copy()
        
        # Random landing position
        x = np.random.randint(0, L)
        
        # Find landing height (stick to tallest neighbor)
        left_height = interface[(x-1) % L]
        center_height = interface[x]
        right_height = interface[(x+1) % L]
        
        landing_height = max(left_height, center_height, right_height) + 1
        
        # Add particle with small thermal noise
        noise = noise_strength * np.random.normal(0, 0.1)
        new_interface[x] = landing_height + noise
        
        return new_interface
    
    # ========================================================================
    # EDWARDS-WILKINSON MODEL (Linear Growth)
    # ========================================================================
    
    @staticmethod
    @jit(nopython=True)  
    def _edwards_wilkinson_step(interface: np.ndarray,
                              diffusion: float = 1.0,
                              noise_strength: float = 1.0,
                              dt: float = 0.1) -> np.ndarray:
        """
        Single time step of Edwards-Wilkinson equation.
        
        Equation: dh/dt = ν∇²h + η(x,t)
        Where ν is surface tension and η is white noise.
        
        Parameters:
        -----------
        interface : np.ndarray
            Current interface heights
        diffusion : float
            Diffusion coefficient (surface tension)
        noise_strength : float
            Noise amplitude
        dt : float
            Time step size
            
        Returns:
        --------
        new_interface : np.ndarray
            Updated interface
        """
        L = len(interface)
        new_interface = interface.copy()
        
        for x in range(L):
            # Calculate Laplacian (discrete second derivative)
            left = interface[(x-1) % L]
            center = interface[x]
            right = interface[(x+1) % L]
            laplacian = left - 2*center + right
            
            # Edwards-Wilkinson evolution
            noise = noise_strength * np.sqrt(dt) * np.random.randn()
            dhdt = diffusion * laplacian + noise
            
            new_interface[x] = center + dt * dhdt
            
        return new_interface
    
    # ========================================================================
    # KPZ EQUATION (Nonlinear Growth)
    # ========================================================================
    
    @staticmethod
    @jit(nopython=True)
    def _kpz_equation_step(interface: np.ndarray,
                         diffusion: float = 1.0,
                         nonlinearity: float = 1.0, 
                         noise_strength: float = 1.0,
                         dt: float = 0.05) -> np.ndarray:
        """
        Single time step of KPZ equation.
        
        Equation: dh/dt = ν∇²h + (λ/2)(∇h)² + η(x,t)
        Where ν is surface tension, λ is nonlinear coefficient.
        
        Parameters:
        -----------
        interface : np.ndarray
            Current interface heights
        diffusion : float
            Surface tension coefficient  
        nonlinearity : float
            Nonlinear term strength
        noise_strength : float
            Noise amplitude
        dt : float
            Time step size
            
        Returns:
        --------
        new_interface : np.ndarray
            Updated interface
        """
        L = len(interface)
        new_interface = interface.copy()
        
        for x in range(L):
            # Spatial derivatives
            left = interface[(x-1) % L]
            center = interface[x]
            right = interface[(x+1) % L]
            
            # Laplacian (surface tension term)
            laplacian = left - 2*center + right
            
            # Gradient squared (nonlinear term)
            gradient = (right - left) / 2.0
            nonlinear_term = nonlinearity * 0.5 * gradient**2
            
            # Noise term
            noise = noise_strength * np.sqrt(dt) * np.random.randn()
            
            # KPZ evolution equation
            dhdt = diffusion * laplacian + nonlinear_term + noise
            new_interface[x] = center + dt * dhdt
            
        return new_interface
    
    # ========================================================================
    # RANDOM DEPOSITION (No correlations - trivial universality)
    # ========================================================================
    
    @staticmethod
    @jit(nopython=True)
    def _random_deposition_step(interface: np.ndarray,
                               noise_strength: float = 1.0) -> np.ndarray:
        """
        Single time step of random deposition.
        
        Rule: Particles land at random x positions and simply stack
        on top - NO lateral interactions. This is the simplest discrete model.
        Results in uncorrelated interface with trivial roughening.
        
        Parameters:
        -----------
        interface : np.ndarray
            Current interface heights
        noise_strength : float
            Noise amplitude
            
        Returns:
        --------
        new_interface : np.ndarray
            Updated interface
        """
        L = len(interface)
        new_interface = interface.copy()
        
        # Random deposition: each site gets random height increment
        for x in range(L):
            new_interface[x] += noise_strength * np.abs(np.random.randn())
        
        return new_interface
    
    # ========================================================================
    # EDEN MODEL (KPZ Universality Class - Discrete)
    # ========================================================================
    
    @staticmethod
    @jit(nopython=True)
    def _eden_step(interface: np.ndarray,
                  growth_probability: float = 1.0) -> np.ndarray:
        """
        Single time step of Eden model.
        
        Rule: Growth occurs at surface sites with probability proportional
        to local curvature. This is a discrete model that belongs to the
        KPZ universality class at long times.
        
        The Eden model simulates cluster growth where new cells are added
        at random boundary sites. For a 1D interface, we approximate this
        by preferentially growing at local minima (concave regions).
        
        Parameters:
        -----------
        interface : np.ndarray
            Current interface heights
        growth_probability : float
            Base probability of growth
            
        Returns:
        --------
        new_interface : np.ndarray
            Updated interface
        """
        L = len(interface)
        new_interface = interface.copy()
        
        # Find growth probabilities based on local geometry
        # Eden model: growth more likely at surface sites (local minima)
        for x in range(L):
            left = interface[(x-1) % L]
            center = interface[x]
            right = interface[(x+1) % L]
            
            # Curvature-dependent growth: more growth at valleys
            curvature = left + right - 2*center  # Positive at valleys
            
            # Growth probability increases at valleys (like Eden cluster)
            prob = growth_probability * (0.5 + 0.3 * np.tanh(curvature))
            
            if np.random.rand() < prob:
                new_interface[x] += 1.0  # Discrete growth event
        
        return new_interface
    
    # ========================================================================
    # TRAJECTORY GENERATION
    # ========================================================================
    
    def generate_trajectory(self, model_type: str, **kwargs) -> np.ndarray:
        """
        Generate a complete growth trajectory for a specific model.
        
        Parameters:
        -----------
        model_type : str
            Type of growth model ('ballistic_deposition', 'edwards_wilkinson', 'kpz_equation')
        **kwargs : dict
            Model-specific parameters
            
        Returns:
        --------
        trajectory : np.ndarray
            Shape (height, width) array of interface evolution
        """
        # Initialize interface (flat with small random perturbations)
        interface = np.random.normal(0, 0.1, self.width)
        trajectory = np.zeros((self.height, self.width))
        
        # Evolve the interface
        for t in range(self.height):
            if model_type == 'ballistic_deposition':
                interface = self._ballistic_deposition_step(interface, **kwargs)
            elif model_type == 'edwards_wilkinson':
                interface = self._edwards_wilkinson_step(interface, **kwargs)
            elif model_type == 'kpz_equation':
                interface = self._kpz_equation_step(interface, **kwargs)
            elif model_type == 'random_deposition':
                interface = self._random_deposition_step(interface, **kwargs)
            elif model_type == 'eden':
                interface = self._eden_step(interface, **kwargs)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Remove global tilt (center the interface)
            interface = interface - np.mean(interface)
            trajectory[t] = interface.copy()
        
        return trajectory
    
    # ========================================================================
    # DATASET GENERATION WITH PARAMETER VARIATIONS
    # ========================================================================
    
    def generate_dataset(self, save_path: Optional[Path] = None) -> Tuple[List[np.ndarray], List[str], List[int]]:
        """
        Generate complete dataset with all three growth models.
        
        Parameters:
        -----------
        save_path : Path, optional
            Path to save the generated dataset
            
        Returns:
        --------
        trajectories : List[np.ndarray]
            List of growth trajectories
        labels : List[str]
            String labels for each trajectory
        class_indices : List[int] 
            Integer class labels (0, 1, 2)
        """
        trajectories = []
        labels = []
        class_indices = []
        
        samples_per_class = SIMULATION_CONFIG['samples_per_class']
        total_samples = len(MODEL_TYPES) * samples_per_class
        
        print(f"\n🎯 Generating {total_samples} samples ({samples_per_class} per class)...")
        print(f"Models: {', '.join(CLASS_NAMES)}")
        
        start_time = time.time()
        
        for class_idx, (model_type, class_name) in enumerate(zip(MODEL_TYPES, CLASS_NAMES)):
            print(f"\n📊 Generating {class_name} samples...")
            
            model_config = get_model_config(model_type)
            
            for sample in range(samples_per_class):
                # Generate random parameters for diversity
                kwargs = self._get_random_parameters(model_type, model_config)
                
                # Generate trajectory
                trajectory = self.generate_trajectory(model_type, **kwargs)
                
                trajectories.append(trajectory)
                labels.append(class_name)
                class_indices.append(class_idx)
                
                # Progress indicator
                if (sample + 1) % 10 == 0:
                    print(f"  ✅ Completed {sample + 1}/{samples_per_class}")
        
        generation_time = time.time() - start_time
        print(f"\n🎉 Dataset generation completed in {generation_time:.1f}s")
        print(f"Total samples: {len(trajectories)}")
        
        # Save dataset if path provided
        if save_path is None:
            save_path = PHYSICS_DATA_PATH
            
        self._save_dataset(trajectories, labels, class_indices, save_path)
        
        return trajectories, labels, class_indices
    
    def _get_random_parameters(self, model_type: str, config: Dict[str, Any]) -> Dict[str, float]:
        """Generate random parameters for a given model type."""
        kwargs = {}
        
        if model_type == 'ballistic_deposition':
            noise_range = config.get('noise_strength_range', (0.1, 0.3))
            kwargs['noise_strength'] = np.random.uniform(*noise_range)
            
        elif model_type == 'edwards_wilkinson':
            diff_range = config.get('diffusion_range', (0.8, 1.2))
            noise_range = config.get('noise_strength_range', (0.8, 1.2))
            kwargs['diffusion'] = np.random.uniform(*diff_range)
            kwargs['noise_strength'] = np.random.uniform(*noise_range)
            kwargs['dt'] = config.get('dt', 0.1)
            
        elif model_type == 'kpz_equation':
            diff_range = config.get('diffusion_range', (0.5, 1.5))
            nonlin_range = config.get('nonlinearity_range', (0.8, 1.2))
            noise_range = config.get('noise_strength_range', (0.8, 1.2))
            kwargs['diffusion'] = np.random.uniform(*diff_range)
            kwargs['nonlinearity'] = np.random.uniform(*nonlin_range)
            kwargs['noise_strength'] = np.random.uniform(*noise_range)
            kwargs['dt'] = config.get('dt', 0.05)
            
        return kwargs
    
    def _save_dataset(self, trajectories: List[np.ndarray], labels: List[str], 
                     class_indices: List[int], save_path: Path) -> None:
        """Save the generated dataset to disk."""
        dataset = {
            'trajectories': trajectories,
            'labels': labels,
            'class_indices': class_indices,
            'model_types': MODEL_TYPES,
            'class_names': CLASS_NAMES,
            'simulation_config': SIMULATION_CONFIG,
            'theoretical_exponents': THEORETICAL_EXPONENTS,
            'metadata': {
                'width': self.width,
                'height': self.height,
                'total_samples': len(trajectories),
                'samples_per_class': SIMULATION_CONFIG['samples_per_class'],
                'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"💾 Dataset saved to: {save_path}")
    
    # ========================================================================
    # VALIDATION AND VISUALIZATION
    # ========================================================================
    
    def validate_physics(self, n_samples: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Validate that simulations produce correct physics by measuring
        scaling exponents and comparing to theory.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples per model to test
            
        Returns:
        --------
        validation_results : Dict[str, Dict[str, float]]
            Measured vs theoretical exponents for each model
        """
        print("\n🔬 Physics Validation Test")
        print("=" * 50)
        
        results = {}
        
        for model_type, class_name in zip(MODEL_TYPES, CLASS_NAMES):
            print(f"\nTesting {class_name}...")
            
            # Generate test samples
            alphas, betas = [], []
            config = get_model_config(model_type)
            
            for i in range(n_samples):
                kwargs = self._get_random_parameters(model_type, config)
                trajectory = self.generate_trajectory(model_type, **kwargs)
                
                # Compute scaling exponents (simplified version)
                alpha, beta = self._compute_basic_scaling_exponents(trajectory)
                
                if alpha > 0 and beta > 0:  # Keep only physical values
                    alphas.append(alpha)
                    betas.append(beta)
            
            if len(alphas) > 0:
                # Calculate statistics
                alpha_mean = np.mean(alphas)
                alpha_std = np.std(alphas)
                beta_mean = np.mean(betas)
                beta_std = np.std(betas)
                
                # Get theoretical values
                theory = get_theoretical_exponents(model_type)
                alpha_theory = theory['alpha']
                beta_theory = theory['beta']
                
                # Store results
                results[model_type] = {
                    'alpha_measured': alpha_mean,
                    'alpha_std': alpha_std,
                    'alpha_theory': alpha_theory,
                    'alpha_error': abs(alpha_mean - alpha_theory) / alpha_theory,
                    'beta_measured': beta_mean,
                    'beta_std': beta_std,
                    'beta_theory': beta_theory,
                    'beta_error': abs(beta_mean - beta_theory) / beta_theory,
                    'n_valid_samples': len(alphas)
                }
                
                print(f"  ✅ Results ({len(alphas)} valid samples):")
                print(f"    α = {alpha_mean:.3f} ± {alpha_std:.3f} (theory: {alpha_theory:.3f})")
                print(f"    β = {beta_mean:.3f} ± {beta_std:.3f} (theory: {beta_theory:.3f})")
                print(f"    Errors: α={results[model_type]['alpha_error']:.1%}, β={results[model_type]['beta_error']:.1%}")
            else:
                print(f"  ❌ No valid samples generated!")
                results[model_type] = {'error': 'No valid samples'}
        
        return results
    
    def _compute_basic_scaling_exponents(self, trajectory: np.ndarray) -> Tuple[float, float]:
        """Basic scaling exponent computation for validation."""
        height, width = trajectory.shape
        
        # Roughness exponent α (spatial scaling)
        final_interface = trajectory[-1]
        lengths = np.logspace(np.log10(width//8), np.log10(width//3), 6).astype(int)
        lengths = np.unique(lengths)
        
        widths = []
        for L in lengths:
            if L >= width//3:
                break
            segments_widths = []
            for _ in range(10):  # Multiple random segments
                start = np.random.randint(0, width - L)
                segment = final_interface[start:start+L]
                if len(segment) > 1:
                    w = np.std(segment - np.mean(segment))
                    if w > 1e-10:
                        segments_widths.append(w)
            if len(segments_widths) > 0:
                widths.append(np.mean(segments_widths))
        
        if len(widths) >= 3:
            try:
                alpha = np.polyfit(np.log(lengths[:len(widths)]), np.log(widths), 1)[0]
                alpha = np.clip(alpha, 0.01, 1.99)
            except:
                alpha = 0.5
        else:
            alpha = 0.5
        
        # Growth exponent β (temporal scaling) 
        start_time = height // 3
        times = np.arange(start_time, height, 2)
        
        interface_widths = []
        for t in times:
            if t >= height:
                break
            interface = trajectory[t] - np.mean(trajectory[t])
            w = np.std(interface)
            if w > 1e-10:
                interface_widths.append(w)
        
        if len(interface_widths) >= 5:
            try:
                beta = np.polyfit(np.log(times[:len(interface_widths)]), 
                                np.log(interface_widths), 1)[0]
                beta = np.clip(beta, 0.01, 0.99)
            except:
                beta = 0.33
        else:
            beta = 0.33
        
        return alpha, beta
    
    def plot_sample_trajectories(self, n_samples: int = 2, save_path: Optional[Path] = None) -> None:
        """Generate and plot sample trajectories from each model."""
        fig, axes = plt.subplots(len(MODEL_TYPES), n_samples, 
                               figsize=(4*n_samples, 3*len(MODEL_TYPES)))
        
        if n_samples == 1:
            axes = axes.reshape(-1, 1)
        
        for class_idx, (model_type, class_name) in enumerate(zip(MODEL_TYPES, CLASS_NAMES)):
            config = get_model_config(model_type)
            
            for sample in range(n_samples):
                kwargs = self._get_random_parameters(model_type, config)
                trajectory = self.generate_trajectory(model_type, **kwargs)
                
                # Plot final interface
                ax = axes[class_idx, sample]
                x = np.arange(self.width)
                final_interface = trajectory[-1]
                
                ax.plot(x, final_interface, 'b-', linewidth=1)
                ax.set_title(f'{class_name} (Sample {sample+1})')
                ax.set_xlabel('Position')
                ax.set_ylabel('Height')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Sample trajectories saved to: {save_path}")
        
        plt.show()

# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================

def generate_physics_data(validate: bool = True, plot_samples: bool = True) -> Path:
    """
    Main function to generate physics simulation data.
    
    Parameters:
    -----------
    validate : bool
        Whether to run physics validation tests
    plot_samples : bool
        Whether to generate sample trajectory plots
        
    Returns:
    --------
    data_path : Path
        Path to the saved physics data
    """
    print("🚀 PHYSICS SIMULATION MODULE")
    print_config_summary()
    print("\n" + "="*60)
    
    # Initialize simulator
    simulator = GrowthModelSimulator()
    
    # Optional: Validate physics first
    if validate:
        validation_results = simulator.validate_physics(n_samples=3)
        print("\n📋 Physics validation completed")
    
    # Optional: Plot sample trajectories
    if plot_samples:
        from config import PLOTS_DIR
        plot_path = PLOTS_DIR / "sample_trajectories.png"
        simulator.plot_sample_trajectories(n_samples=2, save_path=plot_path)
    
    # Generate main dataset
    print("\n" + "="*60)
    trajectories, labels, class_indices = simulator.generate_dataset()
    
    print("\n✅ Physics simulation completed successfully!")
    print(f"📁 Data saved to: {PHYSICS_DATA_PATH}")
    
    return PHYSICS_DATA_PATH

def load_physics_data(data_path: Optional[Path] = None) -> Tuple[List[np.ndarray], List[str], List[int]]:
    """
    Load previously generated physics data.
    
    Parameters:
    -----------
    data_path : Path, optional
        Path to the physics data file
        
    Returns:
    --------
    trajectories : List[np.ndarray]
        List of growth trajectories
    labels : List[str]
        String labels for each trajectory  
    class_indices : List[int]
        Integer class labels
    """
    if data_path is None:
        data_path = PHYSICS_DATA_PATH
    
    if not data_path.exists():
        raise FileNotFoundError(f"Physics data not found at {data_path}. Run generation first.")
    
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"📂 Loaded physics data from: {data_path}")
    print(f"  • Total samples: {len(dataset['trajectories'])}")
    print(f"  • Classes: {', '.join(dataset['class_names'])}")
    
    return dataset['trajectories'], dataset['labels'], dataset['class_indices']

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Physics Simulation Module")
    parser.add_argument("--validate", action="store_true", 
                       help="Run physics validation tests")
    parser.add_argument("--plot", action="store_true",
                       help="Generate sample trajectory plots")
    parser.add_argument("--load-only", action="store_true",
                       help="Only load existing data without generating new")
    
    args = parser.parse_args()
    
    if args.load_only:
        try:
            trajectories, labels, class_indices = load_physics_data()
            print("✅ Data loaded successfully")
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
    else:
        generate_physics_data(validate=args.validate, plot_samples=args.plot)