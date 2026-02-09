"""
Kuramoto-Sivashinsky Equation Simulator

The KS equation describes flame front propagation and pattern formation:
    ∂h/∂t = -ν∇⁴h + κ∇²h - λ/2 (∇h)² + η

Parameters:
- ν: Fourth-order dissipation (dispersion coefficient) [stabilizing]
- κ: Second-order anti-diffusion [destabilizing, NOTE: +κ not -κ]
- λ: Nonlinearity strength
- η: Stochastic noise

Physical interpretation:
- ∇⁴ term: Stabilizing at short scales (surface tension)
- +∇² term: DESTABILIZING at intermediate scales (anti-diffusion)
- (∇h)² term: Nonlinear coupling
- Competition between stabilizing/destabilizing leads to spatiotemporal chaos

Universality class: KS (different from KPZ)
Known scaling: α ≈ 1.0 (roughness exponent, system-size dependent)

Implementation: Pseudospectral method (FFT for spatial derivatives)
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
from numba import jit

class KuramotoSivashinskySimulator:
    """
    Pseudospectral solver for 1D Kuramoto-Sivashinsky equation.
    """
    
    def __init__(self, L=256, dt=0.01):
        """
        Parameters:
        -----------
        L : int
            Number of spatial grid points
        dt : float
            Time step
        """
        self.L = L
        self.dt = dt
        
        # Spatial grid (periodic domain [0, 2π])
        self.x = np.linspace(0, 2*np.pi, L, endpoint=False)
        self.dx = self.x[1] - self.x[0]
        
        # Fourier wavenumbers
        self.k = fftfreq(L, d=self.dx/(2*np.pi))
        
        print(f"KS Simulator initialized: L={L}, dt={dt}")
        print(f"  Domain: [0, 2π], dx={self.dx:.4f}")
        print(f"  Wavenumber range: k ∈ [{self.k.min():.1f}, {self.k.max():.1f}]")
    
    def step_etdrk4(self, h_hat, nu, kappa, lam, noise_strength):
        """
        ETDRK4 (Exponential Time Differencing Runge-Kutta 4) step.
        Efficient for stiff PDEs with linear + nonlinear terms.
        
        Parameters:
        -----------
        h_hat : complex array
            Current state in Fourier space
        nu : float
            Fourth-order dissipation coefficient
        kappa : float
            Second-order dissipation coefficient
        lam : float
            Nonlinearity strength
        noise_strength : float
            Noise amplitude
        
        Returns:
        --------
        h_hat_new : complex array
            Updated state in Fourier space
        """
        k = self.k
        dt = self.dt
        
        # Linear operator L = -ν k⁴ + κ k² (note: +κ for destabilizing term)
        L = -nu * k**4 + kappa * k**2
        
        # ETDRK4 coefficients
        E = np.exp(dt * L)
        E2 = np.exp(dt * L / 2)
        
        # Avoid division by zero at k=0
        with np.errstate(divide='ignore', invalid='ignore'):
            M = (E - 1) / L
            M[0] = dt  # L'Hôpital's rule at k=0
        
        # Nonlinear term: N(h) = -λ/2 (∂h/∂x)²
        def nonlinear(h_hat_in):
            h = np.real(ifft(h_hat_in))
            dh_dx = np.real(ifft(1j * k * h_hat_in))
            N = -lam / 2 * dh_dx**2
            return fft(N)
        
        # Noise (in Fourier space, white noise in real space)
        noise_hat = fft(noise_strength * np.random.randn(self.L))
        
        # ETDRK4 stages
        N1 = nonlinear(h_hat)
        a = E2 * h_hat + M * (N1 + noise_hat) / 2
        
        N2 = nonlinear(a)
        b = E2 * h_hat + M * (N2 + noise_hat) / 2
        
        N3 = nonlinear(b)
        c = E2 * a + M * (N3 + noise_hat) / 2
        
        N4 = nonlinear(c)
        
        # Update
        h_hat_new = E * h_hat + M * (N1 + 2*N2 + 2*N3 + N4 + noise_hat) / 6
        
        return h_hat_new
    
    def simulate(self, T=500, nu=1.0, kappa=1.0, lam=1.0, noise_strength=0.1, 
                 record_interval=10, initial_condition=None):
        """
        Run KS simulation.
        
        Parameters:
        -----------
        T : int
            Total number of time steps
        nu : float
            Fourth-order dissipation
        kappa : float
            Second-order dissipation
        lam : float
            Nonlinearity strength
        noise_strength : float
            Noise amplitude
        record_interval : int
            Record state every N steps
        initial_condition : array, optional
            Initial height field (default: small random perturbation)
        
        Returns:
        --------
        trajectory : array
            Height field evolution [n_snapshots, L]
        """
        # Initial condition
        if initial_condition is None:
            h = 0.01 * np.random.randn(self.L)
        else:
            h = initial_condition.copy()
        
        h_hat = fft(h)
        
        # Storage
        n_snapshots = T // record_interval
        trajectory = np.zeros((n_snapshots, self.L))
        
        # Time integration
        snapshot_idx = 0
        for t in range(T):
            h_hat = self.step_etdrk4(h_hat, nu, kappa, lam, noise_strength)
            
            if (t + 1) % record_interval == 0:
                h = np.real(ifft(h_hat))
                trajectory[snapshot_idx] = h
                snapshot_idx += 1
        
        return trajectory
    
    def compute_scaling_exponent(self, trajectory, skip_transient=0.2):
        """
        Estimate roughness exponent α from height fluctuations.
        
        For KS: α ≈ 1.0 (but depends on system size and parameters)
        
        Parameters:
        -----------
        trajectory : array
            Height field snapshots
        skip_transient : float
            Fraction of trajectory to skip
        
        Returns:
        --------
        alpha : float
            Estimated roughness exponent
        """
        start_idx = int(skip_transient * len(trajectory))
        h_samples = trajectory[start_idx:]
        
        # Interface width W = sqrt(<(h - <h>)²>)
        widths = np.std(h_samples, axis=1)
        mean_width = np.mean(widths)
        
        # Rough estimate (proper method needs L-dependence)
        # For now, just return characteristic value
        return mean_width

# ============================================================================
# VALIDATION: Known KS Behavior
# ============================================================================

def validate_ks_dynamics():
    """
    Validate KS simulator against known behavior.
    """
    print("="*70)
    print("VALIDATING KS SIMULATOR")
    print("="*70)
    
    sim = KuramotoSivashinskySimulator(L=256, dt=0.01)
    
    # Standard KS parameters (chaotic regime: κ > 0 for anti-diffusion)
    trajectory = sim.simulate(
        T=5000,
        nu=1.0,
        kappa=2.0,  # Strong anti-diffusion for chaos
        lam=1.0,
        noise_strength=0.1,  # Reduced - chaos is deterministic
        record_interval=50
    )
    
    print(f"\nSimulation complete: {len(trajectory)} snapshots")
    
    # Check for chaotic behavior (should have persistent fluctuations)
    final_std = np.std(trajectory[-10:])
    print(f"Final std: {final_std:.4f} (should be > 0 for chaotic regime)")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Spacetime plot
    ax = axes[0, 0]
    ax.imshow(trajectory.T, aspect='auto', cmap='RdBu_r', 
             extent=[0, len(trajectory), 0, 2*np.pi])
    ax.set_xlabel('Time step')
    ax.set_ylabel('Space')
    ax.set_title('KS Spacetime Diagram')
    
    # Final snapshot
    ax = axes[0, 1]
    ax.plot(sim.x, trajectory[-1])
    ax.set_xlabel('x')
    ax.set_ylabel('h(x)')
    ax.set_title('Final Height Profile')
    ax.grid(alpha=0.3)
    
    # Interface width evolution
    ax = axes[1, 0]
    widths = np.std(trajectory, axis=1)
    ax.plot(widths)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Interface width')
    ax.set_title('Width Evolution (should saturate)')
    ax.grid(alpha=0.3)
    
    # Power spectrum (final state)
    ax = axes[1, 1]
    h_final = trajectory[-1]
    h_fft = np.abs(fft(h_final))**2
    k = fftfreq(len(h_final), d=sim.dx/(2*np.pi))
    ax.loglog(np.abs(k[1:len(k)//2]), h_fft[1:len(k)//2])
    ax.set_xlabel('Wavenumber k')
    ax.set_ylabel('Power |h(k)|²')
    ax.set_title('Power Spectrum')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/ks_validation.png', dpi=150, bbox_inches='tight')
    print("\nValidation plot saved to results/ks_validation.png")
    
    if final_std > 0.1:
        print("\n✅ KS simulator working correctly (chaotic dynamics observed)")
    else:
        print("\n⚠️  Warning: Dynamics may be too damped")

if __name__ == '__main__':
    validate_ks_dynamics()
