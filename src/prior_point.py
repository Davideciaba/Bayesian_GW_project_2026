from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils.py_logger import CustomLogger
from utils.utils_functions import savefig

def plot_priors(
    output_dir: Path = None,
    a: float = 1.0,
    b: float = 60.0,
    resolution: int = 800
) -> None:
    """
    Plot the Prior distributions for the parameters (m1, m2) and (M,q).
    """
    # Path injection
    current_dir = Path(__file__).parent.resolve()
    project_root_dir = current_dir.parent
    base_out = output_dir.resolve() if output_dir else project_root_dir
    prior_out = base_out / "Prior_Distributions"
    prior_out.mkdir(parents=True, exist_ok=True)

    # Logger initialization
    logger = CustomLogger(name="PriorPoint")
    logger.add_console_handler(level="DEBUG", use_colors=True)
    
    logger.info("Plotting prior distributions p(m1, m2 | H, I) and p(M, q | H, I)...")
    norm_const = 2.0 / ((b - a) ** 2)
    cmap = cm.inferno

    # Uniform Prior in (m1, m2) with a < m2 < m1 < b
    m_vec = np.linspace(a, b, resolution)
    M1, M2 = np.meshgrid(m_vec, m_vec)
    Prior_m = np.where(M2 < M1, norm_const, np.nan)

    # Plotting the prior distribution in (m1, m2)
    fig, ax = plt.subplots(figsize=(12, 6))
    mesh = ax.pcolormesh(M1, M2, Prior_m, cmap=cmap, shading="auto")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Probability Density", fontsize=8)
    ax.set_title(r"Prior Distribution $p(m_1, m_2 \mid H, I)$", fontsize=14, fontweight="bold")
    ax.set_xlabel(r"$m_1 \ [M_{\odot}]$", fontsize=10)
    ax.set_ylabel(r"$m_2 \ [M_{\odot}]$", fontsize=10)
    ax.set_xlim(0, b + 3)
    ax.set_ylim(0, b + 3)
    ax.grid(True, linestyle="--", alpha=0.7)
    savefig(prior_out / "prior_distribution_m1m2.png", log=logger)

    # Mc max is when m1=b and m2=b -> Mc = b / (2**(1/5))
    mc_max = b / (2**0.2)

    # Transformed Prior in (M, q) via Jacobian determinant
    Mc = np.linspace(0.5, mc_max, resolution)
    q = np.linspace(0.02, 1.0, resolution)
    MC, Q = np.meshgrid(Mc, q)
    m1_transf = MC * (((1.0 + Q) / (Q ** 3)) ** 0.2)
    m2_transf = MC * ((Q ** 2 * (1.0 + Q)) ** 0.2)
    mask_theta = (m1_transf > a) & (m1_transf < b) & (m2_transf > a)
    jacobian_factor = MC * (Q ** -1.2) * ((1.0 + Q) ** 0.4)
    Prior_theta = np.where(mask_theta, norm_const * jacobian_factor, np.nan)

    # Plotting the prior distribution in (M, q)
    fig, ax = plt.subplots(figsize=(12, 6))
    mesh = ax.pcolormesh(MC, Q, Prior_theta, cmap=cmap, shading="auto")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Probability Density", fontsize=8)
    ax.set_title(r"Prior Distribution $p(\mathcal{M}, q \mid H, I)$", fontsize=14, fontweight="bold")
    ax.set_xlabel(r"$\mathcal{M} \ [M_{\odot}]$", fontsize=10)
    ax.set_ylabel(r"$q$", fontsize=10)
    ax.set_xlim(0, mc_max + 2)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.7)
    savefig(prior_out / "prior_distribution_Mq.png", log=logger)