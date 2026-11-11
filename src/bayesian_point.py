import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import signal, fft
from scipy.special import logsumexp
from utils.py_logger import CustomLogger
from prior_point import plot_priors
from utils.welch_overlapped_snr import welch_overlapped_snr
from utils.utils_functions import (
    savefig,
    welch_psd,
    taylorf2_2pn,
    log_prior_mc_q
)



def bayesian_point(
    output_dir: Path = None,
    input_dir: Path = None,
    bandpass: tuple[float, float] = (20.0, 500.0)
) -> None:
    """
    Executes the third point of the analysis (Prior, Template Bank, Matched Filter SNR,
    and Posterior) for LIGO-Hanford data (GW150914).
    - Estimates the Welch PSD from raw_data (4s segments, average="mean") as in wss_point.
    - Locates the event block using the center of the template bank (M=30.0, q=0.75) as trial.
    - Constructs a 2PN TaylorF2 Template Bank in M in [25, 35] (step 0.1), q in [0.5, 1.0] (step 0.05).
    - Computes the Log-Likelihood Ratio ln Lambda_max(M, q) = (d, h_0)^2 / (2 * (h_0, h_0)) for each template.
    - Computes the Log-Likelihood ln L_max(d | M, q, H_1, I) = ln L_0 + ln Lambda_max(M, q).
    - Computes the Log-Prior ln p(M, q | H, I) and Log-Posterior ln p(M, q | d, H, I) on the template grid.
    - Computes the Log-Bayes Factor ln(B_10) = ln(Z) - ln(L_0)
    - Finds the best-fit parameters (M_best, q_best)
    - Computes the final Matched Filter SNR vs Time using (M_best, q_best) to extract t_c,ML.
    """
    # Path injection
    current_dir = Path(__file__).parent.resolve()
    project_root_dir = current_dir.parent
    source_dir = input_dir.resolve() if input_dir else project_root_dir
    base_out = output_dir.resolve() if output_dir else project_root_dir
    bayes_dir = base_out / f"Bayesian_Point_Results_{bandpass[0]}_{bandpass[1]}"
    bayes_dir.mkdir(parents=True, exist_ok=True)

    # Logger initialization
    logger = CustomLogger(name="BayesianPoint")
    logger.add_console_handler(level="DEBUG", use_colors=True)

    # Data load
    logger.info("Loading LIGO-Hanford data (GW150914)...")
    file_path = source_dir / "H-H1_GWOSC_4KHZ_R1-1126257415-4096.txt.gz"
    if not file_path.exists():
        logger.critical(
            f"{file_path.name} not found. Place the .txt.gz file in the {input_dir.name} directory"
            f" or modify {file_path} path."
        )
        sys.exit(1)

    df = pd.read_csv(file_path, header=2, compression='gzip', names=['Strain'])
    logger.success("LIGO-Hanford data (GW150914) loaded successfully!")
    raw_data = df['Strain'].values
    N = len(raw_data)
    FREQUENCY = 4096.0          # Frequency of the data sampling
    WINDOW_SEC = 4.0            # Duration of the Tukey window for Welch's method
    time_line = np.arange(N) / FREQUENCY


    # Estimate Welch PSD from raw_data 
    # (the PDS is bandpassed later in the welch_overlapped_snr function)
    logger.info("Computing Welch PSD from raw_data...")
    freqs, psd = welch_psd(
        raw_data, fs=FREQUENCY, bandpass=None, segment_seconds=WINDOW_SEC, average="mean"
    )

    # Preliminary Matched Filter Scan across the 4096s stream to locate the tc_block
    # Use the grid-center template (Mc=30.0, q=0.75) as trial
    logger.info("Running Matched Filter scan across the 4096s data stream with grid-center"
                " template (M=30.0, q=0.75) to locate the event block...")
    _, _, tc_init, tc_block_idx = welch_overlapped_snr(
        raw_data,
        fs=FREQUENCY,
        duration=WINDOW_SEC,
        psd=psd,
        freqs=freqs,
        mc_ref=30.0,
        q_ref=0.75,
        bandpass=bandpass
    )
    logger.success(f"Signal localized in block #{tc_block_idx} (around t = {tc_init:.4f} s).")

    # Construct Template Bank Grid: M in [25, 35] (step 0.1), q in [0.5, 1.0] (step 0.05)
    dMc = 0.1
    dq = 0.05
    chirp_mass = np.round(np.arange(25.0, 35.0 + 0.01, dMc), 2)  # 101 points
    ratio_mass = np.round(np.arange(0.50, 1.00 + 0.005, dq), 2)  # 11 points
    M_mesh, q_mesh = np.meshgrid(chirp_mass, ratio_mass)         # Shape: (11, 101)

    logger.info(
        f"Constructing 2PN TaylorF2 Template Bank: "
        f"{len(chirp_mass)} (M) x {len(ratio_mass)} (q) = {M_mesh.size} waveforms..."
    )

    # Extract the block containing the event from data
    # and apply the 4s Tukey window
    window_samples = int(WINDOW_SEC * FREQUENCY)
    half_samples = window_samples // 2
    window = signal.windows.tukey(window_samples, alpha=0.5)
    window_rms = np.sqrt(np.mean(window ** 2))
    tc_block = raw_data[tc_block_idx * half_samples : (tc_block_idx + 2) * half_samples] * window

    # Compute the FFT of the event block with amplitude compensation
    d_fft_tc = fft.rfft(tc_block) / (FREQUENCY * window_rms)

    # Bandpass the FFT, PSD and freqs to the analysis range [20, 500] Hz
    band_mask = (freqs >= bandpass[0]) & (freqs <= bandpass[1])
    freqs_cut = freqs[band_mask]
    psd_cut = psd[band_mask]
    d_fft_cut = d_fft_tc[band_mask]

    # Inner product of data with itself: (d | d)
    df = freqs_cut[1] - freqs_cut[0]
    prod_dd = 4.0 * df * np.sum((np.abs(d_fft_cut) ** 2) / psd_cut)

    # Generate the 1111 reference templates for all the freqs in [20, 500] Hz: shape (11, 101, 1921)
    h0 = taylorf2_2pn(freqs_cut, M_mesh, q_mesh)

    # Compute the inner product (h_0 | h_0)
    # Sum over the frequency axis (11, 101, 1921) -> (11, 101)
    prod_h0h0 = 4.0 * df * np.sum((np.abs(h0) ** 2) / psd_cut[None, None, :], axis=-1)

    # Matched filter integrand: 4 * d(f) * h_0*(f) / S(f) in band_mask
    integrand = np.zeros((M_mesh.shape[0], M_mesh.shape[1], window_samples), dtype=np.complex128)
    band_indices = np.where(band_mask)[0]   # Get indices where mask is True
    # Broadcast d_fft_cut and psd_cut to match the shape of h0
    integrand[:, :, band_indices] = 4.0 * (d_fft_cut[None, None, :] * np.conj(h0)) / psd_cut[None, None, :]

    # IFFT along the last axis (time) to get z(t_c) for each (M, q)
    z_t = FREQUENCY * fft.ifft(integrand, n=window_samples, axis=-1, workers=-1)
    # Maximize |z(t_c)| in the central region of the event block to avoid edge effects
    q1, q3 = window_samples // 4, 3 * (window_samples // 4)
    z_t_max = np.max(np.abs(z_t[:, :, q1:q3]), axis=-1) # Max over the last axis (time)

    # Maximized Log-Likelihood Ratio ln Lambda_max(M, q) = (d, h_0)^2 / (2 * (h_0, h_0))
    loglikelihood_ratio = 0.5 * ((z_t_max ** 2) / prod_h0h0)
    
    # Noise-only Log-Likelihood: ln L_0 = -0.5 * (N_f * ln(2*pi) + sum(ln(S_n)) + (d | d))
    log_L0 = -0.5 * (len(psd_cut) * np.log(2.0 * np.pi) + np.sum(np.log(psd_cut)) + prod_dd)

    # Log-Likelihood: ln L_max(d | M, q, H_1, I) = ln L_0 + ln Lambda_max(M, q)
    loglikelihood = log_L0 + loglikelihood_ratio

    # Normalize Log-Prior on the template grid
    logprior_unnorm = log_prior_mc_q(M_mesh, q_mesh)
    logprior = logprior_unnorm - logsumexp(logprior_unnorm, b=dMc * dq)

    # Normalize Log-Posterior on the template grid
    logpost_unnorm = logprior + loglikelihood
    log_evidence = logsumexp(logpost_unnorm, b=dMc * dq)
    logposterior = logpost_unnorm - log_evidence

    # Model Selection: Log-Bayes Factor ln(B_10) = ln(Z) - ln(L_0)
    log_bayes_factor = log_evidence - log_L0
    bayes_factor_log10 = log_bayes_factor / np.log(10.0)

    # 1D Marginal Log-Posteriors and Marginal Posteriors
    mc_log_marginal = logsumexp(logposterior, axis=0, b=dq)
    q_log_marginal = logsumexp(logposterior, axis=1, b=dMc)
    mc_marginal = np.exp(mc_log_marginal)
    q_marginal = np.exp(q_log_marginal)

    # Best-fit parameters
    M_max_idx = int(np.argmax(mc_log_marginal))
    q_max_idx = int(np.argmax(q_log_marginal))
    mc_best = chirp_mass[M_max_idx]
    q_best = ratio_mass[q_max_idx]

    # Compute final Matched Filter SNR time series using the best-fit parameters
    logger.info(f"Computing final 4096s Matched Filter SNR time series with best-fit (M={mc_best:.1f}, q={q_best:.2f})...")
    snr, mfsnr, tc_ml, _ = welch_overlapped_snr(
        raw_data,
        fs=FREQUENCY,
        duration=WINDOW_SEC,
        psd=psd,
        freqs=freqs,
        mc_ref=mc_best,
        q_ref=q_best,
        bandpass=bandpass
    )

    logger.success(
        f"BAYESIAN ESTIMATION COMPLETE -> Best M = {mc_best:.1f} M_sun | "
        f"Best q = {q_best:.2f} | t_c,ML = {tc_ml:.4f} s | "
        f"Max SNR = {np.max(snr):.2f} (Max ln Lambda = {np.max(mfsnr):.2f})"
    )
    logger.success(
        f"MODEL SELECTION -> ln(Z) = {log_evidence:.2f} | ln(L_0) = {log_L0:.2f} | "
        f"Log-Bayes Factor ln(B_10) = {log_bayes_factor:.2f} "
        f"(B_10 = 10^{bayes_factor_log10:.1f})"
    )


    cmap = cm.inferno

    # Plotting log prior
    fig, ax = plt.subplots(figsize=(12, 6))
    cont = ax.contourf(M_mesh, q_mesh, logprior, levels=100, cmap=cmap)
    cbar = fig.colorbar(cont, ax=ax, pad = 0.02)
    cbar.set_label("Log-Prior Density", fontsize=8)
    ax.set_title(r"LogPrior $\ln[p(\mathcal{M}, q \mid H, I)]$", fontsize=14, fontweight="bold")
    ax.set_xlabel(r"Chirp Mass $\mathcal{M} \ [M_{\odot}]$", fontsize=10)
    ax.set_ylabel(r"Mass Ratio $q$", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.7)
    savefig(bayes_dir / "logprior.png", log=logger)

    # Plotting log likelihood
    fig, ax = plt.subplots(figsize=(12, 6))
    cont = ax.contourf(M_mesh, q_mesh, loglikelihood, levels=100, cmap=cmap)
    cbar = fig.colorbar(cont, ax=ax, pad = 0.02)
    cbar.set_label("Log-Likelihood Density", fontsize=8)
    ax.set_title(r"LogLikelihood $\ln[p(\vec{d} \mid \mathcal{M}, q, H, I)]$",
                  fontsize=14, fontweight="bold")
    ax.set_xlabel(r"Chirp Mass $\mathcal{M} \ [M_{\odot}]$", fontsize=10)
    ax.set_ylabel(r"Mass Ratio $q$", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.7)
    savefig(bayes_dir / "loglikelihood.png", log=logger)

    # Plotting log posterior
    fig, ax = plt.subplots(figsize=(12, 6))
    cont = ax.contourf(M_mesh, q_mesh, logposterior, levels=100, cmap=cmap)
    ax.axvline(mc_best, lw=2.0, ls="--", color="blue",
               label=rf"Best Fit: $\mathcal{{M}}={mc_best:.1f}\,M_\odot, q={q_best:.2f}$")
    ax.axhline(q_best, lw=2.0, ls="--", color="blue")
    ax.axvline(30.0, lw=2.0, ls="-", color="white",
               label=r"Trial: $\mathcal{M}= 30\,M_\odot, q= 0.75$")
    ax.axhline(0.75, lw=2.0, ls="-", color="white")
    cbar = fig.colorbar(cont, ax=ax, pad = 0.02)
    cbar.set_label("Log-Posterior Density", fontsize=8)
    ax.set_title(r"LogPosterior $\ln[p(\mathcal{M}, q \mid \vec{d}, H, I)]$", fontsize=14, fontweight="bold")
    ax.set_xlabel(r"Chirp Mass $\mathcal{M} \ [M_{\odot}]$", fontsize=10)
    ax.set_ylabel(r"Mass Ratio $q$", fontsize=10)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.7)
    savefig(bayes_dir / "logposterior.png", log=logger)


    # Plotting marginal logposterior for M and q
    fig, (axm, axq) = plt.subplots(1, 2, figsize=(14, 6))

    axm.plot(chirp_mass, mc_log_marginal, "b-", lw=2.0)
    axm.axvline(mc_best, ls="--", color="r", lw=1.5, label=rf"$\hat{{\mathcal{{M}}}} = {mc_best:.1f}\,M_\odot$")
    axm.axvline(30.0, ls="--", color="k", lw=1.5, label=r"$\mathcal{M}_{\mathrm{trial}} = 30.0\,M_\odot$")
    axm.axvspan(28.5, 32.5, alpha=0.20, color="green", label=r"LIGO 90% C.I. (Det. Frame)")
    axm.set_title(r"$\mathcal{M}$ Marginal of LogPosterior", fontsize=14, fontweight="bold")
    axm.set_xlabel(r"Chirp Mass $\mathcal{M} \ [M_{\odot}]$", fontsize=10)
    axm.set_ylabel(r"$\log p(\mathcal{M} \mid \vec{d}, H, I)$", fontsize=10)
    axm.grid(True, linestyle="--", alpha=0.7)
    axm.legend(fontsize=8)
    
    axq.plot(ratio_mass, q_log_marginal, "b-", lw=2.0)
    axq.axvline(q_best, ls="--", color="r", lw=1.5, label=rf"$\hat{{q}} = {q_best:.2f}$")
    axq.axvline(0.75, ls="--", color="k", lw=1.5, label=r"$q_{\mathrm{trial}} = 0.75$")
    axq.axvspan(0.62, 0.99, alpha=0.20, color="green", label=r"LIGO 90% C.I.")
    axq.set_title(r"$q$ Marginal of LogPosterior", fontsize=14, fontweight="bold")
    axq.set_xlabel(r"Mass Ratio $q$", fontsize=10)
    axq.set_ylabel(r"$\log p(q \mid \vec{d}, H, I)$", fontsize=10)
    axq.grid(True, linestyle="--", alpha=0.7)
    axq.legend(fontsize=8)

    savefig(bayes_dir / "marginal_logposterior.png", log=logger)


    # Plotting marginal posterior for M and q
    fig, (axm, axq) = plt.subplots(1, 2, figsize=(14, 6))

    axm.plot(chirp_mass, mc_marginal, "b-", lw=2.0)
    axm.axvline(mc_best, ls="--", color="r", lw=1.5, label=rf"$\hat{{\mathcal{{M}}}} = {mc_best:.1f}\,M_\odot$")
    axm.axvline(30.0, ls="--", color="k", lw=1.5, label=r"$\mathcal{M}_{\mathrm{trial}} = 30.0\,M_\odot$")
    axm.axvspan(28.5, 32.5, alpha=0.20, color="green", label=r"LIGO 90% C.I. (Det. Frame)")
    axm.set_title(r"$\mathcal{M}$ Marginal of Posterior $p(\mathcal{M} \mid \vec{d}, H, I)$",
                  fontsize=14, fontweight="bold")
    axm.set_xlabel(r"Chirp Mass $\mathcal{M} \ [M_{\odot}]$", fontsize=10)
    axm.set_ylabel("Probability Density", fontsize=10)
    axm.grid(True, linestyle="--", alpha=0.7)
    axm.legend(fontsize=8)

    axq.plot(ratio_mass, q_marginal, "b-", lw=2.0)
    axq.axvline(q_best, ls="--", color="r", lw=1.5, label=rf"$\hat{{q}} = {q_best:.2f}$")
    axq.axvline(0.75, ls="--", color="k", lw=1.5, label=r"$q_{\mathrm{trial}} = 0.75$")
    axq.axvspan(0.62, 0.99, alpha=0.20, color="green", label=r"LIGO 90% C.I.")
    axq.set_title(r"$q$ Marginal of Posterior $p(q \mid \vec{d}, H, I)$",
                  fontsize=14, fontweight="bold")
    axq.set_xlabel(r"Mass Ratio $q$", fontsize=10)
    axq.set_ylabel("Probability Density", fontsize=10)
    axq.grid(True, linestyle="--", alpha=0.7)
    axq.legend(fontsize=8)

    savefig(bayes_dir / "marginal_posterior.png", log=logger)


    # Plotting Matched Filter SNR vs Time (full and zoomed around tc_ml)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(time_line, snr, color="steelblue", lw=0.7,
             label=rf"Matched Filter SNR $\rho(t)$ ($\mathcal{{M}}={mc_best:.1f}\,M_\odot, q={q_best:.2f}$)")
    ax1.set_xlim(0, time_line[-1])
    ax1.set_title("Matched Filter Signal-to-Noise Ratio vs Time",
                  fontsize=14, fontweight="bold")
    ax1.set_xlabel("Time [s]", fontsize=10)
    ax1.set_ylabel(r"SNR $\rho(t)$", fontsize=10)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.7)

    zoom_mask = (time_line >= tc_ml - 0.15) & (time_line <= tc_ml + 0.15)
    ax2.plot(time_line[zoom_mask], snr[zoom_mask], color="darkblue", lw=1.8)
    ax2.axvline(tc_ml, color="red", ls="--", lw=1.2,
                label=rf"$\hat{{t}}_{{c,\mathrm{{ML}}}} = {tc_ml:.4f}\ \mathrm{{s}}$ ($\rho_{{\max}} = {np.max(snr):.2f}$)")
    ax2.set_xlim(tc_ml - 0.15, tc_ml + 0.15)
    ax2.set_title(rf"Zoomed Matched Filter SNR vs Time ($\hat{{t}}_{{c,\mathrm{{ML}}}} \pm 0.15\ \mathrm{{s}}$)",
                  fontsize=14, fontweight="bold")
    ax2.set_xlabel("Time [s]", fontsize=10)
    ax2.set_ylabel(r"SNR $\rho(t)$", fontsize=10)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.7)

    savefig(bayes_dir / "matched_filter_snr_vs_time.png", log=logger)






























