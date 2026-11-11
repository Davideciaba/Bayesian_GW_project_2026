import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from utils.py_logger import CustomLogger
from utils.utils_functions import (
    normalized_acf,
    local_blocks, 
    welch_psd, 
    fft_bandpass, 
    savefig
)


def wss_point(
    output_dir: Path = None, 
    input_dir: Path = None,
) -> None:
    """
    Executes the first point of the analysis (WSS and time scale) for LIGO-Hanford data (GW150914).
    Plots and saves the raw data, filtered data (via fft_bandpass in the band [20, 500] Hz),
    local means, local standard deviations and normalized ACFs for different window lengths 
    (4, 8, 16, 32, 64, 128 seconds). Also plots and saves the local PSDs, global PSD (mean) and 
    median PSD for a 32s window length.
    """
    # Path injection
    current_dir = Path(__file__).parent.resolve()
    project_root_dir = current_dir.parent
    source_dir = input_dir.resolve() if input_dir else project_root_dir
    base_out = output_dir.resolve() if output_dir else project_root_dir
    wss_point_dir = base_out/"WSS_Point_Results"
    wss_point_dir.mkdir(parents=True, exist_ok =True)

    # Logger initialization
    logger = CustomLogger(name="WSSPoint")
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
    FREQUENCY = 4096.0  # Frequency of the data sampling
    time_line = np.arange(N) / FREQUENCY


    # Plot raw data
    logger.info("Plotting raw LIGO-Hanford data (GW150914)...")
    plt.figure(figsize=(12, 6))
    plt.plot(time_line, raw_data / 1e-19, color='darkblue', linewidth=1.0)
    plt.xlabel('Time [s]', fontsize=10)
    plt.ylabel('Amplitude (Strain) [$10^{-19}$]', fontsize=10)
    plt.xlim(0, time_line[-1])
    plt.title('LIGO-Hanford Data (GW150914) - Raw Data', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    raw_dir = wss_point_dir /"Raw_Data"
    raw_dir.mkdir(parents=True, exist_ok =True)   # Create Raw Data directory
    savefig(raw_dir/"raw_data_plot.png", log=logger)


    # Raw Data Global Statistics 
    logger.info("Calculating global statistics...")
    mu_global = np.mean(raw_data)
    std_global = np.std(raw_data)
    var_global = np.var(raw_data)
    logger.info(f"Raw Global Mean: {mu_global:.3e} | Std: {std_global:.3e} | Var: {var_global:.3e}")


    # Window lengths to examine
    WINDOWS = (4.0, 8.0, 16.0, 32.0, 64.0, 128.0)


    # Raw data local means
    logger.info("Calculating local means statistics for different window lengths (Raw)...")
    plt.figure(figsize=(12, 6))
    for window in WINDOWS:
        blocks, centers = local_blocks(raw_data, duration=window, fs=FREQUENCY)
        block_means = np.mean(blocks, axis=1)
        plt.plot(centers, block_means/1e-19, label=f'{window:.1f} s')
    
        std_block_means = np.std(block_means)
        median_block_std = np.median(np.std(blocks, axis=1))
        ratio = std_block_means / median_block_std
        logger.info(
            f"[RAW] Window: {window:.1f}s | Std(means): {std_block_means:.3e} | "
            f"Median(stds): {median_block_std:.3e} | Ratio: {ratio*100:.2f}%"
        )
    logger.info("Plotting local means for different window lengths (Raw)...")
    plt.axhline(std_global/1e-19, color='black', ls='--', lw=0.8, label='Global Std', alpha = 0.8)
    plt.axhline(-std_global/1e-19, color='black', ls='--', lw=0.8, alpha = 0.8)
    plt.xlabel('Time [s]', fontsize=10)
    plt.ylabel('Mean Amplitude [$10^{-19}$]', fontsize=10)
    plt.xlim(0, time_line[-1])
    plt.title('Local Means for different window lengths (Raw Data)', fontsize=14, fontweight='bold')
    plt.legend(ncol=len(WINDOWS) // 2, fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    savefig(raw_dir/"raw_local_means_plot.png", log=logger)



    # Local Standard Deviation / Median (Raw Data)
    logger.info("Plotting local std / median for different window lengths (Raw)...")
    plt.figure(figsize=(12, 6))
    for window in WINDOWS:
        blocks, centers = local_blocks(raw_data, duration=window, fs=FREQUENCY)
        block_std = np.std(blocks, axis=1)
        plt.plot(centers, block_std / np.median(block_std), label=f"{window:.1f} s")
    plt.axhline(1.0, color="black", ls="--", lw=0.8, alpha = 0.8)
    plt.xlabel("Time [s]", fontsize=10)
    plt.ylabel("Local Std / Median Local Std", fontsize=10)
    plt.title("Local Std / Median Local Std for different window lengths (Raw Data)", fontsize=14, fontweight='bold')
    plt.xlim(0, time_line[-1])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(ncol=len(WINDOWS) // 2, fontsize=8)
    savefig(raw_dir/"raw_local_std_plot.png", log=logger)

    cmap = cm.plasma_r

    # Normalized ACF (Raw Data)
    logger.info("Plotting normalized ACF for different window lengths (Raw)...")
    zoom = 1.0
    for window in WINDOWS:
        blocks, centers = local_blocks(raw_data, duration=window, fs=FREQUENCY)
        fig, ax = plt.subplots(figsize=(12, 6))
        norm = mcolors.Normalize(vmin=centers.min(), vmax=centers.max())
        for block, center_time in zip(blocks, centers):
            autocorr = normalized_acf(block, log=logger, maxlag=int(FREQUENCY * zoom))
            lags = np.arange(len(autocorr)) / FREQUENCY
            ax.plot(lags, autocorr, color=cmap(norm(center_time)), alpha=0.7, lw=1.0)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Window Center Time (s)", fontsize=8)
        ax.set_title(f"Normalized ACF for {window:.1f}s window length (Raw Data, zoomed to {zoom}s)", fontsize=14, fontweight='bold')
        ax.set_xlabel(f"Lag $\\tau$ (seconds)", fontsize=10)
        ax.set_ylabel(f"Normalized ACF", fontsize=10)
        ax.set_xlim(0, zoom)
        ax.grid(True, linestyle='--', alpha=0.7)
        savefig(raw_dir/f"raw_norm_acf_{window:.1f}.png", log=logger)


    # Local PSDs, Global PSD and Median PSD for 32s window length (Raw)
    psd_window =32.0
    logger.info(f"Plotting Local PSDs, Global PSD (mean) and median PSD for {psd_window}s window length (Raw)...")
    blocks_psd, centers_psd = local_blocks(raw_data, duration=psd_window, fs=FREQUENCY)
    fig, ax = plt.subplots(figsize=(12, 6))
    norm = mcolors.Normalize(vmin=centers_psd.min(), vmax=centers_psd.max())  
    all_psds = []
    for block, center_time in zip(blocks_psd, centers_psd):
        # Use average="mean" to show the glitch at t~2160s
        freqs, psd_block = welch_psd(
            block,
            fs=FREQUENCY,
            segment_seconds=4.0,
            average="mean"
        )
        all_psds.append(psd_block)
        ax.loglog(freqs, psd_block, color=cmap(norm(center_time)), alpha=0.6, lw=0.8)
        
    # Global PSD (mean) and median PSD
    freqs_glob, psd_global = welch_psd(
        raw_data, 
        fs=FREQUENCY, 
        segment_seconds=4.0,
        average="mean"
    )
    psd_median = np.median(np.array(all_psds), axis=0)

    ax.loglog(freqs_glob, psd_global, color='magenta', ls='--', label='Global Mean PSD')
    ax.loglog(freqs_glob, psd_median, color='cyan', label='Median PSD')  
    ax.axvline(20, color='red', ls=':', lw=1.5, label='Bandpass Cutoff (20 - 500 Hz)')
    ax.axvline (500, color='red', ls=':', lw=1.5)
    ax.axvspan(1, 20, color='gray', alpha=0.15, label='Seismic Wall (< 20 Hz)')
    ax.axvspan (500, FREQUENCY // 2, color='orange', alpha=0.10, label='High-Freq Shot Noise (> 500 Hz)')    
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Window Center Time (s)", fontsize=8)   
    ax.set_xlim(1, FREQUENCY // 2)
    ax.set_xlabel("Frequency $f$ [Hz]", fontsize=10)
    ax.set_ylabel(r"Power Spectral Density $[\mathrm{strain}^2/\mathrm{Hz}]$", fontsize=10)
    ax.set_title(f"Power Spectral Density: {psd_window:.1f}s window, Global and Median (Raw Data)", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, which="both", linestyle='--', alpha=0.7)
    savefig(raw_dir/f"raw_psd_{psd_window:.1f}.png", log=logger)


    logger.info(f"Plotting Local PSDs, Global PSD (mean) and median PSD for {psd_window:.1f}s window length and in 20 - 500 Hz...")
    blocks_psd, centers_psd = local_blocks(raw_data, duration=psd_window, fs=FREQUENCY)    
    fig, ax = plt.subplots(figsize=(12, 6))
    norm = mcolors.Normalize(vmin=centers_psd.min(), vmax=centers_psd.max())  
    all_psds = []
    for block, center_time in zip(blocks_psd, centers_psd):
        # Use average="mean" to show the glitch at t~2160s
        freqs, psd_block = welch_psd(
            block, 
            fs=FREQUENCY,
            bandpass = (20.0, 500.0), 
            segment_seconds=4.0,  
            average="mean"
        )
        all_psds.append(psd_block)
        ax.loglog(freqs, psd_block, color=cmap(norm(center_time)), alpha=0.6, lw=0.8)
                
    # Global PSD (mean) and median PSD
    freqs_glob, psd_global = welch_psd(
        raw_data, 
        fs=FREQUENCY,
        bandpass = (20.0, 500.0), 
        segment_seconds=4.0,
        average="mean"
    )
    psd_median = np.median(np.array(all_psds), axis=0)
    
    ax.loglog(freqs_glob, psd_global, color='magenta', ls='--', label='Global Mean PSD')
    ax.loglog(freqs_glob, psd_median, color='cyan', label='Median PSD')   
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Window Center Time (s)", fontsize=8)   
    ax.set_xlim(20, 500)
    ax.set_xlabel("Frequency $f$ [Hz]", fontsize=10)
    ax.set_ylabel(r"Power Spectral Density $[\mathrm{strain}^2/\mathrm{Hz}]$", fontsize=10)
    ax.set_title(f"Power Spectral Density: {psd_window:.1f}s window, Global and Median in 20 - 500 Hz", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, which="both", linestyle='--', alpha=0.7)
    savefig(raw_dir/f"raw_psd_20_500_{psd_window:.1f}.png", log=logger)


    # FFT Bandpass filter
    logger.info("Applying FFT Bandpass filter (20 - 500 Hz)...")
    filt_data, tukey_win = fft_bandpass(
        raw_data,
        fs=FREQUENCY,
        bandpass=(20.0, 500.0),
        taper_alpha=0.002,
        return_window=True
    )

    # Filtered Data Global Statistics 
    logger.info("Calculating global statistics...")
    mu_global_filt = np.mean(filt_data)
    std_global_filt = np.std(filt_data)
    var_global_filt = np.var(filt_data)
    logger.info(
        f"Filtered Global Mean: {mu_global_filt:.3e} | "
        f"Std: {std_global_filt:.3e} | Var: {var_global_filt:.3e}"
    )

    # Plot filtered data
    logger.info("Plotting LIGO-Hanford data (GW150914) - Filtered (20 - 500 Hz)...")
    plt.figure(figsize=(12, 6))
    plt.plot(time_line, filt_data / 1e-21, color='darkblue', linewidth=1.0)
    plt.xlabel('Time [s]', fontsize=10)
    plt.ylabel('Amplitude (Strain) [$10^{-21}$]', fontsize=10)
    plt.xlim(0, time_line[-1])
    plt.title('LIGO-Hanford Data (GW150914) - Filtered (20 - 500 Hz)', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    filt_dir = wss_point_dir /"Filtered_Data"
    filt_dir.mkdir(parents=True, exist_ok =True)  # Create Filtered Data directory
    savefig(filt_dir/"filt_data_plot.png", log=logger)


    # Filtered data local means
    logger.info("Calculating local means statistics for different window lengths - Filtered (20 - 500 Hz)...")
    plt.figure(figsize=(12, 6))
    for window in WINDOWS:
        blocks, centers = local_blocks(filt_data, duration=window, fs=FREQUENCY)
        block_means = np.mean(blocks, axis=1)
        plt.plot(centers, block_means/1e-21, label=f'{window:.1f} s')
        
        std_block_means = np.std(block_means)
        median_block_std = np.median(np.std(blocks, axis=1))
        ratio = std_block_means / median_block_std
        logger.info(
            f"[FILT: 20-500 Hz] Window: {window:.1f}s | Std(means): {std_block_means:.3e} | "
            f"Median(stds): {median_block_std:.3e} | Ratio: {ratio*100:.3f}%"
        )
    logger.info("Plotting local means for different window lengths (20 - 500 Hz)...")
    plt.axhline(std_global_filt/1e-21, color='black', ls='--', lw=0.8, label='Filtered Global Std', alpha = 0.8)
    plt.axhline(-std_global_filt/1e-21, color='black', ls='--', lw=0.8, alpha = 0.8)
    plt.xlabel('Time [s]', fontsize=10)
    plt.ylabel('Mean Amplitude [$10^{-21}$]', fontsize=10)
    plt.xlim(0, time_line[-1])
    plt.title('Local Means for different window lengths (20 - 500 Hz)', fontsize=14, fontweight='bold')
    plt.legend(ncol=len(WINDOWS) // 2, fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    savefig(filt_dir/"filt_local_means_plot.png", log=logger)


    # Local Standard Deviation / Median (Filtered Data)
    logger.info("Plotting local std / median for different window lengths - Filtered (20 - 500 Hz)...")
    plt.figure(figsize=(12, 6))
    for window in WINDOWS:
        blocks, centers = local_blocks(filt_data, duration=window, fs=FREQUENCY)
        win_blocks, _ = local_blocks(tukey_win, duration=window, fs=FREQUENCY)

        # RMS of the Tukey window in each block
        win_rms = np.sqrt(np.mean(win_blocks**2, axis=1))
        # Compensate for the attenuation caused by tapering at the edges to estimate the true std
        block_std = np.std(blocks, axis=1) / win_rms

        plt.plot(centers, block_std / np.median(block_std), label=f"{window:.1f} s")
    plt.axhline(1.0, color="black", ls="--", lw=0.8, alpha = 0.8)
    plt.xlabel("Time [s]", fontsize=10)
    plt.ylabel("Local Std / Median Local Std", fontsize=10)
    plt.title("Local Std / Median Local Std for different window lengths (20 - 500 Hz)", fontsize=14, fontweight='bold')
    plt.xlim(0, time_line[-1])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(ncol=len(WINDOWS) // 2, fontsize=8)
    savefig(filt_dir/"filt_local_std_plot.png", log=logger)

    # Normalized ACF (Filtered Data)
    logger.info("Plotting normalized ACF for different window lengths - Filtered (20 - 500 Hz)...")
    zoom_filt = 0.2
    for window in WINDOWS:
        blocks, centers = local_blocks(filt_data, duration=window, fs=FREQUENCY)
        fig, ax = plt.subplots(figsize=(12, 6))
        norm = mcolors.Normalize(vmin=centers.min(), vmax=centers.max())
        for block, center_time in zip(blocks, centers):
            autocorr = normalized_acf(block, log=logger, maxlag=int(FREQUENCY * zoom_filt))
            lags = np.arange(len(autocorr)) / FREQUENCY
            ax.plot(lags, autocorr, color=cmap(norm(center_time)), alpha=0.7, lw=1.0)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Window Center Time (s)", fontsize=8)
        ax.set_title(f"Normalized ACF for {window:.1f}s window length (20 - 500 Hz, zoomed to {zoom_filt}s)", fontsize=14, fontweight='bold')
        ax.set_xlabel(f"Lag $\\tau$ (seconds)", fontsize=10)
        ax.set_ylabel(f"Normalized ACF", fontsize=10)
        ax.set_xlim(0, zoom_filt)
        ax.grid(True, linestyle='--', alpha=0.7)
        savefig(filt_dir/f"filt_norm_acf_{window:.1f}.png", log=logger)

    