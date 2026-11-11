import numpy as np
from scipy import signal, fft
from utils.utils_functions import taylorf2_2pn


def welch_overlapped_snr(
    data: np.ndarray,
    fs: float,
    duration: float,
    psd: np.ndarray,
    freqs: np.ndarray,
    mc_ref: float,
    q_ref: float,
    bandpass: tuple[float, float] = (20.0, 500.0)
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """
    Computes the Matched Filter SNR across the entire 4096s stream 
    Welch's 50%-overlapped 4s blocks weighted by the Tukey window.
    Returns the SNR time series, MFSNR (= 0.5 * SNR^2), peak time t_c,ML,
    and its 4s block index.
    """
    n_total = len(data)
    samples = int(duration * fs)
    half_samples = samples // 2
    n_blocks = (n_total // half_samples) - 1  # 2047 overlapped blocks of 4s

    # Tukey window and overlap weights
    window = signal.windows.tukey(samples, alpha=0.5)
    window_rms = np.sqrt(np.mean(window ** 2))
    left_window = window[:half_samples]
    right_window = window[half_samples:]
    weight = left_window + right_window

    # Bandpass mask and frequency resolution
    f_min, f_max = bandpass
    band_mask = (freqs >= f_min) & (freqs <= f_max)
    df = freqs[1] - freqs[0]

    # Reference template h_0 in band_mask and its norm sqrt((h_0 | h_0))
    h0 = np.zeros_like(freqs, dtype=np.complex128)
    h0[band_mask] = taylorf2_2pn(freqs[band_mask], mc_ref, q_ref)
    prod_h0h0 = 4.0 * df * np.sum((np.abs(h0[band_mask]) ** 2) / psd[band_mask])
    norm_h0h0 = np.sqrt(prod_h0h0)

    # Loop over all 2047 overlapped 4s blocks
    snr_blocks = np.zeros((n_blocks, samples), dtype=np.float64)
    for i in range(n_blocks):
        # 4s block of data with Tukey window applied
        data_block = data[i * half_samples : (i + 2) * half_samples] * window
        # FFT of data block with window_rms amplitude-compensation
        d_fft = fft.rfft(data_block) / (fs * window_rms)

        # Matched filter integrand: 4 * d(f) * h_0*(f) / S(f) in band_mask
        integrand = np.zeros_like(freqs, dtype=np.complex128)
        integrand[band_mask] = 4.0 * (d_fft[band_mask] * np.conj(h0[band_mask])) / psd[band_mask]

        # IFFT to obtain z(t_c) and then compute SNR(t_c) = |z(t_c)| / sqrt((h_0 | h_0))
        z_t =  fs * fft.ifft(integrand, n=samples, workers=-1)
        snr_blocks[i] = np.abs(z_t) / norm_h0h0

    # Reconstruct full 4096s SNR time series
    snr = np.zeros(n_total, dtype=np.float64)
    # First block is fully included
    snr[:half_samples] = snr_blocks[0, :half_samples]
    # Loop over the remaining blocks
    for i in range(1, n_blocks):
        # Weighted average of the overlapping blocks 
        # (right wing of the previous block and left wing of the current block)
        snr[i * half_samples : (i + 1) * half_samples] = (
            snr_blocks[i - 1, half_samples:] * right_window + snr_blocks[i, :half_samples] * left_window
        ) / weight
    # Last block is fully included
    snr[n_blocks * half_samples :] = snr_blocks[-1, half_samples:]

    # MFSNR defined as ln Lambda = 0.5 * SNR^2
    mfsnr = 0.5 * (snr ** 2)

    # Identify the peak SNR
    tc_idx = int(np.argmax(snr)) # peak index in samples
    tc_ml = tc_idx / fs          # peak time in seconds

    # A sample tc_idx belongs to two overlapping 4s blocks
    # Take the current 4s block index
    curr_block_idx = int(tc_idx // half_samples)
    # Previous 4s block index in the valid range [0, n_blocks-1]
    tc_block_idx = min(max(curr_block_idx - 1, 0), n_blocks - 1)
    # If the peak is closer to the 2s flat window center of the next block,
    # increment tc_block_idx, otherwise keep it as is (previous block)
    if abs((tc_idx - tc_block_idx * half_samples) - half_samples) > abs((tc_idx - (tc_block_idx + 1) * half_samples) - half_samples):
        tc_block_idx += 1

    return snr, mfsnr, tc_ml, tc_block_idx