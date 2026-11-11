from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fft
from utils.py_logger import CustomLogger


def savefig(file_path: Path, log: CustomLogger):
    """
    Save the figures.
    """
    try:
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
        log.success(f"Plot saved to: {file_path}")
    except Exception as e:
        log.error(f"Failed to save the plot: {e}")
    


def local_blocks(data: np.ndarray, duration: float, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Return non-overlapping blocks and their center times.
    """
    samples = int(duration * fs)
    n_blocks = len(data) // samples
    blocks = data[:n_blocks * samples].reshape(n_blocks, samples)
    centers = (np.arange(n_blocks) + 0.5) * duration
    return blocks, centers


def normalized_acf(data: np.ndarray, log: CustomLogger, maxlag: int = None) -> np.ndarray:
    """
    Computes the normalized autocorrelation function of a given 1D data array using FFT.
    """
    # Center the data by subtracting the mean
    data = data - np.mean(data)

    # Compute the autocorrelation using FFT-based convolution
    # mode='full' computes it for all the lags (from -(n-1) to +(n-1))
    autocorr = signal.correlate(data, data, mode='full', method='fft')
    
    if maxlag is None:
        # Take the entire autocorrelation function
        autocorr = autocorr[len(data)-1:]
    else:
        # Take only up to maxlag
        autocorr = autocorr[len(data)-1:len(data) + maxlag]

    # Normalize by lag 0 value (which is the variance of the data)
    if autocorr[0] == 0 or np.isnan(autocorr[0]):
        log.warning("Lag 0 is zero, cannot normalize autocorrelation function.")
        return np.zeros_like(autocorr) 

    autocorr /= autocorr[0]
    return autocorr


def welch_psd(
    data: np.ndarray,
    fs: float,
    bandpass:  tuple[float, float] =None, 
    segment_seconds: float = 4.0, 
    average: str = "mean"
) -> tuple[np.ndarray, np.ndarray]:
    """
    PSD estimate; 50% overlap, Tukey window, constant detrend.
    If bandpass is set, return PSD in the band (f_min, f_max).
    """
    nperseg = min(len(data), int(segment_seconds * fs))
    freq, psd = signal.welch(
                data,
                fs=fs,
                window=("tukey", 0.5),
                nperseg=nperseg,
                noverlap=nperseg // 2,
                detrend="constant",
                scaling="density",
                average=average
            )
    if bandpass is not None:
        f_min, f_max = bandpass
        mask = (freq >= f_min) & (freq <= f_max)
        return freq[mask], psd[..., mask]
    else:
        return freq, psd

def fft_bandpass(
    data: np.ndarray,
    fs: float,
    bandpass: tuple[float, float],
    taper_alpha: float = 0.002,
    return_window: bool = False
) -> np.ndarray:
    """
    Applies a frequency-domain bandpass filter to a 1D time series via FFT.
    
    Parameters:
    - bandpass : Target passband frequencies (f_min, f_max) in Hz
    - taper_alpha: Shape parameter of the Tukey window applied in the time domain before FFT
        (default: 0.002, tapering ~4.1 s at each edge for a 4096 s signal)
    - return_window: If True, returns both the filtered time series and the Tukey window array
    """
    n_samples = len(data)
    f_min, f_max = bandpass

    # Remove global mean and apply Tukey tapering to avoid edge discontinuities
    data_zero_mean = data - np.mean(data)
    time_window = signal.windows.tukey(n_samples, alpha=taper_alpha)
    data_tapered = data_zero_mean * time_window

    # Compute real FFT and corresponding frequency grid
    freq = fft.rfftfreq(n_samples, d=1.0 / fs)
    spectrum = fft.rfft(data_tapered, workers=-1)

    # Rectangular frequency mask on [f_min, f_max]
    mask = (freq >= f_min) & (freq <= f_max)

    # Inverse real FFT back to the time domain
    filtered = fft.irfft(spectrum * mask, n=n_samples, workers=-1)
    
    if return_window:
        return filtered, time_window
    return filtered


def log_prior_mc_q(mc: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Computes the unnormalized log-prior ln[ M * q^(-6/5) * (1 + q)^(2/5) ].
    """
    return np.log(mc) - 1.2 * np.log(q) + 0.4 * np.log(1.0 + q)


# Solar time constant G * M_sun / c^3 [seconds]
T_SUN_SEC = 4.925490947e-6

def taylorf2_2pn(
    freqs: np.ndarray,
    mc_solar: np.ndarray | float,
    q: np.ndarray | float
) -> np.ndarray:
    """
    Computes the 2PN TaylorF2 frequency-domain reference waveform h_0(f; M, q)
    at t_c = 0 and phi_c = 0.
    """
    # Convert input to arrays and add a new axis for broadcasting
    mc_arr = np.asarray(mc_solar)[..., None]
    q_arr = np.asarray(q)[..., None]
    f_arr = np.asarray(freqs)

    # Symmetric mass ratio nu and total mass M in seconds
    nu = q_arr / ((1.0 + q_arr) ** 2)
    m_tot_sec = mc_arr * (nu ** (-3.0 / 5.0)) * T_SUN_SEC

    # Post-Newtonian orbital velocity parameter v = (pi * M * f)^(1/3)
    v = (np.pi * m_tot_sec * f_arr) ** (1.0 / 3.0)
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v3 * v2

    # 2PN phase expansion
    psi_0 = 1.0
    psi_2 = (3715.0 / 756.0 + (55.0 / 9.0) * nu) * v2
    psi_3 = -16.0 * np.pi * v3
    psi_4 = (15293365.0 / 508032.0 + (27145.0 / 504.0) * nu + (3085.0 / 72.0) * nu**2) * v4

    psi_f = -0.25 * np.pi + (3.0 / (128.0 * nu * v5)) * (psi_0 + psi_2 + psi_3 + psi_4)

    # Negative phase exponent exp(-i * psi_f) matches forward FFT convention
    return (f_arr ** (-7.0 / 6.0)) * np.exp(-1j * psi_f)