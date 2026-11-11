# Bayesian Gravitational-Wave Data Analysis (GW150914)

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
![GitHub repo size](https://img.shields.io/github/repo-size/Davideciaba/CMEPDA_project_2024)


A Python pipeline for stochastic noise characterization, matched filtering, and Bayesian parameter estimation of the **GW150914** binary black hole merger, using strain data recorded by the **LIGO Hanford (H1)** detector (`H-H1_GWOSC_4KHZ_R1-1126257415-4096.txt.gz`, duration $T = 4096\text{ s}$, sampling frequency $f_s = 4096\text{ Hz}$).

## Project Overview

The analysis pipeline is structured into three main stages:

### Wide-Sense Stationarity (WSS) Assessment and Spectral Analysis (`wss_point.py`)

   * Multi-scale evaluation of the local mean, local standard deviation, and normalized autocorrelation function (ACF) over non-overlapping time windows of duration $T_w \in \{4, 8, 16, 32, 64, 128\}\text{ s}$.

   * Power Spectral Density (PSD) estimation via Welch's method ($4\text{ s}$ sub-segments, Tukey window with $\alpha = 0.5$, $50\%$ overlap) and application of the Wiener--Khinchin theorem to identify distinct noise regimes (seismic wall at $f < 20\text{ Hz}$ and high-frequency shot noise at $f > 500\text{ Hz}$).

   * Frequency-domain bandpass filtering in the $[20, 500]\text{ Hz}$ range via FFT (with time-domain Tukey tapering) and verification that the filtered stochastic process is wide-sense stationary across all examined timescales.

### Prior Probability Distributions (`prior_point.py`)

   * Construction and visualization of the uniform prior distribution over the individual component masses $p(m_1, m_2 \mid H, I)$ within the domain $1\,M_\odot \le m_2 < m_1 \le 60\,M_\odot$.

   * Analytical derivation and visualization of the transformed prior distribution $p(\mathcal{M}, q \mid H, I)$ expressed in terms of the chirp mass $\mathcal{M} = (m_1 m_2)^{3/5}/(m_1 + m_2)^{1/5}$ and mass ratio $q = m_2 / m_1$ via the Jacobian determinant of the coordinate transformation.

### Matched Filtering, Bayesian Parameter Estimation, and Model Selection (`bayesian_point.py`)

   * Generation of a discrete template bank comprising $101 \times 11 = 1111$ **2PN TaylorF2** waveforms over the grid $\mathcal{M} \in [25.0, 35.0]\,M_\odot$ (step $\Delta\mathcal{M} = 0.1\,M_\odot$) and $q \in [0.50, 1.00]$ (step $\Delta q = 0.05$).

   * Reconstruction of the full $4096\text{ s}$ Matched Filter Signal-to-Noise Ratio (SNR) time series using $50\%$-overlapped $4\text{ s}$ Tukey-windowed blocks, maximizing analytically over amplitude $\mathcal{A}$ and coalescence phase $\phi_c$, and numerically over coalescence time $t_c$ via Inverse Fast Fourier Transform (IFFT) (`utils/welch_overlapped_snr.py`).

   * Evaluation of the 2D Log-Prior, Log-Likelihood, Log-Posterior, 1D marginal (Log-) posterior distributions for $\mathcal{M}$ and $q$, Maximum A Posteriori (MAP) estimators $(\hat{\mathcal{M}}, \hat{q}, \hat{t}_{c,\text{ML}})$, and the Log-Bayes Factor ($\ln \mathcal{B}_{10}$) comparing the signal hypothesis $H_1$ against the pure-noise hypothesis $H_0$.

   * Comparison between the WSS passband $[20, 500]\text{ Hz}$ and the inspiral passband $[35, 150]\text{ Hz}$ against the official $90\%$ credible intervals reported by the LIGO-Virgo Collaboration.


## Requirements and Dataset Setup

### Python Dependencies

The project requires **Python 3.10+**  and the packages listed in requirements.txt. Install them from the project root directory via:

```bash
pip install -r requirements.txt
```

### Downloading the LIGO Hanford Dataset

The strain data file `H-H1_GWOSC_4KHZ_R1-1126257415-4096.txt.gz` is **not included** in this repository due to its size. Before running the code, download it directly from the following link:

* **Download Link:** [`H-H1_GWOSC_4KHZ_R1-1126257415-4096.txt.gz`](https://www.gw-openscience.org/eventapi/html/GWTC-1-confident/GW150914/v3/H-H1_GWOSC_4KHZ_R1-1126257415-4096.txt.gz)

Once downloaded, place `H-H1_GWOSC_4KHZ_R1-1126257415-4096.txt.gz` inside the project root directory `Bayesian_GW_project_2026/` (or pass its custom folder path via the `-in` command-line flag).

## Usage

Run `main.py` from the project root directory using the command-line flags described below:

### Run the Complete Pipeline (WSS + Priors + Bayesian Analysis)

```bash
cd Bayesian_GW_project_2026/src
python main.py -wss -prior -bayes
```

### Run Individual Analysis Stages

* **WSS assessment and Spectral analysis:**

```bash
cd Bayesian_GW_project_2026/src
python main.py -wss
```

* **Prior distribution plots:**

```bash
cd Bayesian_GW_project_2026/src
python main.py -prior
```

* **Bayesian estimation in the default `[20, 500] Hz` WSS band:**

```bash
cd Bayesian_GW_project_2026/src
python main.py -bayes
```

* **Bayesian estimation in the refined `[35, 150] Hz` inspiral band:**

```bash
cd Bayesian_GW_project_2026/src
python main.py -bayes -bp 35.0 150.0
```

### Command-Line Arguments

Refer to help for the directory setup and the different analysis stages:

```bash
cd Bayesian_GW_project_2026/src
usage: main.py [-h] [-out OUTPUT_DIR] [-in INPUT_DIR] [-wss] [-prior] [-bayes] [-bp BANDPASS BANDPASS]

LIGO-Hanford data (GW150914) analysis pipeline.
Executes three main points of the analysis:
  1. WSS (Wide-sense stationarity) assessment and find a WSS time scale.
  2. Plots the Prior distributions for the parameters (m1, m2) and (M, q).
  3. Bayesian parameter estimation and Matched Filter SNR.

options:
  -h, --help            show this help message and exit
  -out OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Target root directory for all outputs. (default: sBayesian_GW_project_2026\Plots)
  -in INPUT_DIR, --input-dir INPUT_DIR
                        Source directory containing LIGO-Hanford data (GW150914). (default: Bayesian_GW_project_2026)
  -wss, --wss-point     Execute first point of the analysis (WSS and time scale). (default: False)
  -prior, --prior-distributions
                        Execute second point of the analysis (Plot prior distributions). (default: False)
  -bayes, --bayesian-point
                        Execute third point of the analysis (Prior, Template Bank, Matched Filter SNR, and Posterior).
                        (default: False)
  -bp BANDPASS BANDPASS, --bandpass BANDPASS BANDPASS
                        Frequency bandpass for the Bayesian analysis. (default: (20.0, 500.0))
```

## References

1. B. P. Abbott *et al.* (LIGO Scientific Collaboration and Virgo Collaboration), *"Observation of Gravitational Waves from a Binary Black Hole Merger"*, **Phys. Rev. Lett.** **116**, 061102 (2016), [`arXiv:1602.03837`](https://arxiv.org/abs/1602.03837).

2. B. P. Abbott *et al.* (LIGO Scientific Collaboration and Virgo Collaboration), *"Properties of the Binary Black Hole Merger GW150914"*, **Phys. Rev. Lett.** **116**, 241102 (2016), [`arXiv:1602.03840`](https://arxiv.org/abs/1602.03840).

3. A. Buonanno, B. R. Iyer, E. Ochsner, Y. Pan, and B. S. Sathyaprakash, *"Comparison of post-Newtonian templates for compact binary inspiral signals in gravitational-wave detectors"*, **Phys. Rev. D** **80**, 084043 (2009), [`arXiv:0907.0700`](https://arxiv.org/abs/0907.0700).


