"""
Credit below for implementation:
Patrice Guyot. (2018, April 19). Fast Python implementation of the Yin algorithm (Version v1.1.1).
(https://github.com/patriceguyot/Yin/tree/master)
"""

import numpy as np
from scipy.signal import fftconvolve
from scipy.io.wavfile import read as wavread

def difference_function(x, N, tau_max):
    """
    Compute difference function of data x. This corresponds to equation (6) in [1]
    :param x: audio data
    :param N: length of data
    :param tau_max: integration window size
    :return: difference function
    :rtype: list
    """
    x = np.array(x, np.float64)
    w = x.size
    tau_max = min(tau_max, w)
    x_cumsum = np.concatenate((np.array([0.]), (x * x).cumsum()))
    size = w + tau_max
    p2 = (size // 32).bit_length()
    nice_numbers = (16, 18, 20, 24, 25, 27, 30, 32)
    size_pad = min(x * 2 ** p2 for x in nice_numbers if x * 2 ** p2 >= size)
    fc = np.fft.rfft(x, size_pad)
    conv = np.fft.irfft(fc * fc.conjugate())[:tau_max]
    return x_cumsum[w:w - tau_max:-1] + x_cumsum[w] - x_cumsum[:tau_max] - 2 * conv

def cumulative_mean_normalized_difference_function(df, N):
    """
    Compute cumulative mean normalized difference function (CMND).

    This corresponds to equation (8) in [1]

    :param df: Difference function
    :param N: length of data
    :return: cumulative mean normalized difference function
    :rtype: list
    """

    cmndf = df[1:] * np.arange(1, N) / np.cumsum(df[1:]).astype(float) # scipy method
    return np.insert(cmndf, 0, 1)

def get_pitch(cmdf, tau_min, tau_max, harmo_th=0.1):
    """
    Return fundamental period of a frame based on CMND function.

    :param cmdf: Cumulative Mean Normalized Difference function
    :param tau_min: minimum period for speech
    :param tau_max: maximum period for speech
    :param harmo_th: harmonicity threshold to determine if it is necessary to compute pitch frequency
    :return: fundamental period if there is values under threshold, 0 otherwise
    :rtype: float
    """
    tau = tau_min
    while tau < tau_max:
        if cmdf[tau] < harmo_th:
            while tau + 1 < tau_max and cmdf[tau + 1] < cmdf[tau]:
                tau += 1
            return tau
        tau += 1

    return 0    # if unvoiced


def get_yin(audio_file, w_len=1024, w_step=256, f0_min=70, f0_max=200, harmo_thresh=0.85):
    """
    Write the results (pitches, harmonic rates, parameters ) in a numpy file.

    :param audio_file: path to the audio file
    :type audio_file: str
    :param w_len: length of the window
    :type w_len: int
    :param w_step: length of the "hop" size
    :type w_step: int
    :param f0_min: minimum f0 in Hertz
    :type f0_min: float
    :param f0_max: maximum f0 in Hertz
    :type f0_max: float
    :param harmo_thresh: harmonic threshold
    :type harmo_thresh: float
    """

    sr, sig = wavread(audio_file)
    tau_min = int(sr / f0_max)
    tau_max = int(sr / f0_min)

    timeScale = range(0, len(sig) - w_len, w_step)  # time values for each analysis window
    times = [t/float(sr) for t in timeScale]
    frames = [sig[t:t + w_len] for t in timeScale]

    pitches = [0.0] * len(timeScale)
    harmonic_rates = [0.0] * len(timeScale)

    for i, frame in enumerate(frames):
        # Compute YIN
        df = difference_function(frame, w_len, tau_max)
        cmdf = cumulative_mean_normalized_difference_function(df, tau_max)
        p = get_pitch(cmdf, tau_min, tau_max, harmo_thresh)

        # Get results
        if p != 0: # A pitch was found
            pitches[i] = float(sr / p)
            harmonic_rates[i] = cmdf[p]
        else: # No pitch, but we compute a value of the harmonic rate
            harmonic_rates[i] = min(cmdf)

    return pitches, harmonic_rates, times
