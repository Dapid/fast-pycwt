'''
Created on 1 Oct 2013

@author: david

Interface to obtain wavelet transforms and other measurements.
'''
from __future__ import division

import numpy as np
from _cwt import _mwt

WINDOW = 100


def morlet(t, W0=5.):
    return np.exp(-0.5 * (t ** 2) - 1j * W0 * t)


def ricker(t):
    raise NotImplementedError

wavel_by_name = {'m': morlet, 'morlet': morlet,
                 'r': ricker, 'ricker': ricker}


def cwt(x, psi, scales):
    '''Continuous Wavelet Transform, general case'''
    N, M = len(scales), len(x)
    out = np.empty((N, M), dtype=np.complex128)
    t = np.arange(M)

    for i, s in enumerate(scales):
        for tau in xrange(M):
            out[i, tau] = np.trapz(psi((t - tau) / s) * x) / np.sqrt(s)
                                # TODO: select only a window of values.
    return out


def mwt(x, scales):
    '''Perform Morlet wavelet transform

    Input:
    -----
        x : 1-D np.ndarray, float32 or float64. The data to be transformed.
        scales: 1-D np.ndarray, float32 or float64. Scales at which to perform
             the transformation. If it is not one of the valid types, it will
             be safely casted.

    Returns:
    -------
        out : the cwt of the x input.
    '''
    if scales.dtype is not x.dtype:
        # FIXME: I have not get the grasp of fused types yet.
        return _mwt(x.astype(np.float64, casting='same_kind'),
                    scales.astype(np.float64, casting='same_kind'))
    return _mwt(x, scales)


def wavel_W(signal, wavel, scales):
    '''Compute wavelet W transform.

    Input:
    signal
    wavel: name of the wavelet
    '''
    wvl = wavel_by_name[wavel]
    if wvl is morlet: return mwt(signal, scales)
    return cwt(signal, wvl, scales)


def move_integral(arr, window):
    '''Perform the integration over a window. The input is padded with 0s
    before and after. The output keeps the same shape as the input.

    TODO: cythonice
    TODO: improve accuracy via better integration.
    FIXME: window has to be even.
    TODO: window as a function of the scale.
    '''
    if window == 0:
        return arr                              # Do nothing.

    assert window % 2 == 0

    out = np.zeros_like(arr)

    newarr = np.zeros((arr.shape[0], arr.shape[1] + window), dtype=arr.dtype)
    newarr[:, window / 2:-window / 2] = arr

    for i in xrange(out.shape[1]):
        out[:, i] = np.trapz(newarr[:, i:i + window], axis=1)

    return out


def wavel_P(signalW, window=WINDOW):
    '''Compute the wavelet power distribution
    '''
    return move_integral(np.abs(signalW * signalW), window)


def wavel_C(signalW1, signalW2, window=WINDOW):
    '''Compute the wavelet cross-spectrum of two signals
    '''
    return move_integral(np.conj(signalW1) * signalW2, window)


def wavel_rho(signalC12, signalP1, signalP2):
    '''Compute the wavelet time-dependent scale-dependent correlation given the
    power distributions and the cross-spectrum
    '''
    return np.abs(signalC12) / np.sqrt(np.abs(signalP1 * signalP2))


def correlate_signals(signal1, signal2, wavel='morlet', scales=[1, 3, 5],
                      window=WINDOW):
    '''Compute directly the wavelet correlation between two signals
    '''

    W1 = wavel_W(signal1, wavel, scales)
    W2 = wavel_W(signal2, wavel, scales)

    return correlate_wavelets(W1, W2, window)


def correlate_wavelets(W1, W2, window=WINDOW):
    '''Compute the time correlation between two signals.

    TODO: optimise memory usage.
    '''
    num = np.abs(move_integral(np.conj(W1) * W2, window))
    den = np.sqrt(np.abs(move_integral(np.abs(W1 * W1), window) *
                         move_integral(np.abs(W2 * W2), window)))

    return np.where(den != 0.0, num / den, 0)


def phase_correlate_signals(signal1, signal2, wavel='morlet', scales=[1, 3, 5],
                            window=WINDOW):
    '''Compute the phase correlation between two signals

    Calls ```phase_correlate_wavelets```
    '''
    W1 = wavel_W(signal1, wavel, scales)
    W2 = wavel_W(signal2, wavel, scales)

    return phase_correlate_wavelets(W1, W2, window)


def phase_correlate_wavelets(W1, W2, window=WINDOW):
    '''Compute the phase correlation between two wavelets.

    The computation is done from the angle of the complex value of the
    wavelets and forcing zeros where the input is zero.

    The results are now nicely normalised.
    '''

    integrand = np.exp(1j * (np.angle(W1.conj()) + np.angle(W2)))
    integrand = np.where(W1 != 0, integrand, 0)     # No signal, no coherence
    integrand = np.where(W2 != 0, integrand, 0)
    assert not np.any(np.isnan(integrand))

    try: norm = 1 / window
    except ZeroDivisionError: norm = 1

    return norm * np.abs(move_integral(integrand, window))


def co_spectrum_signals(signal1, signal2, wavel='morlet', scales=[1, 3, 5],
                        window=WINDOW, norm=True):
    '''Compute the co-spectrum of two signals.

    The output is a tuple (real, imag)

    norm: Whether or not normalise the result
    '''
    W1 = wavel_W(signal1, wavel, scales)
    W2 = wavel_W(signal2, wavel, scales)

    res = W1.conj() * W2

    if norm is True:
        den = np.sqrt(np.abs(move_integral(np.abs(W1 * W1), window) *
                             move_integral(np.abs(W2 * W2), window)))

        out1 = np.where(den != 0, np.real(res) / den, 0)
        out2 = np.where(den != 0, np.imag(res) / den, 0)
        return out1, out2

    if norm is False:
        return np.real(res), np.imag(res)


