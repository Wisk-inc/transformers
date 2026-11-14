# Copyright 2026 Nathan. Apache-2.0.
"""
Power spectra: at which scales does a forecast still have detail?

Blurring is a loss of variance at small scales. The radially averaged power spectrum of a rain field says how much
variance each wavelength carries; the ratio of a forecast's spectrum to the observed one shows where it falls short.
The *effective resolution* is the shortest wavelength at which the forecast keeps at least half the observed power.
An AI model trained on a squared-error loss, or any ensemble mean, loses the small scales first and more with lead; a
physical single run, or a coupled scenario (:mod:`.ecc`), should not.
"""

from __future__ import annotations

import numpy as np


def radial_spectrum(field: np.ndarray, step_km: float) -> tuple[np.ndarray, np.ndarray]:
    """``(wavelength km, power)`` of a 2-D field on a regular grid: the squared Fourier amplitudes averaged over rings
    of equal wavenumber. The mean is removed and a Hann window applied, so the edges do not leak; NaN counts as zero."""
    f = np.nan_to_num(np.asarray(field, np.float64))
    f = f - f.mean()
    h, w = f.shape
    f = f * np.hanning(h)[:, None] * np.hanning(w)[None, :]
    power = np.abs(np.fft.fft2(f)) ** 2
    ky = np.fft.fftfreq(h, d=step_km)[:, None]
    kx = np.fft.fftfreq(w, d=step_km)[None, :]
    k = np.hypot(ky, kx)
    k_step = 1.0 / (max(h, w) * step_km)
    index = np.rint(k / k_step).astype(np.int64)
    n_bins = min(h, w) // 2
    sums = np.bincount(index.ravel(), weights=power.ravel(), minlength=n_bins + 1)[1:n_bins + 1]
    counts = np.bincount(index.ravel(), minlength=n_bins + 1)[1:n_bins + 1]
    wavenumber = np.arange(1, n_bins + 1) * k_step
    return 1.0 / wavenumber, sums / np.maximum(counts, 1)


class SpectrumAccumulator:
    """Mean spectra of forecasts and observations over many days."""

    def __init__(self, step_km: float) -> None:
        self.step_km = step_km
        self.wavelength = None
        self.forecast = self.observed = None
        self.count = 0

    def add(self, forecast: np.ndarray, observed: np.ndarray) -> None:
        wl, pf = radial_spectrum(forecast, self.step_km)
        _, po = radial_spectrum(observed, self.step_km)
        if self.wavelength is None:
            self.wavelength, self.forecast, self.observed = wl, np.zeros_like(pf), np.zeros_like(po)
        self.forecast += pf
        self.observed += po
        self.count += 1

    def ratio(self) -> np.ndarray:
        """Forecast power over observed power at each wavelength."""
        return self.forecast / np.maximum(self.observed, 1e-12)

    def effective_resolution(self, share: float = 0.5) -> float:
        """The shortest wavelength (km) from which the forecast keeps at least ``share`` of the observed power at
        every longer wavelength (the largest wavelength when it never does)."""
        r = self.ratio()
        ok = r >= share
        short = self.wavelength[0]
        for k in range(len(r)):                                  # wavelengths run from longest to shortest
            if not ok[k]:
                break
            short = self.wavelength[k]
        return float(short)


__all__ = ["SpectrumAccumulator", "radial_spectrum"]
