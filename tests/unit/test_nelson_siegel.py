import numpy as np

from corpbond_rv.curves.nelson_siegel import fit_nelson_siegel, nelson_siegel


def test_nelson_siegel_fit_recovers_smooth_curve():
    maturity = np.array([0.5, 1, 2, 3, 5, 7, 10, 15, 20], dtype=float)
    y = nelson_siegel(maturity, 100.0, -30.0, 50.0, 3.0)
    fit = fit_nelson_siegel(maturity, y, tau_grid=np.array([1.0, 2.0, 3.0, 5.0]))
    pred = fit.predict(maturity)
    assert fit.n_obs == len(maturity)
    assert np.sqrt(np.mean((pred - y) ** 2)) < 1e-8
