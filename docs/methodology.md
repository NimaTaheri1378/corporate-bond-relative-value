# Methodology

The project follows a curve-first, alpha-second fixed-income research design:

1. clean TRACE transactions;
2. join FISD security-master data;
3. fit guarded issuer-date curves;
4. construct residual features;
5. build leakage-aware monthly labels;
6. compare transparent baselines, CPU LightGBM, and GPU PyTorch MLP;
7. evaluate cost-aware portfolios and exposure robustness.
