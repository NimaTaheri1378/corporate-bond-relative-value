# Data contracts

Step 02 converts the account-specific WRDS schema discovery into explicit table contracts.

## Selected primary sources

| Contract | WRDS table | Role |
|---|---|---|
| `trace_enhanced_clean` | `wrdsapps_bondret.trace_enhanced_clean` | Primary cleaned TRACE transaction source |
| `bondret_monthly` | `wrdsapps_bondret.bondret` | Primary monthly return labels and bond pricing reference |
| `fisd_issue_issuer` | `fisd.issue_issuer` | Security master and issuer metadata |
| `fisd_coupon_info` | `fisd.fisd_coupon_info` | Coupon, frequency, day-count terms |
| `fisd_rating_hist` | `fisd.fisd_rating_hist` | Time-aware rating buckets |
| `fisd_amount_outstanding` | `fisd.fisd_amount_outstanding` | Amount-outstanding history |
| `crsp_bond_link` | `wrdsapps_link_crsp_bond.bondcrsp_link` | Bond-to-equity link |
| `fang_bond_firm_link` | `contrib_bond_firm_link.fang_link` | Issuer-to-firm backup link |
| `contrib_bond_returns` | `contrib_corporate_bond_returns.bonds` | Alternative return labels for robustness |

## Extraction policy

Raw WRDS data is never committed. Full pulls write partitioned Parquet under `data/raw/wrds/`, while smoke pulls write small validation samples under `data/raw/wrds/smoke/`.

WRDS extraction is parallelized cautiously. Remote database work defaults to a small worker count, while later local cleaning and feature engineering can use all available CPU cores.
