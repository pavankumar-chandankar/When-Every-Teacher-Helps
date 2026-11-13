# Results

The classification matrix CSVs contain three-seed summaries. Diagonal cells are standalone macro-F1 fractions; off-diagonal XRFv2 Plus and OPERAnet values are percentage-point gains, while MM-Fi values are F1 fractions.

| Dataset | Matrix summary | Observability |
| --- | --- | --- |
| XRFv2 Plus | `xrfv2-plus/matrix_published_rounded.csv` | `xrfv2-plus/observability.csv` (archived per-seed ratios) |
| OPERAnet | `operanet/matrix_published_rounded.csv` (rounded manuscript values) | `operanet/pose_observability_summary.csv` (4 modality aggregates) |
| MM-Fi | `mmfi/v1_12frame_matrix.csv`, `mmfi/v2_48frame_matrix.csv` | `mmfi/pose_observability_summary.csv` (10 modality–target aggregates) |

The newly added observability summaries report mean and sample standard deviation over seeds 11, 22, 33. They were transcribed from the verified Omni terminal integrity output on 2026-11-13. The corresponding 12 OPERAnet and 30 MM-Fi original per-seed CSV rows and their provenance-bearing JSON files are **not included** in this archive; therefore these committed aggregates cannot presently be independently recomputed from raw seed records.

The real runs used the existing `aligned_v7r1_10` and `compact_v2_48` caches on Omni. The checks verified all 42 per-seed records were present, numerical values were finite, and `H = R_E - R_O` held. The accompanying [protocol](../experiments/observability/README.md) describes the experiment and explains why these pose-motion proxy measurements are not numerically interchangeable with XRFv2's IMU-motion measurements.

Additional archived XRFv2 materials include `xrfv2-plus/room_analysis.txt` and `xrfv2-plus/synthetic_controls.txt`. The original cache data and model checkpoints are not bundled. Not all historical diagnostic experiments referenced by the paper are reproduced by scripts in this repository.
