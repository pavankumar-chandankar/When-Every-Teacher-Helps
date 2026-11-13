# When Every Teacher Helps

Experiments and aggregate results for *When Every Teacher Helps: A Post-Mortem of Cross-Modal Learning for WiFi Activity Recognition*.

The study compares cross-modal classification on XRFv2 Plus (30 directed pairs), OPERAnet (12), and MM-Fi (20). The MM-Fi matrix was run with 12 and 48 sampled frames.

- `experiments/` — preprocessing, training, and figure code
- `results/` — reported classification matrices and available diagnostic summaries
- `checks/` — consistency checks against the manuscript

Run `python checks/check_results.py` to validate the reported matrix summaries. The experiments require the original datasets and their respective Python dependencies; no raw data or model weights are included.

Some original run files were not recovered from OMNI. See `results/README.md` for what is available. The OPERAnet v7r1 source is reconstructed from the archived v7 source and correction script. It is not an independently verified copy of the cluster snapshot.

Code: CC0-1.0 (existing repository license). Dataset use is governed by the individual dataset licenses.
