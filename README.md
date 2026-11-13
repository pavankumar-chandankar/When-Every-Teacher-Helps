# When Every Teacher Helps

Code and aggregate results for *When Every Teacher Helps: A Post-Mortem of Cross-Modal Learning for WiFi Activity Recognition*.

The classification experiments cover XRFv2 Plus (30 directed teacher–student pairs), OPERAnet (12), and MM-Fi (20). MM-Fi matrices include both 12-frame and 48-frame sampling.

`experiments/xrfv2-plus/`, `experiments/operanet/`, and `experiments/mmfi/` contain the available matrix training and preprocessing code. `experiments/observability/` contains the additional OPERAnet and MM-Fi pose-motion observability experiment, with a reproducibility protocol and synthetic tests. `results/` contains classification matrices, XRFv2 observability records, and the newly reported pose-motion observability aggregates.

The cross-dataset observability targets are **not the same physical quantity**: XRFv2 uses IMU-derived motion references, while OPERAnet and MM-Fi use Kinect/RGB-pose displacement proxies. See [the observability protocol](experiments/observability/README.md) for precise targets, splits, and limitations.

Validate the committed summaries using:

```bash
python checks/check_results.py
python experiments/observability/pose_observability.py self-test
python experiments/observability/smoke_test.py
```

The consistency checker uses standard-library Python; the observability tests require NumPy and PyTorch. Other matrix scripts have additional requirements listed in `requirements.txt`. Original datasets, intermediate caches, checkpoint weights, and full OPERAnet/MM-Fi observability per-seed files are not bundled. See [results inventory](results/README.md).

Code license: CC0-1.0. Dataset licenses remain with their respective providers.
