# Pose-motion observability

This experiment uses the existing OPERAnet v7r1 and MM-Fi v2_48 aligned caches. It supplements the cross-modal classification matrices; it does not rerun them.

## Data and targets

| Dataset | Cached input | Split (train/dev/test) | Target |
| --- | --- | --- | --- |
| OPERAnet | `aligned_v7r1_10/expNNN_aligned.npz` | 4018 / 1688 / 1474 windows | Mean squared Kinect marker displacement per 0.1-second frame interval, averaged over two 19-marker cameras and 14 intervals within a 15-frame window |
| MM-Fi | `compact_v2_48/SNN.npz` | 864 / 108 / 108 sequences | Mean squared 2D pose displacement divided by original *frame-index* gaps, averaged over all 17 joints or wrist joints 9 and 10 |

The MM-Fi values are in image-coordinate displacement per sampled-frame-index squared, **not** physical velocity per second. The OPERAnet target is a Kinect-derived velocity proxy, **not** wearable acceleration. Neither target is interchangeable with the XRFv2 IMU-motion targets. Controls (`kinect` and `rgbpose`) are self-reference inputs and are flagged as such.

Subject-disjoint training/development/test splits follow the corresponding matrix cache. The original datasets and model weights are not included.

## Training and evaluation

For each available modality and target, train two separate encoders on training subjects: a motion regressor (O) and an activity classifier (E). The networks follow the corresponding matrix-model architecture family, but the encoders are **newly trained**: previously trained matrix weights were not reused. There are 30 epochs per encoder, batch size 64, and seeds 11, 22, 33. Activity checkpoint selection uses development accuracy; motion checkpoint selection uses development MSE. This checkpoint criterion can differ from the matrix experiment's development macro-F1 criterion.

Normalize input features using training-subject statistics. Normalize the motion target using training-subject moments when fitting O. Freeze each trained encoder, then fit separate ridge readouts from its penultimate embeddings to the motion target, using training-subject samples and development-selected ridge penalties.

The label-only reference predicts the *training-subject mean target for the true activity class*. On held-out test subjects, the test MSEs are `Err_O`, `Err_E`, and `Err_oracle`. Report:

- `R_O = Err_O / Err_oracle`
- `R_E = Err_E / Err_oracle`
- `H = R_E - R_O`

Values below one indicate that the tested readout achieves lower error than the class-conditional label-only reference. A positive H by itself does **not** demonstrate useful motion observability. Observability refers to tested models and proxies, not an inherent information-theoretic sensor limit.

## Validation and provenance

The archived script checks dataset layouts, subject-disjoint splits, input completeness, finite losses and readouts, and refuses to overwrite existing output files. Each per-seed output JSON records SHA-256 hashes for processed NPZ inputs. The original 42 per-seed output rows and JSON files are currently retained on Omni; the checked-in `results/{operanet,mmfi}/pose_observability_summary.csv` files contain **only the ten aggregates transcribed from the verified terminal summary**. They are not substitutes for the raw experiment outputs, and must not be described as independently recomputed from archived per-seed values.

The two real runs were conducted on Omni with 30 epochs, seeds 11/22/33, batch 64, CUDA, and the input caches described above. SLURM jobs: OPERAnet `12446086`, MM-Fi `12446087`. Their logs reported 12 and 30 records, respectively; an independent terminal integrity check confirmed each group's three seeds, finite metrics, and H consistency. Job accounting state and complete source data hashes have not been supplied for repository inclusion.

## Reproduction

Dependencies: Python 3.11, NumPy, PyTorch 2.x. On a login node, use only the read-only preflight and synthetic tests:

```bash
python experiments/observability/pose_observability.py self-test
python experiments/observability/smoke_test.py
python experiments/observability/pose_observability.py preflight --dataset operanet --input /path/to/aligned_v7r1_10
python experiments/observability/pose_observability.py preflight --dataset mmfi --input /path/to/compact_v2_48
```

To reproduce full GPU training, submit a new job using `experiments/observability/run_observability.sbatch` with `DATASET`, `INPUT_DIR`, `OUTPUT_DIR`, `PYTHON_BIN`, and `SCRIPT_DIR` exported to the job. The output directory must not contain any existing experiment result. Avoid training on the login node. No MATLAB is used.

The results are exploratory rather than direct reproductions of the frozen XRFv2 IMU observability protocol. Do not pool physical quantities across datasets.
