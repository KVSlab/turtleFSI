# Completed simulation datasets

Use `simulations/<problem>-<requested-duration>-<material>/`.
Each dataset contains `Checkpoint/default_variables.json` and the original
`ROM/dvp_snapshots.npz` and `ROM/dvp_snapshots_summary.json` exports.
`stvk` means St. Venant–Kirchhoff; numerical stiffness values stay in the metadata.

| Dataset | Saved end (s) | Snapshots | Supremizers |
| --- | --- | --- | --- |
| aneurysm-20ms-stvk | 0.021 | 21 | No |
| aneurysm-50ms-stvk | 0.051 | 51 | Yes |
| aneurysm-200ms-stvk | 0.201 | 201 | No |
| aneurysm-500ms-stvk | 0.501 | 501 | No |
| arterial-stenosis-20ms-stvk | 0.021 | 21 | Yes |
| arterial-stenosis-200ms-stvk | 0.201 | 201 | Yes |
| flag-pole-100ms-stvk | 0.11 | 11 | No |
| flag-pole-10s-stvk | 10.01 | 1001 | No |

All eight archives were checked for finite arrays, dimensions consistent with
summaries, increasing times/counters, time-step consistency, and coverage of the
requested duration. Each has one extra solver timestep. SHA-256 hashes and the
cleanup record are in `notes/simulation-cleanup.json`.

The two incomplete 30-second flag runs were removed. The reusable integration
check is now `tests/manual/flag_paper_check.py`.

## ROM results

Save ROM experiments under `rom-results/<source-campaign-or-dataset>/` at the repository root. Full-order snapshot exports remain under each simulation’s `ROM/` directory.

## Paper suitability

These are completed StVK datasets useful for development, not validated
reproductions of the paper's linear-elastic model. The paper workflow needs new
linear-elastic, paper-extension runs with the required supremizer exports.
Completion and finite values alone do not establish physical accuracy.

Obsolete ROM experiments, failed diagnostic outputs, old logs, and migration
metadata were deleted after approval. Only the eight completed datasets remain.
