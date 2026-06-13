# Transfer-gap metrics (manual review)

Primary reviewer: `TS` — 264/837 items labeled. Reviewers: {'TS': 264}

## iphone

| pipeline | events | labeled | TP | FP | unsure | precision | 95% CI | events/min | est. missed | est. recall* |
|---|---|---|---|---|---|---|---|---|---|---|
| image | 15 | 1 | 0 | 1 | 0 | 0.000 | [0.00, 0.79] | 0.27 | 0.0 | None |
| sensor | 455 | 143 | 9 | 76 | 58 | 0.106 | [0.06, 0.19] | 8.08 | 0.0 | 1.0 |
| hybrid | 383 | 119 | 6 | 68 | 45 | 0.081 | [0.04, 0.17] | 6.8 | 0.0 | 1.0 |

Probes: 46 labeled, 0 contained a pothole (background rate 0.000, unsure 9).

## nuscenes

| pipeline | events | labeled | TP | FP | unsure | precision | 95% CI | events/min | est. missed | est. recall* |
|---|---|---|---|---|---|---|---|---|---|---|
| image | 4 | 2 | 0 | 1 | 1 | 0.000 | [0.00, 0.79] | 2.07 | 0.0 | None |
| sensor | 0 | 0 | 0 | 0 | 0 | — | — | 0.0 | 0.0 | None |
| hybrid | 2 | 1 | 0 | 1 | 0 | 0.000 | [0.00, 0.79] | 1.04 | 0.0 | None |

Probes: 6 labeled, 0 contained a pothole (background rate 0.000, unsure 4).

## all

| pipeline | events | labeled | TP | FP | unsure | precision | 95% CI | events/min | est. missed | est. recall* |
|---|---|---|---|---|---|---|---|---|---|---|
| image | 19 | 3 | 0 | 2 | 1 | 0.000 | [0.00, 0.66] | 0.33 | 0.0 | None |
| sensor | 455 | 143 | 9 | 76 | 58 | 0.106 | [0.06, 0.19] | 7.82 | 0.0 | 1.0 |
| hybrid | 385 | 120 | 6 | 69 | 45 | 0.080 | [0.04, 0.16] | 6.61 | 0.0 | 1.0 |

Probes: 52 labeled, 0 contained a pothole (background rate 0.000, unsure 13).

## Paired comparisons (exact McNemar on pothole-labeled items)

- **hybrid vs image**: discordant {'only_hybrid': 6, 'only_image': 0}, p = 0.0312
- **hybrid vs sensor**: discordant {'only_hybrid': 3, 'only_sensor': 6}, p = 0.5078
- **sensor vs image**: discordant {'only_sensor': 9, 'only_image': 0}, p = 0.0039

\* recall is a probe-sampling ESTIMATE (see docs/TRANSFER_PROTOCOL.md), not exhaustive ground truth.