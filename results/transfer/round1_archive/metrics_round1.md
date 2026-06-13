# Transfer-gap metrics (manual review)

Primary reviewer: `TS` — 260/837 items labeled. Reviewers: {'TS': 260}

## iphone

| pipeline | events | labeled | TP | FP | unsure | precision | 95% CI | events/min | est. missed | est. recall* |
|---|---|---|---|---|---|---|---|---|---|---|
| image | 15 | 1 | 0 | 1 | 0 | 0.000 | [0.00, 0.79] | 0.27 | 30.1 | 0.0 |
| sensor | 455 | 140 | 14 | 126 | 0 | 0.100 | [0.06, 0.16] | 8.08 | 4.7 | 0.748 |
| hybrid | 383 | 118 | 12 | 106 | 0 | 0.102 | [0.06, 0.17] | 6.8 | 12.0 | 0.5 |

Probes: 55 labeled, 2 contained a pothole (background rate 0.036, unsure 0).

## nuscenes

| pipeline | events | labeled | TP | FP | unsure | precision | 95% CI | events/min | est. missed | est. recall* |
|---|---|---|---|---|---|---|---|---|---|---|
| image | 4 | 2 | 0 | 2 | 0 | 0.000 | [0.00, 0.66] | 2.07 | 7.5 | 0.0 |
| sensor | 0 | 0 | 0 | 0 | 0 | — | — | 0.0 | 8.7 | 0.0 |
| hybrid | 2 | 1 | 0 | 1 | 0 | 0.000 | [0.00, 0.79] | 1.04 | 8.0 | 0.0 |

Probes: 10 labeled, 3 contained a pothole (background rate 0.300, unsure 0).

## all

| pipeline | events | labeled | TP | FP | unsure | precision | 95% CI | events/min | est. missed | est. recall* |
|---|---|---|---|---|---|---|---|---|---|---|
| image | 19 | 3 | 0 | 3 | 0 | 0.000 | [0.00, 0.56] | 0.33 | 65.7 | 0.0 |
| sensor | 455 | 140 | 14 | 126 | 0 | 0.100 | [0.06, 0.16] | 7.82 | 12.2 | 0.534 |
| hybrid | 385 | 119 | 12 | 107 | 0 | 0.101 | [0.06, 0.17] | 6.61 | 27.5 | 0.304 |

Probes: 65 labeled, 5 contained a pothole (background rate 0.077, unsure 0).

## Paired comparisons (exact McNemar on pothole-labeled items)

- **hybrid vs image**: discordant {'only_hybrid': 12, 'only_image': 0}, p = 0.0005
- **hybrid vs sensor**: discordant {'only_hybrid': 4, 'only_sensor': 6}, p = 0.7539
- **sensor vs image**: discordant {'only_sensor': 14, 'only_image': 0}, p = 0.0001

\* recall is a probe-sampling ESTIMATE (see docs/TRANSFER_PROTOCOL.md), not exhaustive ground truth.