# Real-World Test Results (final — 2026-06-12)

## The short version

I froze my best lab models and ran them on real drives. All of them got
much worse, and the hybrid model — the lab winner — was no better than
the sensor model alone, because its camera half went blind on dashcam
video. I labeled every detection clip without knowing which model made
it, and I did the whole labeling twice to check myself. The conclusion
came out the same both times.

The experiment rules were written down before any labeling started
([docs/TRANSFER_PROTOCOL.md](../../docs/TRANSFER_PROTOCOL.md)), so the
rules could not be bent after seeing results. The raw numbers behind
everything on this page are in [metrics.md](metrics.md).

**Two stats terms used below:**
- **Precision** — out of all the alerts a model made, what fraction were
  real potholes. Precision 0.10 means 1 alert in 10 was real.
- **95% confidence range** — because I only labeled a few hundred clips,
  each number could be off a bit. The range in brackets shows how far
  off it could reasonably be. Fewer examples = wider range.

## Main result

Six real iPhone drives. The models were used exactly as trained — no
tweaking allowed:

| Model | Alerts per minute | Precision [95% range] | Real / false / unsure |
|---|---|---|---|
| Camera only | 0.27 | 0.00 (only 1 labeled alert) | 0 / 1 / 0 |
| Sensor only | 8.1 | 0.106 [0.057, 0.189] | 9 / 76 / 58 |
| Hybrid | 6.8 | 0.081 [0.038, 0.166] | 6 / 68 / 45 |

What this says:

- **The hybrid model did not beat the sensor model.** Its precision was
  actually a little lower, and the difference between the two is small
  enough that it could easily be luck (a paired statistics test puts the
  chance well above the standard cutoff — p = 0.51, where anything above
  0.05 means "could be luck").
- The camera model found **zero** real potholes on its own. The sensor
  and hybrid models clearly beat it, but that says the camera failed,
  not that fusion helped.
- The hidden probe clips (random quiet moments mixed into my labeling,
  to catch potholes the models missed): 0 of 46 contained a pothole. So
  the models weren't obviously missing potholes left and right — at
  most about 8% of random moments could contain a missed one, given how
  many probes I checked.

## I labeled everything twice — did I agree with myself?

After my first labeling pass, I worried I had been inconsistent, so I
relabeled all 266 clips from scratch (still blinded). Both rounds are
saved in this folder.

- The big numbers barely moved: sensor precision 0.100 → 0.106, hybrid
  0.102 → 0.081. **The conclusion was the same both times: no hybrid
  advantage.**
- On clips where I gave a definite answer both times, I agreed with
  myself **97% of the time**. Almost all of my changes were clips I
  moved into "unsure" in round 2 — I got more careful, not different.
  Only 5 clips out of 260 actually flipped between "pothole" and "not
  pothole."

## Raising the alert bar doesn't fix it

You might think: just make the models pickier — only count their most
confident alerts. I checked. It makes things **worse**:

| Confidence required | Sensor precision | Hybrid precision |
|---|---|---|
| 0.5 (normal) | 0.10 | 0.10 |
| 0.7 (pickier) | 0.05 | 0.04 |
| 0.8–0.9 (pickiest) | 0.00 | 0.00 |

The models' most confident alerts were almost all wrong. On real roads,
their confidence went up on the *hardest shakes* — speed bumps, rough
patches, expansion joints — not on potholes. That matches what I found
in the lab: bumps shake a car harder than potholes do. This means the
problem can't be fixed with a settings change. The models need better
training data.

## Why combining sensors didn't help

The fusion model was trained to trust what the camera model sees. But
the camera model was trained on close-up pothole photos, and real
dashcam video (wide view, motion blur, night driving) looks nothing
like that — so on real drives the camera was effectively blind. It made
19 alerts across every drive combined, with zero confirmed potholes.

When one half of a hybrid system is blind, "combining" just adds noise.
On the night drives, the fusion model overruled the sensor's alerts
about as often when the sensor was right as when it was wrong.

**The lesson: a hybrid system is only as strong as its weakest sensor
in the new environment.**

## The nuScenes drives

The 10 self-driving-car scenes (Boston and Singapore) were short and
smooth: 5 total alerts, none confirmed. Too little data to compare
models there, so I report it for completeness only. Both kinds of
sensors transferred even worse to the more-different vehicle.

## What's honest to claim

1. In the lab, hybrid fusion won — by a little (F1 0.622 vs 0.605
   across 94 tests).
2. On real roads, every model got much worse, and the hybrid advantage
   disappeared. I found the same thing in two separate labeling rounds.
3. The takeaway for anyone building car tech: **combining sensors is
   not a free win.** If one sensor doesn't work in the real world, the
   combination won't either. The fix is better training data for the
   weak sensor (for me: dashcam-style pothole images), not a cleverer
   way of combining.

## Limits of this study

One person did the labeling (me — blinded, and double-checked against
myself, but a second labeler would be stronger). The "missed potholes"
estimate comes from a sample of quiet moments, not from checking every
second of video. Six drives from one car and one phone. Night driving
was rare in the training data. And the camera model's training photos
were a known weak point going in — I wrote that down in the plan before
running the test.
