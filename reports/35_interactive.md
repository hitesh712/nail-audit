# Interactive refinement, as specified in the draft

Model `efficientnet_b0`. Corrections stream from the validation split (191 images); retention measured on the held-out test split (191), which no update touches. 3 stream orderings.

Baseline retention accuracy before any update: **92.15%**

| condition | corrections | final retention | worst retention | forgetting |
| --- | --- | --- | --- | --- |
| as specified (LR 0.01, buffer) | 160 | 37.35% ± 2.88 | 13.96% | **+54.80 pp** |
| LR 0.01, no buffer | 153 | 17.45% ± 2.12 | 13.61% | **+74.69 pp** |
| LR 1e-4, buffer | 40 | 91.62% ± 0.00 | 90.58% | **+0.52 pp** |
| LR 1e-4, no buffer | 61 | 76.61% ± 4.95 | 73.82% | **+15.53 pp** |

## Verdict

**The procedure as specified is unsafe.** Applying the draft's own settings costs 54.80 percentage points of accuracy on data the model previously handled correctly. A mechanism that degrades the model while appearing to 'learn' from corrections cannot be deployed, and the abstract should not advertise it.

Compare the learning-rate rows: the specified 1e-2 against 1e-4 under an
otherwise identical protocol isolates whether the rate, rather than the idea,
is the problem. The buffer rows isolate whether replay prevents forgetting.

## Caveat

The correction stream is the validation split, which was used for checkpoint
selection though never for weight updates. A genuinely prospective stream
would be preferable; this is the closest available without new data.