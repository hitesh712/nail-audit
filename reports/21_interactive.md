# Interactive refinement, as specified in the draft

Model `efficientnet_b0`. Corrections stream from the validation split (554 images); retention measured on the held-out test split (546), which no update touches. 3 stream orderings.

Baseline retention accuracy before any update: **98.53%**

| condition | corrections | final retention | worst retention | forgetting |
| --- | --- | --- | --- | --- |
| as specified (LR 0.01, buffer) | 474 | 36.02% ± 2.39 | 11.97% | **+62.52 pp** |
| LR 0.01, no buffer | 470 | 18.68% ± 3.12 | 10.99% | **+79.85 pp** |
| LR 1e-4, buffer | 37 | 97.68% ± 1.01 | 97.19% | **+0.85 pp** |
| LR 1e-4, no buffer | 149 | 92.92% ± 1.87 | 89.01% | **+5.62 pp** |

## Verdict

**The procedure as specified is unsafe.** Applying the draft's own settings costs 62.52 percentage points of accuracy on data the model previously handled correctly. A mechanism that degrades the model while appearing to 'learn' from corrections cannot be deployed, and the abstract should not advertise it.

Compare the learning-rate rows: the specified 1e-2 against 1e-4 under an
otherwise identical protocol isolates whether the rate, rather than the idea,
is the problem. The buffer rows isolate whether replay prevents forgetting.

## Caveat

The correction stream is the validation split, which was used for checkpoint
selection though never for weight updates. A genuinely prospective stream
would be preferable; this is the closest available without new data.