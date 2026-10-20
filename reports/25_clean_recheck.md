# Sensitivity to rotation-group contamination

3 seeds. Test partition n=546, of which **7** (1.3%) share a rotation-invariant duplicate group with training data.

| metric | full test set | contaminated removed | change |
| --- | --- | --- | --- |
| balanced_accuracy | 0.9881 ± 0.0041 | 0.9880 ± 0.0042 | **-0.01 pp** |
| macro_f1 | 0.9889 ± 0.0043 | 0.9888 ± 0.0044 | **-0.01 pp** |
| accuracy | 0.9890 ± 0.0048 | 0.9889 ± 0.0049 | **-0.01 pp** |

Accuracy on the 7 contaminated images alone: 100.0%.

## Verdict

Removing the contaminated images changes accuracy by 0.01 percentage points. **No conclusion in this paper depends on them.** We report the clean figure and disclose the contamination rate; rebuilding the split would not alter any finding.

The reportable number is the one measured with contaminated images
excluded, with the contamination rate stated alongside it.