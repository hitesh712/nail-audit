# Orientation-invariant duplicate audit

pHash Hamming <= 6, matched under all 8 dihedral
transforms (4 rotations x 2 mirrorings), computed per source.

| source    | images | groups_plain | groups_dihedral | redundant_plain | redundant_dihedral | pct_redundant_plain | pct_redundant_dihedral | extra_pairs_from_orientation |
|-----------|--------|--------------|-----------------|-----------------|--------------------|---------------------|------------------------|------------------------------|
| gurav     | 3835   | 3824         | 3790            | 11              | 45                 | 0.300               | 1.200                  | 34                           |
| rasanjana | 1465   | 1111         | 1078            | 354             | 387                | 24.200              | 26.400                 | 94                           |

## Reading this

If `redundant_dihedral` greatly exceeds `redundant_plain` for a source,
that source was distributed with orientation-augmented copies. Those
copies are invisible to a standard deduplication and would sit on both
sides of any split, including the group-aware one used elsewhere in this
pipeline.

Where that holds, every accuracy reported on that source - including this
pipeline's own group-aware numbers - needs recomputing with
`group_id_dihedral` as the split unit.