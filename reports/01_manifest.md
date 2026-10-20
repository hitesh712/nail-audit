# Manifest

- Taxonomy: **corrected** (8 classes)
- Files found: **5300**
- Mapped to a canonical class: **5300**
- Unmapped: **0**
- Unreadable/corrupt: **0**

## Splits the sources shipped with

The draft did not construct a split. These are the folders the two
Kaggle datasets arrived in, and the draft's Table II reproduces them
exactly - which is why its per-class validation fractions range from
1.5% to 12.6%.

| class                      | ('gurav', 'train') | ('gurav', 'val') | ('rasanjana', 'test') | ('rasanjana', 'train') |
|----------------------------|--------------------|------------------|-----------------------|------------------------|
| Acral_Lentiginous_Melanoma | 735                | 18               | 0                     | 0                      |
| Blue_Finger                | 603                | 9                | 0                     | 0                      |
| Clubbing                   | 767                | 16               | 0                     | 0                      |
| Healthy_Nail               | 323                | 20               | 62                    | 248                    |
| Onychogryphosis            | 677                | 12               | 0                     | 0                      |
| Onychomycosis              | 0                  | 0                | 147                   | 575                    |
| Pitting                    | 639                | 16               | 0                     | 0                      |
| Psoriasis                  | 0                  | 0                | 91                    | 342                    |

## Mapped images per class per source

| class                      | gurav | rasanjana | total |
|----------------------------|-------|-----------|-------|
| Acral_Lentiginous_Melanoma | 753   | 0         | 753   |
| Blue_Finger                | 612   | 0         | 612   |
| Clubbing                   | 783   | 0         | 783   |
| Healthy_Nail               | 343   | 310       | 653   |
| Onychogryphosis            | 689   | 0         | 689   |
| Onychomycosis              | 0     | 722       | 722   |
| Pitting                    | 655   | 0         | 655   |
| Psoriasis                  | 0     | 433       | 433   |
