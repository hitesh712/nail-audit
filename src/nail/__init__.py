"""Leakage-free nail disease classification pipeline."""

__version__ = "0.2.0"

# The taxonomy the draft reports. Reproducible exactly from the two Kaggle
# sources, and retained only as a control condition - see PAPER_MERGES.
PAPER_CLASSES = [
    "Acral_Lentiginous_Melanoma",
    "Blue_Finger",
    "Clubbing",
    "Healthy_Nail",
    "Onychogryphosis",
    "Pitting",
]

# The taxonomy actually present on disk once the source folders are read
# literally. Onychomycosis is a fungal nail infection; onychogryphosis is a
# gross thickening deformity ("ram's horn nail"). They are different diagnoses
# and the draft merges them.
CORRECTED_CLASSES = [
    "Acral_Lentiginous_Melanoma",
    "Blue_Finger",
    "Clubbing",
    "Healthy_Nail",
    "Onychogryphosis",
    "Onychomycosis",
    "Pitting",
    "Psoriasis",
]

# How the draft's six classes were built from the eight real ones.
# Reproducing this is the only way to compare like with like against the
# reported 97.67%.
PAPER_MERGES = {
    "Onychomycosis": "Onychogryphosis",  # clinically wrong; 722 images
    "Psoriasis": None,                    # silently dropped; 434 images
}

TAXONOMIES = {
    "corrected": CORRECTED_CLASSES,
    "paper": PAPER_CLASSES,
}


def class_to_idx(classes: list[str]) -> dict[str, int]:
    return {c: i for i, c in enumerate(classes)}


# Backwards-compatible default used where a taxonomy is not passed explicitly.
CLASSES = CORRECTED_CLASSES
CLASS_TO_IDX = class_to_idx(CLASSES)
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}
