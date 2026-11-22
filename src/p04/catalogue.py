"""Fixed class catalogue for the mechanical-component vision inspection project.

Target class IDs 1-5 are fixed and must not be changed during testing.
Class ID 0 is the reserved rejection label ``non_catalogue``; it is a
required rejection label, not a sixth target class.
"""

TARGET_CLASSES = [
    {"class_id": 1, "class_name": "long_screw",  "size": "M4 x 25"},
    {"class_id": 2, "class_name": "thick_screw", "size": "M5 x 10"},
    {"class_id": 3, "class_name": "thin_screw",  "size": "M4 x 10"},
    {"class_id": 4, "class_name": "hex_nut",     "size": "M4"},
    {"class_id": 5, "class_name": "flat_washer", "size": "M5"},
]

NON_CATALOGUE = {"class_id": 0, "class_name": "non_catalogue"}

ALL_CLASSES = [NON_CATALOGUE] + TARGET_CLASSES

TARGET_IDS = [c["class_id"] for c in TARGET_CLASSES]
NON_CATALOGUE_ID = 0

MODEL_ID = "baseline1"

LABEL_PASS = "PASS"
LABEL_FAIL = "FAIL"
IMAGE_LABELS = [LABEL_PASS, LABEL_FAIL]


def class_name(class_id: int) -> str:
    """Return the fixed class name for a class ID."""
    for c in ALL_CLASSES:
        if c["class_id"] == class_id:
            return c["class_name"]
    raise KeyError(f"Unknown class_id: {class_id}")
