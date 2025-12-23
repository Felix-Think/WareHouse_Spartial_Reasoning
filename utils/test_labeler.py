import re
import json

VALID_CLASSES = {"pallet", "transporter", "shelf", "buffer"}


def infer_mask_classes_general(question):
    q = question.lower()

    tokens = re.findall(r"(pallets?|transporters?|shelves?|buffers?|<mask>)", q)

    classes = []
    current_class = None

    for tok in tokens:
        if tok in ["pallet", "pallets"]:
            current_class = "pallet"
        elif tok in ["transporter", "transporters"]:
            current_class = "transporter"
        elif tok in ["shelf", "shelves"]:
            current_class = "shelf"
        elif tok in ["buffer", "buffers"]:
            current_class = "buffer"
        elif tok == "<mask>":
            classes.append(current_class)

    return classes


JSON_PATH = "PhysicalAI_Warehouse/train_new.json"
with open(JSON_PATH, "r") as f:
    data = json.load(f)
TARGET_NAME = "065627.png"
samples = data[:1000]
for idx, sample in enumerate(samples):
    if sample["image"] == TARGET_NAME:

        print(sample["conversations"])
        q = sample["conversations"][0]["value"]
        print(infer_mask_classes_general(q))
