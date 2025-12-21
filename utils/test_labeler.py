import re

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


q = "among the pallets <mask> <mask> and the transporters <mask> <mask>"

print(infer_mask_classes_general(q))
