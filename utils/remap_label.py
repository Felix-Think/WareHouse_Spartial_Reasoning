from pathlib import Path


def remap_labels(labels_dir):
    label_dir = Path(labels_dir)

    for label_file in label_dir.glob("*.txt"):
        new_lines = []

        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                idx = int(parts[0])

                # drop buffer
                if idx == 0:
                    continue

                new_idx = idx - 1

                new_line = " ".join([str(new_idx)] + parts[1:])
                new_lines.append(new_line)

        with open(label_file, "w") as f:
            f.write("\n".join(new_lines))


if __name__ == "__main__":
    remap_labels("datasets/train/labels")
    remap_labels("datasets/valid/labels")  # rất nên remap luôn
