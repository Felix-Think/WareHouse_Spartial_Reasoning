---
license: cc-by-4.0
---
# Physical AI Spatial Intelligence Warehouse

## Overview
The Physical AI Spatial Intelligence Warehouse is a comprehensive synthetic dataset designed to advance 3D scene understanding in warehouse environments. Generated using NVIDIA's Omniverse, this dataset focuses on spatial reasoning through natural language question-answering pairs that cover four key categories: spatial relationships (left/right), multi-choice questions, distance measurements, and object counting. Each data point includes RGB-D images, object masks, and natural language Q&A pairs with normalized single-word answers. The annotations are automatically generated using rule-based templates and refined using LLMs for more natural language responses. We hope this dataset will inspire new research directions and innovative solutions in warehouse automation, from intelligent inventory management to advanced safety monitoring.

## Dataset Description

### Dataset Owner(s)
NVIDIA

## Dataset Creation Date:
We started to create this dataset in January 2025. 

### Dataset Characterization
- Data Collection Method:
    - Synthetic: RGB images, depth images
- Labeling Method: 
    - Automatic:
        - Object tags: Automatic with IsaacSim / Omniverse
        - Region masks: Automatic with IsaacSim / Omniverse
        - Text annotations, question-answer pairs: Automatic with rule-based template, optionally refined with Llama-3.1-70B-Instruct (subject to redistribution and use requirements in the Llama 3.1 Community License Agreement at https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct/blob/main/LICENSE.).

## Dataset Quantification
We have 499k QA pairs as training set, with 19k QA pairs for testing, and 1.9k QA for validation. The data also comes with around 95k RGB-D image pairs in total.
Questions cover 4 major categories:
- `left_right`: understand the spatial relationship between different objects / regions
- `multi_choice_question(mcq)`: identify the index of target from multiple candidate objects / regions
- `distance`: estimate the distance (in meters) between different objects / regions
- `count`: ask about the number of certain type of objects that satisifies the condition (leftmost, specific categories)

### Directory Structure
```shell
├── train
│   ├── depths
│   │   ├── <frame_id1>_depth.png
|   |   |    ...
│   │   └── <frame_idn>_depth.png
│   └── images
│       ├── <frame_id1>.png
|       |    ...
│       └── <frame_idn>.png
├── val
│   ├── depths
│   └── images
├── test
│   ├── depths
│   └── images
├── train.json
├── val.json
└── test.json
```

### Annotation Format (3D-VLM-Challenge) for `Warehouse Spatial Intelligence`
Annotations are provided in the `train.json`, `val.json`, containing multiple single-round QnA pair with related meta information following LLaVA[1] format for VLM training. 
In addition to that, 
- we provide `normalized_answer` field for quantitative evaluation with accuracy and error metrics between Ground-truth and predicted answer
- the original answer from 'gpt' becomes `freeform_answer` field
- `rle` denotes the corresponding masks per object in order following pycoco format (we provide sample code for loading)
- `category` denotes the question category
Note that `test.json` only contains `id`, `image`, `"conversations"`, and `rle` fields

See below for detailed example.
```json
{
  "id": "9d17ba0ab1df403db91877fe220e4658",
  "image": "000190.png",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nCould you measure the distance between the pallet <mask> and the pallet <mask>?"
    },
    {
      "from": "gpt",
      "value": "The pallet [Region 0] is 6.36 meters from the pallet [Region 1]."
    }
  ],
  "rle": [
    {
    "size": [
        1080,
        1920
    ],
    "counts": "bngl081MYQ19010ON2jMDmROa0ol01_RO2^m0`0PRODkm0o0bQOUO[n0U2N2M3N2N2N3L3N2N1N1WO_L]SOa3el0_LYSOb3il0]LTSOf3ll0ZLRSOh3nl0XLPSOj3Pm0VLmROn3Rm0SLkROo3Um0QLiROQ4Wm081O00N3L3N2O10000010O0000000001O01O00000001O10O01O003M2N0010O0000000001O01O00000M3N201N1001O00000001O01O000001O0001O000000000010O00000000010O0002N00001O3N1N000000000001O000000000O2M200O1M3N20001CQSOoKol0n3TSORLll0k3WSOVLhl0g3[SOYLel0f3\\SOZLdl0c3_SO]Lal0Z2nROcNe0RO^l0\\1kSO_OJUOmn0g0WQOZOhn0a0]QO_Odn0=_QOCan0:bQOF^n08eQOG[n07gQOIYn04jQOMUn00nQO0d[nm0"
    },
    {
    "size": [
        1080,
        1920
    ],
    "counts": "^PmU1j1no000000000000000000001O0000000000001O0000000000001O0000000000001O0000000000001O00000g1YN000001O01O00gNfQOTOZn0d0fQODZn06eQO_N1\\1Zn0OfQO<[n0^OgQOe0Yn0UOmQOl0Rn0oNnQOV1Rn0dNPRO`1Pn0[NTROf1lm0TNZROl1fm0oM_ROQ2en00O000000000M3K6J5K5K5K5N201O0000000000010O000000000010O0000000001O01O00M3K5K5K6N10000001O01O0000000001O01O00000000010OmLWROW2im0dM\\RO\\2dm0`M`RO`2`m0[MeROe2Xn0O0001O00000001O3NO01ON2L4L4O110O000000000010O0000000010O0000000000000001O0eMaQO]1_n0`NdQO`1\\n0\\NiQO_1[n0]NiQOY1an0aNeQOW1bo0H8G9G[oN_OfP19aoNG^fjc0"
    }
  ],
  "category": "distance",
  "normalized_answer": "6.36",
  "freeform_answer": "The pallet [Region 0] is 6.36 meters from the pallet [Region 1]."
}

```

## Usage
#### Getting started
First download the dataset
```shell
# You can also use `huggingface-cli download`
git clone https://huggingface.co/datasets/nvidia/PhysicalAI-Spatial-Intelligence-Warehouse
cd PhysicalAI-Spatial-Intelligence-Warehouse

# we need to untar images for train/test subsets
for dir in train test; do
    for subdir in images depths; do
        if [ -d "$dir/$subdir" ]; then
            echo "Processing $dir/$subdir"
            cd "$dir/$subdir"
            tar -xzf chunk_*.tar.gz
            # rm chunk_*.tar.gz
            cd ../..
        fi
    done
done
```

#### Visualization
```shell
python ./utils/visualize.py \
    --image_folder ./val/images/ \
    --depth_folder ./val/depths/ \
    --annotations_file ./val.json \
    --num_samples 10 
```


#### Evaluation
For sanity check and understand your model performance, you could locally evaluate your results on the provided validation set. We require the submission format (JSON) on test set following below format, in which `id` and `normalized_answer` are all necessary. 

```json
[
    {
        "id": "000123",
        "normalized_answer": "1.22"
    },
    {
        "id": "ab23dm",
        "normalized_answer": "left"
    },
    {
        "id": "ac348d",
        "normalized_answer": "4"
    },
    …
]
```

Suppose you have your prediction results under `utils/assets/perfect_predictions_val.json`, you could check your predictions by: 
```
# sanity check with perfect answer
python ./utils/compute_scores.py \
    --gt_path ./val.json \
    --pred_path ./utils/assets/perfect_predictions_val.json
```


## References
[1] Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). Visual instruction tuning. In NeurIPS.

## Ethical Considerations
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.   

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

## Changelog
- **2025-05-24**: Initial data drop with train/val/test splits
