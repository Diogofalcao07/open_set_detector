# Robust Open-Vocabulary Robot Perception

<p align="center">
  <em>
    <code>Open-set object and human-feature detection for service robotics using GroundingDINO and ChatRex</code>
  </em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="python-version">
  <img src="https://img.shields.io/badge/robotics-perception-blue" alt="robotics-perception">
  <img src="https://img.shields.io/badge/models-GroundingDINO%20%7C%20ChatRex%20%7C%20Detectron2-green" alt="models">
  <img src="https://img.shields.io/badge/task-open--vocabulary%20detection-purple" alt="task">
</p>

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Perception Pipeline](#perception-pipeline)
- [Models Evaluated](#models-evaluated)
- [Experiments](#experiments)
- [Key Results](#key-results)
- [Prompt Sensitivity](#prompt-sensitivity)
- [Human Pose and Gesture Recognition](#human-pose-and-gesture-recognition)
- [Final Repository Structure](#final-repository-structure)
- [Installation](#installation)
- [Environment Variables](#environment-variables)

---

## Overview

This repository presents a robotics perception project focused on improving object and human-feature detection for domestic service robots.

The project compares a traditional closed-set object detector, **Detectron2**, with open-vocabulary vision-language models, **GroundingDINO** and **ChatRex**. The goal is to evaluate whether open-vocabulary perception can help robots recognize unseen objects, understand natural language prompts, and detect human actions such as pointing, waving, sitting, and standing.

The work is motivated by service-robot scenarios where the robot must operate in dynamic household environments and respond to flexible user instructions.

Example tasks include:

```text
Find the candle on the shelf.
Detect the pen, glass, and water bottle.
Give the object to the person who is sitting.
Identify whether the person is pointing or waving.
```

---

## Motivation

Traditional robotic perception systems often rely on **closed-set detection**, where the model can only recognize a fixed set of object classes seen during training.

This is a major limitation for domestic service robots. In real environments, a robot may encounter new objects, uncommon packaging, ambiguous user instructions, or human actions that were not explicitly defined in advance.

This project investigates whether open-vocabulary models can improve robotic perception by using natural language prompts to detect objects and human features beyond a fixed label set.

The project focuses on two main perception problems:

1. **Open-vocabulary object detection**
   - Detecting household objects using natural language prompts.
   - Comparing closed-set and open-vocabulary models.
   - Evaluating generalization to unseen or less frequent objects.

2. **Human-feature and gesture detection**
   - Detecting actions such as sitting, standing, pointing, waving, and laying.
   - Comparing binary pose detection with multi-class pose classification.
   - Studying how ambiguity affects human-action recognition.

---

## Perception Pipeline

The project follows a robotics perception pipeline:

```text
Input image / robot scene
        ↓
Natural language prompt
        ↓
Detection model
        ↓
Bounding boxes and labels
        ↓
Object or human-feature prediction
        ↓
Evaluation and visualization
```

The pipeline evaluates both model accuracy and practical deployment factors such as prompt design, inference time, and GPU memory usage.

---

## Models Evaluated

### Detectron2

Detectron2 is used as the closed-set object detection baseline. It detects objects from a fixed set of known categories and represents the type of detector commonly used in many robotics pipelines.

### GroundingDINO

GroundingDINO is an open-vocabulary object detector that grounds natural language prompts to image regions. It can detect objects described in text without requiring task-specific retraining.

### ChatRex

ChatRex is a multimodal vision-language model evaluated for both object detection and human-feature understanding. It combines visual proposals with language reasoning, making it suitable for prompt-based perception tasks.

---

## Experiments

The project contains four main experimental components.

### 1. Object Detection Evaluation

The models were evaluated using COCO-style metrics:

- Average Precision, AP
- mean Average Precision, mAP
- Average Recall, AR
- mean Average Recall, mAR

The goal was to compare detection quality across closed-set and open-vocabulary systems.

### 2. Open-Vocabulary Comparison

Detectron2, GroundingDINO, and ChatRex were compared on:

- shared closed-set categories
- open-vocabulary categories
- visually distinctive objects
- less frequent or fine-grained household objects

This experiment tests whether open-vocabulary models can generalize better than a closed-set detector.

### 3. Prompt Sensitivity

ChatRex was evaluated using six prompt styles, ranging from restrictive prompts with exact object labels to generic open-ended prompts.

This experiment studies how prompt wording affects:

- detection performance
- inference time
- reliability
- practical usability in robotics tasks

### 4. Human Pose and Gesture Recognition

ChatRex was evaluated on human-centered images with pose and action labels.

Two tasks were tested:

- **Binary pose detection:** decide whether a specific pose is present.
- **Multi-class pose classification:** choose the correct pose from a set of possible actions.

The evaluated actions were:

```text
waving
sitting
laying
standing
pointing
```

---

## Key Results

### Shared Closed-Set Categories

| Model | mAP | mAR | Improvement |
|---|---:|---:|---:|
| Detectron2 | 0.381 | 0.418 | — |
| ChatRex | 0.556 | 0.649 | +45.7% mAP, +55.3% mAR |
| GroundingDINO | 0.599 | 0.668 | +57.2% mAP, +59.9% mAR |

Both open-vocabulary models improved over the Detectron2 closed-set baseline on shared categories.

---

### Open-Vocabulary Categories

| Model | mAP | mAR | Improvement |
|---|---:|---:|---:|
| Detectron2 | 0.256 | 0.276 | — |
| ChatRex | 0.720 | 0.772 | +181.0% mAP, +179.3% mAR |
| GroundingDINO | 0.616 | 0.649 | +140.3% mAP, +134.8% mAR |

The open-vocabulary evaluation showed a large improvement over the closed-set baseline. ChatRex achieved the strongest overall performance in this setting.

---

## Prompt Sensitivity

Prompt design had a major effect on both detection quality and inference time.

| Prompt | Description | Avg. Time / Image |
|---|---|---:|
| P1 | Exact target labels | 1.60 s |
| P2 | Full household label set | 12.44 s |
| P3 | Identify all visible objects | 10.05 s |
| P4 | Natural open-ended detection prompt | 6.52 s |
| P5 | Scene description prompt | 4.50 s |
| P6 | Very generic prompt | 3.27 s |

The results show that prompt specificity matters. Very long prompts can increase inference time, while overly generic prompts may reduce detection reliability. Well-scoped prompts are more suitable for robotics tasks where both accuracy and speed are important.

---

## Human Pose and Gesture Recognition

### Binary Pose Detection

| Pose | Accuracy |
|---|---:|
| Waving | 93.75% |
| Sitting | 90.62% |
| Laying | 96.88% |
| Standing | 75.00% |
| Pointing | 81.25% |

### Multi-Class Pose Classification

| Pose | Accuracy |
|---|---:|
| Waving | 90.62% |
| Sitting | 93.75% |
| Laying | 100.00% |
| Standing | 100.00% |
| Pointing | 96.88% |

Multi-class classification performed better than binary detection for several poses. This suggests that forcing the model to choose the most distinctive action can reduce ambiguity between visually similar poses, such as standing, pointing, and waving.

---

## Final Repository Structure

```text
open_set_detector/
├── README.md
├── .gitignore
├── .env.example
├── requirements-lite.txt
│
├── configs/
│   ├── prompts.yaml
│   └── labels_socrob.yaml
│
├── docs/
│   ├── pipeline.md
│   └── evaluation_methodology.md
│
├── media/
│   ├── detector_comparison.png
│   ├── model_comparison_results.png
│   ├── prompt_sensitivity_heatmap.png
│   └── human_pose_results.png
│
├── results/
│   ├── figures/
│   │   ├── single_image_detection.png
│   │   ├── ap_ar_comparison.png
│   │   └── prompt_timing.png
│   │
│   └── tables/
│       ├── closed_set_results.csv
│       ├── open_vocabulary_results.csv
│       ├── prompt_timing_results.csv
│       └── pose_recognition_results.csv
│
├── scripts/
│   ├── check_environment.py
│   ├── run_single_image.py
│   ├── evaluate_coco.py
│   └── summarize_results.py
│
├── perception_open_set_detector/
│   ├── __init__.py
│   │
│   ├── classes/
│   │   ├── dataset.py
│   │   ├── detection.py
│   │   ├── image.py
│   │   ├── model.py
│   │   ├── prompt_stats.py
│   │   └── prompts.py
│   │
│   ├── utils/
│   │   ├── core_functions.py
│   │   └── helper_functions.py
│   │
│   ├── tests/
│   │   ├── evaluate_prompts.py
│   │   └── test_evaluate_prompts.py
│   │
│   ├── coco_data/
│   │   ├── merged_annotations.coco.json
│   │   ├── test_prompt_1_detector.json
│   │   ├── test_prompt_2_detector.json
│   │   ├── test_prompt_3_detector.json
│   │   ├── test_prompt_4_detector.json
│   │   ├── test_prompt_5_detector.json
│   │   └── test_prompt_6_detector.json
│   │
│   ├── compare_models/
│   │   ├── comparison_code.py
│   │   ├── evaluation_results_detectron.json
│   │   └── test_open_question_detector.json
│   │
│   ├── images/
│   │   └── test_single_image_detection.jpeg
│   │
│   ├── raw_data/
│   │   ├── evaluation_results_model.json
│   │   ├── model_detections.pkl
│   │   └── performance_prompts_summary.json
│   │
│   ├── evaluator.py
│   ├── run_detections.py
│   └── run_and_visualize_detection.py
│
└── chatrex/
    └── ChatRex model code and tools
```

---

## Installation

Clone the repository with submodules:

```bash
git clone --recurse-submodules https://github.com/Diogofalcao07/open_set_detector.git
cd open_set_detector
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install lightweight dependencies:

```bash
pip install -r requirements-lite.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-lite.txt
```

Full model inference may require additional model-specific dependencies, GPU access, model checkpoints, API credentials, or dataset access.

---

## Environment Variables

Some scripts use Roboflow dataset access.

Create a local `.env` file:

```bash
cp .env.example .env
```

Then add your own API key:

```text
ROBOFLOW_API_KEY=your_api_key_here
```

The `.env` file is ignored by Git and should never be committed.