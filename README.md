# EGRR :  Evidence-Guided Risk Reasoning for Explainable Autonomous Driving 

> 자율주행 위험 판단을 위한 증거 기반 설명형 추론 프레임워크입니다.

```text
This repository is a workspace for Evidence-Guided Risk Reasoning in autonomous driving scenes.
The system predicts road-scene risk levels while providing object-level evidence and natural-language reasoning.
```

본 프로젝트는 도로 주행 장면 이미지와 객체 탐지 기반 증거를 함께 활용하여, 자율주행 상황의 위험도를 판단하고 그 근거를 설명하는 Vision-Language 기반 위험 추론 시스템입니다.

기존의 위험도 분류 방식은 최종 위험 등급만 예측하는 경우가 많아, 어떤 객체와 장면 요소가 판단에 영향을 주었는지 설명하기 어렵습니다.
본 프로젝트에서는 YOLO 기반 객체 탐지 결과와 거리 정보를 구조화된 evidence로 변환하고, 이를 VLM 입력에 함께 제공하여 모델이 이미지뿐 아니라 명시적인 객체 수준 근거를 참고하도록 설계했습니다.

---

## Paper

**EGRR: Evidence-Guided Risk Reasoning for Explainable Autonomous Driving**

* Korean Title: 자율주행 위험 판단을 위한 증거 기반 설명형 추론 프레임워크
* Target Task: Autonomous Driving Risk Reasoning
* Keywords: autonomous driving, risk reasoning, explainable AI, object-level evidence, chain-of-thought

---

## Datasets

The datasets used or referred to in this project are listed below.

| Name                      | Description                                |
| ------------------------- | ------------------------------------------ |
| AI Hub 생활도로 객체인식 자율주행 데이터 | 생활도로 주행 장면 이미지, 객체 라벨, LiDAR 기반 좌표 및 거리 정보 |
| Risk Label            | 이미지별 위험도 등급 및 위험 설명 라벨                     |
| Object Evidence    | YOLO 기반 객체 탐지 결과와 거리 정보를 정리한 구조화 증거        |

---

## Key Features

### Evidence Builder

The Evidence Builder converts object detection results into structured textual evidence.

It includes:

* Object category
* Bounding box position
* Object count
* Ego-vehicle-relative distance
* Distance-prioritized object information

This allows the VLM to reason with explicit object-level evidence instead of relying only on image-level visual understanding.

---

### Object-aware Risk Reasoning

The system receives both:

1. Road scene image
2. Structured object evidence JSON

Then it generates:

1. Decision-critical objects
2. Factual scene description
3. Risk grade
4. Risk explanation

---

### CoT-style Structured Output

The model is trained to produce a fixed reasoning format.

```text
[Decision-Critical Objects]
- Objects that directly affect driving safety and why they matter.

[Factual Scene Description]
- Fact-grounded description of the observed road scene.

[Risk Grade]
L / M / H

[Risk]
A one-line summary of the key reason for the predicted risk level.
```

This structure helps the model connect observed evidence to the final risk decision.

---

### Ablation Experiment Support

The project supports ablation experiments by selectively removing evidence components.

| Setting      | Description                                                 |
| ------------ | ----------------------------------------------------------- |
| EGRR         | Full framework with object evidence and CoT-style reasoning |
| w/o Evidence | Removes structured object evidence                          |
| w/o CoT      | Removes intermediate reasoning steps                        |
| w/o BBox     | Removes bounding box information                            |
| w/o Distance | Removes ego-relative distance information                   |
| w/o Count    | Removes object count information                            |

---

## System Pipeline

```text
Road Scene Image
        │
        ▼
YOLO Object Detection
        │
        ▼
Object Evidence JSON
        │
        ▼
Evidence Builder
        │
        ▼
Image + Structured Evidence
        │
        ▼
Vision-Language Model Fine-tuning
        │
        ▼
CoT-style Risk Reasoning
        │
        ├── Decision-Critical Objects
        ├── Factual Scene Description
        ├── Risk Grade
        └── Risk Explanation
```

---

## Model Experiments & Results

### 1. Object Detector Comparison

YOLOv8s and YOLOv11s were compared as object detection modules for the Evidence Builder.

| Detector | Precision | Recall | mAP50 | mAP50-95 |
| -------- | --------- | ------ | ----- | -------- |
| YOLOv8s  | 0.77      | 0.68   | 0.73  | 0.52     |
| YOLOv11s | 0.77      | 0.69   | 0.74  | 0.55     |

YOLOv11s showed slightly better recall and mAP, and was used as the main detector for the risk reasoning experiments.

---

### 2. VLM Backbone Comparison

The proposed framework was evaluated with multiple VLM backbones.

| Model                        | BLEU   | METEOR | ROUGE-L | CIDEr | SPICE | BERTScore-F1 | Acc   | Macro-F1 | MAE   |
| ---------------------------- | ------ | ------ | ------- | ----- | ----- | ------------ | ----- | -------- | ----- |
| LLaVA1.6-Mistral-7B          | 75.281 | 0.769  | 0.837   | 6.574 | 0.776 | 0.943        | 0.786 | 0.720    | 0.285 |
| Llama3.2-11B-Vision-Instruct | 77.546 | 0.819  | 0.900   | 7.429 | 0.844 | 0.969        | 0.801 | 0.735    | 0.254 |
| Qwen2-VL-7B-Instruct         | 78.590 | 0.827  | 0.899   | 7.514 | 0.847 | 0.965        | 0.855 | 0.818    | 0.230 |

Qwen2-VL-7B-Instruct achieved the best overall balance between risk description generation and risk grade prediction.

---

### 3. Component Ablation Results

The contribution of Evidence Builder and CoT-style reasoning was analyzed through ablation experiments.

| Model Setting | BLEU   | METEOR | ROUGE-L | CIDEr | SPICE | BERTScore-F1 | Acc   | Macro-F1 | MAE   |
| ------------- | ------ | ------ | ------- | ----- | ----- | ------------ | ----- | -------- | ----- |
| w/o Evidence  | 63.312 | 0.669  | 0.808   | 5.758 | 0.699 | 0.935        | 0.669 | 0.491    | 0.492 |
| w/o CoT       | 78.601 | 0.834  | 0.903   | 7.552 | 0.853 | 0.967        | 0.752 | 0.668    | 0.247 |
| EGRR          | 78.590 | 0.827  | 0.899   | 7.514 | 0.847 | 0.965        | 0.855 | 0.818    | 0.230 |

The results show that structured object evidence is crucial for both risk explanation and risk prediction.
CoT-style reasoning contributes especially to the stability and consistency of final risk grade prediction.

---

### 4. Evidence Component Analysis

The contribution of each structured evidence component was further analyzed.

| Setting      | Acc   | Macro-F1 | MAE   |
| ------------ | ----- | -------- | ----- |
| w/o BBox     | 0.794 | 0.709    | 0.243 |
| w/o Distance | 0.789 | 0.726    | 0.263 |
| w/o Count    | 0.802 | 0.754    | 0.270 |
| EGRR         | 0.855 | 0.818    | 0.230 |

* Removing bounding box information caused a large drop in Macro-F1.
* Removing distance information caused a large drop in Accuracy.
* Removing object count increased MAE, indicating that object density helps adjust risk severity.

---

## Output Example

```text
[Decision-Critical Objects]
- 전방의 차량은 자차 주행 경로와 가까워 감속 또는 거리 유지가 필요할 수 있다.
- 측면의 보행자는 도로 가장자리와 인접해 있어 갑작스러운 진입 가능성에 주의가 필요하다.

[Factual Scene Description]
- 도로 전방에 차량이 존재한다.
- 주변에 보행자 또는 도로 이용자가 관측된다.
- 일부 객체는 자차 진행 방향과 가까운 위치에 있다.

[Risk Grade]
M

[Risk]
전방 차량과 주변 보행자 가능성으로 인해 주행 중 감속 및 주의가 필요한 상황이다.
```

---

## My Contributions

* **Research Framework Design**

  * 자율주행 위험 판단을 위한 Evidence-Guided Risk Reasoning 프레임워크 설계

* **Evidence Builder Design**

  * YOLO 기반 객체 탐지 결과와 LiDAR 기반 거리 정보를 구조화된 텍스트 evidence로 변환

* **Risk Labeling & Dataset Construction**

  * AI Hub 생활도로 데이터를 기반으로 위험도 등급 및 위험 설명 학습 데이터 구성

* **VLM Fine-tuning**

  * Qwen2-VL-7B-Instruct 모델을 4bit QLoRA 방식으로 fine-tuning

* **CoT-style Reasoning Format Design**

  * Decision-Critical Objects, Factual Scene Description, Risk Grade, Risk Explanation으로 구성된 출력 형식 설계

* **Ablation Experiments**

  * Evidence, CoT, bounding box, distance, object count 요소별 제거 실험 수행

* **Evaluation & Analysis**

  * Accuracy, Macro-F1, MAE와 BLEU, ROUGE-L, METEOR, CIDEr, SPICE, BERTScore 기반 성능 분석

---

## Limitations & Future Work

* 현재는 단일 이미지 프레임 기반 위험 판단이므로 연속 프레임 기반 temporal reasoning 확장이 필요함
* 객체 탐지 결과의 오류가 Evidence Builder를 거쳐 위험 판단에 영향을 줄 수 있음
* 비, 야간, 역광, 복잡한 교차로 등 다양한 실제 주행 조건에 대한 추가 평가가 필요함
* 향후 영상 입력, LiDAR point cloud, HD Map 등 멀티모달 센서 정보와의 통합이 필요함
* 위험도 판단을 넘어 감속, 정지, 차선 변경 회피 등 주행 행동 제안으로 확장 가능함

---
