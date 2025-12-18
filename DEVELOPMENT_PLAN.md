# Q-Sight Africa – Development Plan & Segmentation Models

This document describes **how the project will be developed**, **how segmentation/models are structured**, and **how sample scripts and PDFs are organized on GitHub**. It is written to satisfy hackathon / academic reviewers who want *clear plans*, even if some artifacts are provided as PDFs.

---

## 1. Purpose of This Document

This development plan serves three goals:

1. Explain the **end-to-end development roadmap** of Q-Sight Africa
2. Describe the **segmentation and modeling pipeline** (classical + ACO + quantum)
3. Clarify how **PDF-only artifacts (plans, scripts, diagrams)** are hosted and referenced on GitHub

This ensures transparency, reproducibility, and readiness for future implementation.

---

## 2. Repository Documentation Strategy (Important)

Because some components (development plans, diagrams, early scripts) are currently available **only as PDFs**, we follow a **documentation-first GitHub strategy**.

### 2.1 How PDFs Are Used on GitHub

PDFs are used for:

* High-level system design
* Development roadmaps
* Algorithm flow diagrams
* Early-stage pseudocode and sample scripts

They are stored in a dedicated folder:

```
/docs/pdfs/
├── Development_Plan.pdf
├── Segmentation_Model_Design.pdf
├── System_Architecture.pdf
└── Sample_Pipelines.pdf
```

Each PDF is **referenced and explained** inside Markdown files (like this one), so reviewers never need to guess what the PDFs contain.

---

## 3. Overall Development Phases

### Phase 1: Research & Design (Hackathon Phase)

**Goal:** Validate feasibility of ACO + Quantum hybrid approach

**Activities:**

* Literature review on DR detection and quantum ML
* Selection of classical backbone (ResNet18)
* Design of ACO feature-selection strategy
* Design of quantum variational classifier

**Outputs:**

* System architecture diagrams (PDF)
* Feature-selection design (PDF)
* Initial development plan (this document)

---

### Phase 2: Classical Pipeline Development

**Goal:** Build a strong classical baseline that works on low-resource devices

**Steps:**

1. Image preprocessing

   * CLAHE contrast enhancement
   * Image resizing (224×224)
   * Normalization

2. Feature extraction

   * Lightweight CNN (ResNet18)
   * Output: 512-dimensional feature vector

3. Baseline evaluation

   * Accuracy, sensitivity, specificity

**Artifacts:**

* Python preprocessing scripts (future `/core/`)
* Model description (PDF)

---

### Phase 3: Feature Segmentation & Selection (ACO)

This is the **segmentation stage** of the project.

### 3.1 What We Mean by Segmentation

Segmentation here refers to **feature-space segmentation**, not pixel-wise masks.

* Retinal image → CNN features (512)
* ACO selects the **most informative feature subset (32)**

This segmentation:

* Reduces dimensionality
* Improves interpretability
* Makes quantum processing feasible

### 3.2 ACO-Based Feature Segmentation

**Process:**

1. Each ant represents a candidate feature subset
2. Pheromone levels encode feature importance
3. Fitness function evaluates classification performance
4. Iterative optimization converges to optimal subset

**Outputs:**

* Selected feature indices
* Feature importance scores (explainability)

**Documentation:**

* Algorithm explanation (PDF)
* Sample pseudocode (PDF)

---

## 4. Quantum Segmentation & Classification Model

### 4.1 Quantum Model Overview

* Model: Variational Quantum Classifier (VQC)
* Qubits: 32 (one per selected feature)
* Encoding: Angle encoding
* Output: DR severity class (multi-class)

### 4.2 Why Hybrid (Not Fully Quantum)

| Approach       | Feasible | Cost      | Accuracy    |
| -------------- | -------- | --------- | ----------- |
| Classical only | ✅        | Low       | Medium      |
| Fully quantum  | ❌        | Very high | Theoretical |
| Hybrid (ours)  | ✅        | Very low  | High        |

### 4.3 Fallback Mode

If quantum hardware is unavailable:

* System automatically switches to classical classifier
* No interruption in screening workflow

---

## 5. Sample Scripts Policy (Even If PDFs)

At this stage, **sample scripts may be provided as PDFs** for clarity and review.

### Why This Is Acceptable

* Hackathon timeline constraints
* Focus on algorithm design, not full production code
* Ensures reviewers understand logic before implementation

### How We Present Them

Each PDF script includes:

* Clear inputs/outputs
* Step-by-step logic
* Mapping to future Python files

Example reference inside README:

```md
See `/docs/pdfs/Sample_Pipelines.pdf` for pseudocode of the ACO–Quantum training loop.
```

---

## 6. Planned Code Migration (Post-Hackathon)

PDF scripts will be converted into executable code following this mapping:

| PDF Section     | Future File                    |
| --------------- | ------------------------------ |
| ACO pseudocode  | `core/aco_feature_selector.py` |
| Quantum circuit | `core/quantum_classifier.py`   |
| Hybrid training | `core/hybrid_trainer.py`       |
| Preprocessing   | `data/preprocessing.py`        |

This guarantees **continuity from design → implementation**.

---

## 7. Validation & Testing Plan

### Metrics

* Accuracy
* Sensitivity (recall for DR cases)
* Specificity
* Inference time

### Validation Strategy

* Compare classical vs hybrid
* Compare ACO-selected vs full feature set
* Ophthalmologist benchmark (future phase)

---

## 8. Ethical & Practical Constraints

* No raw patient data in repository
* All examples use public datasets
* African data sovereignty respected
* Explainability prioritized

---

## 9. File Naming & GitHub Placement (Summary)

### Recommended File Name (This File)

```
DEVELOPMENT_PLAN.md
```

### Location

```
/docs/DEVELOPMENT_PLAN.md
```

### Related Files

```
/docs/pdfs/
├── Development_Plan.pdf
├── Segmentation_Model_Design.pdf
├── Sample_Pipelines.pdf
```

---

## 10. Final Note

This document ensures that **even when parts of the project are currently represented as PDFs**, the development process is:

* Transparent
* Scientifically grounded
* Easy to convert into full implementation

It demonstrates that Q-Sight Africa is not just an idea, but a **structured, executable development plan**.

---

**Team Q-Sight Africa**
