# AI Recruiter System — Reasoning-First Training

## Overview

This project implements an AI Recruiter system designed to **reason about candidates**, not memorize data.
The model evaluates candidates using role context, job requirements, resumes, and interview transcripts, and produces an explainable hiring decision.

The focus is on **real-world performance**:

* Works on unseen candidates
* Stable during live inference
* Trained to understand relationships, not patterns alone

---

## How the System Works

The pipeline is built to avoid overfitting and data leakage.

Raw hiring data is:

1. Cleaned and de-duplicated
2. Structured into clear **inputs** and **outputs**
3. Split into train / validation / test sets
4. Used to train a reasoning-focused model with LoRA
5. Deployed with low-latency, quantized inference

At no point is the model trained to simply repeat answers from the dataset.

---

## Data Design (Key Idea)

Each sample is intentionally separated into three parts:

**Input**

* Role
* Job description
* Resume
* Interview transcript

**Reasoning**

* Evidence of skills
* Behavioral signals
* Decision justification

**Output**

```json
{
  "reasoning": "...",
  "status": "select or reject"
}
```

This separation forces the model to **learn relationships** between signals rather than memorizing outcomes.

---

## File Structure and Responsibilities

### `prepare_graph.py`

Handles dataset preparation.

* Cleans raw CSV data
* Removes duplicates and spam samples
* Masks sensitive information
* Converts data into instruction / input / output format
* Creates 70% train, 15% validation, 15% test splits

This step is critical for generalization.

---

### `train_graph.py`

Responsible for training.

* Loads clean datasets
* Applies LoRA fine-tuning
* Trains only on assistant responses
* Uses cosine learning rate scheduling
* Evaluates on a strict held-out test set

Training is designed to capture **relationships across fields**, not isolated text.

---

### `model.py`

Inference engine.

* Loads the trained model in 4-bit mode
* Accepts real candidate data
* Produces structured JSON output
* Uses low temperature for consistent reasoning

This is optimized for real-time usage.

---

### `run_live.py`

Live evaluation runner.

* Takes a candidate JSON file
* Runs the full reasoning pipeline
* Outputs the final hiring decision

Used for testing, demos, and validation.

---

### `docker-compose.yml`

Environment control.

* Ensures reproducibility
* Manages CUDA, volumes, and dependencies
* Keeps training and inference consistent across machines

---

## Why This Generalizes Well

* Strict train / validation / test separation
* No label or reasoning leakage
* Reasoning trained before final decisions
* Real interview transcripts
* Deterministic inference

As a result, the model performs reliably on **new, unseen candidates**, not just the training data.

---

## Running the System

Prepare data:

```bash
python prepare_graph.py
```

Train the model:

```bash
python train_graph.py
```

Run live inference:

```bash
python run_live.py
```

---

## Final Notes

This is not a prompt-engineering demo.

It is a structured learning system built to:

* Make explainable decisions
* Generalize in production
* Behave like a real recruiter, not a classifier
