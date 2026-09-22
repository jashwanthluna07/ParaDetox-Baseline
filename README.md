# ParaDetox-Baseline

A small multilingual text detoxification project based on the **ParaDetox** task.

The goal of this project is to explore whether a multilingual sequence-to-sequence model can transform toxic text into a less toxic / neutral version while preserving the original meaning.

> This repository contains a baseline implementation and experiments. It is not intended to reproduce the full-scale ParaDetox research system.

---

## What is Text Detoxification?

Text detoxification is a text style-transfer task where a toxic sentence is rewritten into a non-toxic or neutral version while keeping its meaning as similar as possible.

For example:

```text
Input:
You made a mistake you ass.

Output:
You made a mistake.
```

The model learns this transformation from parallel pairs:

```text
toxic sentence → neutral sentence
```

---

## Project Goal

This project explores multilingual detoxification using:

- English
- Hindi

The initial goal is to build a simple working baseline and understand the complete ML pipeline:

```text
Dataset
   ↓
Data inspection
   ↓
Data cleaning
   ↓
Leakage-safe train/validation/test split
   ↓
mT5 tokenization
   ↓
Fine-tuning
   ↓
Held-out evaluation
   ↓
English vs Hindi analysis
```

---

## Model

### mT5-small

We use Google's `mT5-small` as the base sequence-to-sequence model.

Model:

```text
google/mt5-small
```

The model is fine-tuned with a language-aware task prefix:

```text
detoxify en: <toxic sentence>
detoxify hi: <toxic sentence>
```

This allows the same model to handle both English and Hindi.

The larger mT5 models used in some research settings were intentionally not used here because this project is designed as a lightweight prototype that can be trained on a consumer GPU.

---

## Dataset

The project uses parallel toxic → neutral sentence pairs.

The initial datasets contain:

```text
English: 400 pairs
Hindi:   400 pairs
```

A larger English dataset was also processed, containing multiple neutral rewrites for individual toxic sentences.

After expanding the neutral variants and combining the datasets:

```text
Total pairs: ~20,030

English: 19,630
Hindi:      400
```

### Important preprocessing decision

Some toxic sentences have multiple neutral rewrites.

Therefore, the dataset was **split by unique toxic source sentence before expanding the pairs**.

This prevents the same toxic sentence from appearing in both training and test sets with different neutral rewrites.

This is important for avoiding data leakage.

---

## Dataset Split

The final dataset was divided into:

| Split | Examples | English | Hindi |
|---|---:|---:|---:|
| Train | 15,968 | 15,648 | 320 |
| Validation | 2,038 | 1,998 | 40 |
| Test | 2,024 | 1,984 | 40 |
| **Total** | **20,030** | **19,630** | **400** |

The language distribution is intentionally preserved in this first baseline experiment.

---

## Training Configuration

The baseline was trained using:

| Parameter | Value |
|---|---|
| Base model | `google/mt5-small` |
| Epochs | 3 |
| Learning rate | `5e-5` |
| Batch size | 1 |
| Gradient accumulation | 8 |
| Optimizer | Adafactor |
| Gradient checkpointing | Enabled |
| Maximum sequence length | 128 |
| Precision | FP32 |
| GPU | NVIDIA RTX 4050 Laptop GPU |

Training:

```text
Training examples: 15,968
Optimizer steps: 5,988
Training time: ~2h 22m
```

---

## Baseline Results

After three epochs:

```text
Validation loss: 1.311
Training loss:   21.28
```

The model was then evaluated on the held-out test set.

### English

The model demonstrated noticeable detoxification behavior.

Example:

```text
Input:
only to replaced by another braindead consensus.

Expected:
Only to replaced by another consensus.

Model:
Only to replaced by another consensus.
```

Another example:

```text
Input:
btw jonhson county is a real shithole in case you were wondering.

Expected:
Johnson county is not good.

Model:
Btw jonhson county is a real thing in case you were wondering.
```

The model is clearly capable of removing or modifying some toxic expressions, although the generated text is not always grammatically or semantically ideal.

### Hindi

The baseline performed substantially worse on Hindi.

For example:

```text
Input:
दोनों एक नंबर के हरामि हो

Expected:
दोनों एक नंबर के बेकार आदमी हो

Model:
दोनों एक नंबर के हरामि हो
```

In several Hindi examples, the model simply reproduced the toxic input.

There were also cases where the model switched partially into English:

```text
Input:
चला है लोडा एक्सपर्ट बनने...

Model:
I'm not good.
```

This suggests that the current model has difficulty maintaining Hindi generation and detoxification under the highly imbalanced training distribution.

---

## Current Observation

The baseline dataset contains approximately:

```text
98% English
 2% Hindi
```

This creates an interesting experimental question:

> How does multilingual data imbalance affect detoxification performance?

The first experiment suggests that the model learns the dominant English task much more effectively than the low-resource Hindi task.

This will be investigated in future experiments rather than assuming that the model architecture itself is the only cause.

---

## Project Structure

```text
ParaDetox-Baseline/
│
├── src/
│   ├── check_overlap.py
│   ├── inspect_large_dataset.py
│   ├── prepare_data.py
│   ├── prepare_large_data.py
│   ├── split_data.py
│   ├── test_model.py
│   ├── test_tokenizer.py
│   ├── test_trained_model.py
│   └── train.py
│
├── inspect_dataset.py
│
├── data/                  # Not tracked by Git
├── models/                # Not tracked by Git
├── outputs/               # Not tracked by Git
│
└── README.md
```

Large datasets and trained model checkpoints are intentionally excluded from Git using `.gitignore`.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/jashwanthluna07/ParaDetox-Baseline.git
cd ParaDetox-Baseline
```

### 2. Install dependencies

The main libraries used are:

```bash
pip install torch
pip install transformers
pip install datasets
pip install pandas
pip install scikit-learn
pip install sentencepiece
```

A CUDA-enabled PyTorch installation is recommended for GPU training.

---

## Running the Pipeline

### Inspect the dataset

```bash
python inspect_dataset.py
```

### Prepare the original multilingual dataset

```bash
python src/prepare_data.py
```

### Prepare the larger dataset

```bash
python src/prepare_large_data.py
```

### Create train/validation/test splits

```bash
python src/split_data.py
```

### Test the tokenizer

```bash
python src/test_tokenizer.py
```

### Train the model

```bash
python src/train.py
```

### Evaluate the trained model

The trained checkpoints are intentionally not included in this repository.

After training, the evaluation script can be used to generate detoxified outputs from the held-out test set:

```bash
python src/test_model.py
```

---

## Current Limitations

This is an early baseline and has several limitations:

- Hindi has very few training examples compared with English.
- The model sometimes copies toxic input instead of detoxifying it.
- Hindi generation can switch into English.
- Generated sentences can contain grammatical errors.
- Semantic preservation has not yet been comprehensively measured.
- Automatic toxicity evaluation has not yet been implemented.
- The current experiment uses a relatively small model compared with larger research models.
- The baseline has only been trained for three epochs.

Therefore, the validation loss alone should not be interpreted as proof of successful detoxification.

---

## Next Experiments

The main planned experiment is to investigate the effect of language imbalance.

### Experiment 1 — Baseline

```text
English: 15,648
Hindi:      320
```

### Experiment 2 — Hindi-balanced training

Increase the representation of Hindi during training while keeping the model architecture and major training parameters unchanged.

Then compare:

```text
                    Baseline       Hindi-balanced
                    --------       --------------
English performance
Hindi performance
Toxicity reduction
Semantic similarity
Language preservation
```

This will help determine whether the poor Hindi performance is related to the severe imbalance in the training data.

Future experiments may also include:

- SARI / BLEU / ROUGE-style evaluation
- Toxicity classification before and after detoxification
- Semantic similarity
- Fluency evaluation
- Hinglish support
- Larger multilingual datasets
- Comparison with other sequence-to-sequence models

---

## References

### ParaDetox

The original ParaDetox work introduced a parallel corpus for text detoxification and explored controlled rewriting of toxic text.

- GitHub: https://github.com/s-nlp/paradetox
- Paper: https://aclanthology.org/2022.acl-long.469.pdf

### Multilingual Text Detoxification

- NAACL 2024 multilingual text detoxification work:
  https://aclanthology.org/2024.naacl-short.12/

### PAN Text Detoxification

- PAN 2025 Text Detoxification task:
  https://pan.webis.de/clef25/pan25-web/text-detoxification.html

### mT5

- Model: `google/mt5-small`

---

## Status

**Current status: Baseline complete ✅**

```text
Dataset preparation       ✅
Leakage-safe splitting    ✅
mT5-small fine-tuning     ✅
GPU training              ✅
Held-out evaluation       ✅
English baseline          ✅
Hindi baseline            ✅
Hindi imbalance analysis  🔄
```

---

## Author

**Jashwanth Lunavath**

IIIT Allahabad

GitHub: https://github.com/jashwanthluna07
```
