# Leveraging LLMs for Automated Grammar Error Correction of Assamese

This project explores **Automated Grammar Error Correction (GEC) for Assamese**, a low-resource Indian language with limited publicly available large-scale GEC resources.

The project introduces a **linguistically grounded synthetic Assamese GEC dataset** and evaluates the ability of modern open-source language models to correct grammatical errors through instruction tuning.

## Overview

Unlike conventional synthetic GEC approaches that generate errors through random character or word-level perturbations, this work uses **Assamese grammar rules and naturally observed learner-error patterns** to generate realistic grammatical errors.

The complete pipeline is:

**Assamese Wikipedia Corpus → Learner Error Analysis → Error Taxonomy → Rule-Based Error Injection → Instruction Dataset → LLM Fine-Tuning → GEC Evaluation**

## Dataset

The dataset is constructed from a cleaned Assamese Wikipedia corpus containing **100,000 sentences**.

A reference corpus of Assamese learner writing was also analyzed:

* 661 handwritten Assamese sentences
* 343 erroneous instances
* 11 grammatical error categories
* 51,000 sentences selected for error generation
* 49,000 clean sentences retained
* 90,000 training examples
* 10,000 test examples

### Error Categories

The dataset covers:

1. Spelling Error
2. Verb Error
3. Tense Error
4. Person Agreement Error
5. Number Agreement Error
6. Pronoun Error
7. Karak / Case Marker Error
8. Bibhakti / Postposition Error
9. Pratyaya / Word Formation Error
10. Syntax / Word Order Error
11. Punctuation Error

Spelling errors were the most frequently observed category in the reference corpus, followed by syntax/word-order errors.

## Methodology

### 1. Reference Error Analysis

Authentic Assamese learner writing was manually analyzed to identify common grammatical error patterns and estimate their empirical distribution.

### 2. Rule-Based Error Injection

Grammatical errors are generated using manually designed linguistic transformation rules derived from standard Assamese grammar references and validated lexical resources.

The transformations can introduce:

* spelling modifications
* verb and tense errors
* agreement errors
* pronoun errors
* case-marker and postposition errors
* word-formation errors
* word-order changes
* punctuation errors

Multiple transformations can be applied sequentially to the same sentence, allowing the dataset to represent sentences containing multiple grammatical errors.

### 3. Instruction Dataset

Each training example contains:

* Incorrect Assamese sentence
* Error type(s)
* Correct Assamese sentence

The error labels provide additional supervision and guide the model toward the appropriate grammatical correction.

### 4. Model Fine-Tuning

The following models were evaluated:

* **Gemma-3 1B**
* **Qwen2.5-1.5B**
* **ByT5-small**

LoRA was used for parameter-efficient fine-tuning of the decoder-only models.

### Training Configuration

| Parameter               | Value                   |
| ----------------------- | ----------------------- |
| Learning Rate           | 2 × 10⁻⁴                |
| Optimizer               | AdamW                   |
| Scheduler               | Cosine                  |
| Maximum Sequence Length | 512                     |
| LoRA Rank               | 16                      |
| LoRA Alpha              | 32                      |
| LoRA Dropout            | 0.05                    |
| Framework               | Hugging Face TRL + PEFT |
| Deep Learning Framework | PyTorch                 |
| Hardware                | NVIDIA A100 GPU         |

## Evaluation

The models are evaluated using:

* **Exact Match Accuracy**
* **BLEU**
* **GLEU**
* **F0.5**

Both **zero-shot** and **instruction-tuned** models are evaluated on the same held-out test set.

## Results

Instruction tuning substantially improves the performance of all evaluated models.

| Model        | Exact Match |   BLEU |   GLEU |       F0.5 |
| ------------ | ----------: | -----: | -----: | ---------: |
| Gemma-3 1B   |      70.38% | 87.45% | 88.26% |     91.75% |
| Qwen2.5-1.5B |      57.13% | 84.02% | 85.29% | **92.28%** |
| ByT5-small   |      58.03% | 82.83% | 84.26% |     91.77% |

Gemma-3 1B achieves the best Exact Match, BLEU, and GLEU, while Qwen2.5-1.5B achieves the highest F0.5 score.


## Project Structure

```text
Assamese_GEC/
│
├── Models/
│   ├── Byt5-small/
│   │   ├── evaluate_byt5_sft.py
│   │   ├── evaluate_byt5_zeroshot.py
│   │   └── train_byt5_sft.py
│   │
│   ├── Gemma3-1b-it/
│   │   ├── evaluate_gem_sft.py
│   │   ├── evaluate_gemma_zeroshot.py
│   │   └── train_gemma_sft.py
│   │
│   └── Qwen2.5-1.5B-Instruct/
│       ├── evaluate_qwen_sft.py
│       ├── evaluate_qwen_zeroshot_v2.py
│       ├── evaluate_qwen_zeroshot.py
│       └── train_qwen_sft.py
│
├── Noise Injection Notebooks/
│   ├── Noise_injection_1.ipynb
│   └── Noise_injection_2.ipynb
│
├── .gitattributes
└── README.md
```
