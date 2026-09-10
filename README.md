# Leveraging LLMs for Automated Grammar Error Correction of Assamese

This project explores **Automated Grammar Error Correction (GEC) for Assamese**, a low-resource Indian language with limited publicly available large-scale GEC resources.

The project introduces a **linguistically grounded synthetic Assamese GEC dataset** and evaluates the ability of modern open-source language models to correct grammatical errors through instruction tuning.

## Overview

Unlike conventional synthetic GEC approaches that generate errors through random character or word-level perturbations, this work uses **Assamese grammar rules and naturally observed learner-error patterns** to generate realistic grammatical errors.

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
