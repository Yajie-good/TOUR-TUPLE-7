# TOUR-TUPLE-7
TOUR-TUPLE-7: A FINE-GRAINED 7-TUPLE GENERATIVE ASPECT-BASED SENTIMENT ANALYSIS BENCHMARK FOR TOURISM SERVICE QUALITY

<p align="center">
  <img src="Framework.png" alt="Overview" width="80%">
</p>

This repository provides resources for **TOUR-TUPLE-7**, a manually annotated benchmark of 49,998 TripAdvisor reviews, each labeled with seven-field tuples  
(aspect category, aspect term, opinion term, sentiment polarity, aspect score, overall score, reason).  
The dataset enables unified evaluation of aspect-based sentiment analysis (ABSA) tasks including extraction, classification, regression, and explanation.

## Repository Structure
- `data/` – A small sample version of the dataset (train/dev/test splits) for demonstration.  
- `Seqtoseqbaseline/` – Baseline sequence-to-sequence training and evaluation scripts (e.g., T5, BART, T5 Large, BART Large).  
- `DPIS-SCD-LLM/` – LLM and Our proposed schema-aware generative framework with Dual-Phase Instruction Schema (DPIS) and Schema-Constrained Decoding (SCD).
<p align="center">
  <img src="aspect_category.png" alt="Aspect Category Distribution" width="60%">
</p>

## Quick Notes
This repository is released for **academic review**.  
Complete dataset, training code, and step-by-step instructions will be made publicly accessible after the paper is accepted and published.

## Usage
1. Environment: Python + PyTorch; we use the same setup as in the paper.
2. Use `Seqtoseqbaseline` for training standard seq2seq baselines.
3. Use `DPIS-SCD-LLM` for fine-tunin LLama and Qwen ICL, fine-tuning, as well as DPIS-SCD and evaluation.


