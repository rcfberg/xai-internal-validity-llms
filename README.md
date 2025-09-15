# Explainable AI: Internal validity of Large Language Models

This repository provides supplementary transparency material for the ICWSM 2025 paper:

“Using Explainable AI to Enhance the Internal Validity of Large Language Models in Ambiguous Political Text Classification.”

The contents include redacted datasets, code, and plots that illustrate the analytical procedures used in the study. This is not a full replication package but is shared to enhance transparency and understanding of the research process.

# Key resources

1. XAI_InternalValidityLLM.py – Python code for training BERT/DistilBERT classifiers and running evaluation.
2. XAI_SHAP_LIME_analysis.pdf – LIME and SHAP plots demonstrating interpretability analyses.
3. XAI_redacted_system_instructions_naive.rtf – Redacted system instructions for the naive prompting setup.
4. XAI_redacted_system_instructions_sophisticated.rtf – Redacted system instructions for the theory-informed prompting setup.
5. xai_reasoning_redacted_data_12092025.csv – Reduced and anonymized dataset used for reasoning experiments.
6. xai_redacted_full_data_12092025.csv – Redacted and anonymized experimental dataset with classifications.

# Computation

1. Machine learning models were trained using both Google Colab A100 GPUs as well as the CSC's Puhti supercomputer cluster.

# Notes on usage

1. Files are provided for illustrative and transparency purposes.
2. Full reproduction is not possible, as original Twitter data cannot be shared due to privacy and platform policy restrictions.
3. Code and plots are shared to demonstrate methodology and support interpretability of results.
