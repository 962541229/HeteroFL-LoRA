# HeteroFL-LoRA

## Overview

This project implements a heterogeneous federated LoRA fine-tuning framework for discriminative language understanding models. The goal is to enable multiple clients with different pretrained language models to collaboratively learn task-specific knowledge without sharing raw data or directly averaging incompatible LoRA parameters.

Unlike conventional federated LoRA methods that assume all clients use the same backbone model, this project considers a more realistic heterogeneous setting where different clients may hold different language foundation models, such as DistilBERT, ERNIE, and RoBERTa. Since LoRA updates are tied to the parameter space of their local base model, directly aggregating LoRA parameters across heterogeneous models can cause severe subspace mismatch. This project addresses this issue through singular-value-focused LoRA updates and public-subspace-guided aggregation.

## Main Idea

The core idea is to represent local LoRA updates through singular value shifts. For each target linear layer, the frozen pretrained weight matrix is decomposed by truncated singular value decomposition. The left and right singular matrices are kept fixed, while only a dense singular value shift matrix is trained. This produces a lightweight and transferable update form:

```text
W = W0 + U ΔΣ V^T
```

where `W0` is the frozen pretrained weight, `U` and `V` define the local singular subspace, and `ΔΣ` is the trainable SVF-LoRA parameter.

To support collaboration across heterogeneous models, the server maintains public subspaces initialized from a public base model, such as BERT-base. Client-side singular value updates are projected into the matched public subspace, aggregated on the server, and then projected back to each client’s local subspace. This allows task-specific knowledge to be shared across different model architectures while keeping each client’s backbone private and unchanged.

## Project Components

The project contains implementations for three types of training settings.

### Full-Parameter Fine-Tuning

Each client model can be independently fine-tuned with all backbone parameters trainable. This setting serves as a conventional centralized baseline for comparison.

### Standard LoRA Fine-Tuning

Each client model can also be fine-tuned with standard LoRA modules. The pretrained backbone is frozen, and only low-rank LoRA matrices together with the classification head are updated.

### SVF-LoRA Fine-Tuning

The project implements Singular Value–Focused LoRA, where each target linear layer is replaced by an SVF-LoRA module. Instead of learning two low-rank matrices, SVF-LoRA freezes the singular bases of the pretrained weight and only learns a dense singular value shift matrix.

### Heterogeneous Federated SVF-LoRA

The main federated framework enables clients with different base models to collaborate through public-subspace projection. Each client trains locally on private data, uploads projected singular value updates, receives aggregated global updates from the server, and maps them back into its own local parameter subspace.

## Supported Models

The current project supports the following discriminative language models:

- DistilBERT
- ERNIE
- RoBERTa
- BERT-base as the public reference model

The implementation supports both same-dimensional and cross-dimensional heterogeneous settings, such as collaboration between 768-dimensional models and 1024-dimensional RoBERTa-large.

## Target Modules

For DistilBERT, SVF-LoRA and LoRA modules are inserted into selected attention and feed-forward layers, including query, value, and feed-forward projection layers.

For BERT-like models such as ERNIE and RoBERTa, the target modules include query projection, value projection, intermediate feed-forward projection, and output feed-forward projection. These modules are mapped into a unified semantic space so that heterogeneous client updates can be matched with corresponding public subspaces.

## Public Subspace Matching

The server maintains multiple public subspaces extracted from the public BERT-base model. Each client-side SVF-LoRA module is first mapped to a canonical module type, such as query, value, intermediate, or output. Then, among public subspaces of the same type, the most similar public subspace is selected according to singular subspace similarity.

This matching strategy avoids naive layer-index matching and allows a client layer to be aligned with the public layer whose singular subspace is most compatible.

## Federated Aggregation

The server aggregates projected singular value updates rather than raw model parameters or standard LoRA matrices. For each matched public subspace, client updates are projected into the public coordinate system and then averaged. The aggregated update is used both as a global task-specific update and as a signal to refine the public subspace.

After aggregation, the global singular value update is projected back to each client’s local subspace and used to initialize the next round of local training.

## Data Setting

The project is designed for natural language understanding tasks, including GLUE-style classification datasets such as SST-2, MNLI, QNLI, QQP, and RTE. It supports both IID and Dirichlet-based non-IID data partitions, allowing evaluation under different federated data heterogeneity settings.

## Purpose

This project provides a practical implementation of heterogeneous federated LoRA fine-tuning for discriminative language models. It is intended to study how parameter-efficient fine-tuning can be extended from homogeneous federated learning to more realistic scenarios where clients own different pretrained models, different architectures, and different parameter spaces.
