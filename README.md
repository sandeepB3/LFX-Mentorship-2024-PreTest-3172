# LFX Mentorship 2024 Program - WasmEdge PreTest 

#### This repository contains documentation for the [pretest](https://github.com/WasmEdge/WasmEdge/discussions/3182) and proposal submitted for the LFX Mentorship 2024 Program, focusing on the integration of burn.rs as a new backend for WASI-NN in the [WasmEdge](https://github.com/WasmEdge/WasmEdge/issues/3172) project.

- **Applied:** [To integrate burn.rs as a new WASI-NN backend](https://github.com/WasmEdge/WasmEdge/issues/3172)
- **Host OS:** macOS Sonoma Version 14.1

## 📚 Table Of Contents
### 1. Framework Execution: Burn.rs
   - [The Burn Book](#the-burn-book)
   - [Examples](#examples)
   - [Pre-trained Models](#pre-trained-models)

### 2. Rustls Plugin
   - [Building and Executing the Plugin](#building-and-executing-the-plugin)
   - [Summary & Execution Result of Example](#summary--execution-result-of-example)

### 3. Approach for Building the WASI-NN Burn Backend
   - [Analogy to Rustls Plugin](#analogy-to-rustls-plugin)
   - [Analogy to WASI-NN PyTorch Plugin](#analogy-to-wasi-nn-pytorch-plugin)
     
### 4. Miscellaneous
   - [Applicant Details](#applicant-details)
   - [References](#references)

## 🌐 Introduction
The following section of the readme file delves into the Burn framework, aiming to comprehend its intricacies and elucidate the process of developing a deep learning model using Burn. Subsequently, we will explore the development of the Rustls plugin, guide through its installation, and demonstrate an example execution with Wasmedge and the Rustls plugin. Lastly, we will examine an approach that I believe is well-suited for constructing the WASI-NN Burn backend. This will be established by creating an analogy to the building and execution process of the Rustls plugin and the WASI-NN PyTorch plugin.

## Framework Execution: Burn.rs
### 1. The Burn Book




