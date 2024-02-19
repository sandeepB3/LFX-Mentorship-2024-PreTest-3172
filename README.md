# LFX Mentorship 2024 Program - WasmEdge PreTest 

#### This repository contains documentation for the [pretest](https://github.com/WasmEdge/WasmEdge/discussions/3182) and proposal submitted for the LFX Mentorship 2024 Program, focusing on the integration of burn.rs as a new backend for WASI-NN in the [WasmEdge](https://github.com/WasmEdge/WasmEdge/issues/3172) project.

- **Applied:** [To integrate burn.rs as a new WASI-NN backend](https://github.com/WasmEdge/WasmEdge/issues/3172)
- **Host OS:** macOS Sonoma Version 14.1

## 📚 Table Of Contents
### 1. Framework Execution: Burn.rs
   - [The Burn Book](#1-the-burn-book)
   - [Model Examples](#2-model-examples)

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
Pedagogical materials serve as the crucial initial step in the learning process. Therefore, my first exploration led me to the ["Burn Book"](https://burn.dev/book/), a comprehensive resource that guides users through the creation of a basic burn application. This tutorial specifically demonstrates the process of achieving the addition of two tensors using the WGPU backend within the burn module.

**Prerequisite:** The host OS must have the **rust compiler** and **cargo package manager** installed.
<div align="center">
   <img width="674" alt="Screenshot 2024-02-19 at 7 09 58 PM" src="https://github.com/sandeepB3/LFX-Mentorship-2024-PreTest-3172/assets/107111616/e46ade8d-f7c5-44ae-a786-d8c9d8cc00aa">
</div>


To install rustup on Linux or macOS:
```bash
curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh
```

<h3 align="center">
   My Burn Application 
</h3>

**Following are the steps to create a burn application to achieve the addition of two tensors using the WGPU backend.**
```bash
cargo new my_burn_app
cd my_burn_app
cargo add burn --features wgpu
cargo build
```
Following is the code snippet present in `src/main.rs`:
<div align="center">
   <img width="995" alt="img2" src="https://github.com/sandeepB3/LFX-Mentorship-2024-PreTest-3172/assets/107111616/1e4b22ec-3cae-4510-a908-44f8aae6be80">
</div>

By running `cargo run`, following is the obtained result:
<div align="center">
   <img width="1148" alt="Screenshot 2024-02-19 at 7 40 09 PM" src="https://github.com/sandeepB3/LFX-Mentorship-2024-PreTest-3172/assets/107111616/ff3c1ed1-2488-469c-8344-33a371aa9d68">
</div>

<h3 align="center">
   Burn Workflow: From Training to Inference 
</h3>

Further along, the burn book walked me through the process of creating a custom model built with Burn. It guided me to train a simple convolutional neural network model on the MNIST dataset and prepare it for inference.

**The Key Learning Takeaways were:**
- Creating neural network models using the burn module
- Importing and preparing datasets using the MNIST dataset provided by Hugging Face
- Training models on data using the ClassificationOutput module & implementating the training and validation steps for the model
- Choosing the Wgpu backend which is compatible with any operating system
- Using a model for inference that is to load the model configuration of the training to make predictions

**Following is the directory structure of the model implemented by me using the burn book:**
```
first-burn-model
├── Cargo.toml
└── src
    ├── data.rs
    ├── inference.rs
    ├── lib.rs
    ├── main.rs
    ├── model.rs
    └── training.rs
```
> Problems Faced: While I was going through model implementation using the burn book, I noticed the existing documentation in production is outdated, resulting in inaccuracies and compiler errors during the study and implementation of codes. I have addressed the related [issue](https://github.com/tracel-ai/burn/issues/1325) in the burn repository, which shall be looked into.

**Troubleshooting:** Though they are small issues which could be identified easily, but studying burn framework becomes time consuming while resolving these errors, so I opted to use the [burn-book](https://github.com/tracel-ai/burn/tree/main/burn-book) present in their repository which has the latest updates but hasn't been pushed to production.

Hence due to inaccuracies in the burn book, first-burn-model consisted lot of errors, so I **cloned the burn repo** used and ran the model present in `./burn/examples/guide` using `cargo run --example guide` to observe the training process through the basic CLI dashboard:
<div align="center">
   <img width="1425" alt="Screenshot 2024-02-19 at 8 31 33 PM" src="https://github.com/sandeepB3/LFX-Mentorship-2024-PreTest-3172/assets/107111616/a19bccfc-baff-49d3-a58c-03f873e6daeb">
</div>

**Inference:** Hence using the burn book, I was able to understand the **Basic Burn Workflow of training to inference**, and obtained a CLI dashboard demonstrating the training progress of the train set and validation set, with the metrices of Loss and Accuracy against the number of iterations. Hence by the end, I understood the key components that serve as the building blocks of the framework: `Backend, Autodiff, Tensor, Module, Learner, Metric, Config, Record & Dataset`

### 2. Model Examples
The "examples" directory within the Burn repository offers a comprehensive implementation of models catering to various use cases such as image classification web, ONNX inference, text generation, simple regression, and more. During my review, I perused the contents of the `image-classification-web directory`. This demo showcases how to execute an image classification task in a web browser using a model converted to Rust code. The project utilizes the Burn deep learning framework, WebGPU and WebAssembly. Specifically, it demonstrates:
 - Converting an ONNX (Open Neural Networks Exchange) model into Rust code compatible with the Burn framework.
 - Executing the model within a web browser using WebGPU via the burn-wgpu backend and WebAssembly through the burn-ndarray and burn-candle backends.

Once the `github.com/tracel-ai/burn` repo is cloned, I opened it in the IDE of my choice and ran the following commands to see the working and execution of `image-classification-web` directory.

Building the WebAssembly Binary and Other Assets
```bash
cd examples/image-classification-web/
cargo install wasm-pack
./build-for-web.sh
```

Launching the Web Server
```bash
./run-server.sh
```

Accessing the Web Demo
```plaintext
http://localhost:8000
```
<div align="center">
   <img width="1470" alt="img5" src="https://github.com/sandeepB3/LFX-Mentorship-2024-PreTest-3172/assets/107111616/6bf2c55a-0bae-4793-aba2-620d828f02d6">
</div>

**Inference:** Upon examining the model examples, I gained insights into how the Burn framework facilitates the access of pretrained architectures and weights from other frameworks. As demonstrated earlier, the image classification web directory utilized the SqueezeNet model by loading the pretrained weights and architecture from the squeezenet1.onnx file. Hence, these examples can also be employed as a test suite for validating the WASI-NN Burn plugin once it is developed.

## Rustls Plugin








