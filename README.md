# LFX Mentorship 2024 Program - WasmEdge PreTest 

#### This repository contains documentation for the [pretest](https://github.com/WasmEdge/WasmEdge/discussions/3182) and proposal submitted for the LFX Mentorship 2024 Program, focusing on the integration of burn.rs as a new backend for WASI-NN in the [WasmEdge](https://github.com/WasmEdge/WasmEdge/issues/3172) project.

- **Applied:** [To integrate burn.rs as a new WASI-NN backend](https://github.com/WasmEdge/WasmEdge/issues/3172)
- **Host OS:** macOS Sonoma Version 14.1

## 📚 Table Of Contents
### 1. Framework Execution: Burn.rs
   - [The Burn Book](#1-the-burn-book)
   - [Model Examples](#2-model-examples)

### 2. Rustls Plugin
   - [Building and Installing the Plugin](#1-building-and-installing-the-plugin)
   - [Summary & Execution Result of Example](#2-summary--execution-result-of-example)

### 3. Approach for Building the WASI-NN Burn Backend
   - [Analogy to Rustls Plugin](#1-analogy-to-rustls-plugin)
   - [Analogy to WASI-NN PyTorch Plugin](#2-analogy-to-wasi-nn-pytorch-plugin)
     
### 4. Miscellaneous
   - [Applicant Details](#1-applicant-details)
   - [References](#2-references)

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
### 1. Building and Installing the Plugin

> **Understanding the Plugin:** WasmEdge Rustls plug-in is a component that allows Rust programs running on WasmEdge to use the Rustls library instead of OpenSSL for secure communication, providing a more modern and secure option.

As per the provided [guide](https://wasmedge.org/docs/contribute/source/plugin/rusttls), to build and execute the rustls plugin. The prerequisites are satisfied:

#### Installing CMake
```bash
brew install cmake
```
<div align="center">
   <img width="798" alt="img6" src="https://github.com/sandeepB3/LFX-Mentorship-2024-PreTest-3172/assets/107111616/53afbd39-15c3-4162-8292-e87c25afed0c">
</div>

#### Installing WasmEdge runtime for current user following the [installation guide](https://wasmedge.org/docs/start/install/)
```bash
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash
```
Running the following command to make the installed binary available in the current session
```bash
source $HOME/.wasmedge/env
```
Moving on, I have cloned the WasmEdge repository, selecting the specified branch `hydai/0.13.5_ggml_lts` as outlined in the pre-test instructions.
```bash
git clone -b hydai/0.13.5_ggml_lts --single-branch https://github.com/WasmEdge/WasmEdge.git
```
<div align="center">
   <img width="688" alt="img7" src="https://github.com/sandeepB3/LFX-Mentorship-2024-PreTest-3172/assets/107111616/924f903d-1020-40b7-b892-fd9094264266">
</div>

Navigating to the Rustls Plug-in Directory & Building the Plug-in 
```bash
cd WasmEdge/plugins/wasmedge_rustls
cargo build --release
```
<div align="center">
   <img width="757" alt="Screenshot 2024-02-19 at 1 35 01 PM" src="https://github.com/sandeepB3/LFX-Mentorship-2024-PreTest-3172/assets/107111616/0409db49-2003-41a6-9c29-bc8410ca6a09">
</div>

Following which, the system has generated a libwasmedge_rustls.dylib dynamic library file in the target/release directory. To install the rustls plug-in for user-specific purposes, we copy the libwasmedge_rustls.so file to the ~/.wasmedge/plugin folder.
> In macOS we obtain the (libwasmedge_rustls.dylib) dynamic library file, which is essentially the same concept as (libwasmedge_rustls.so) shared object files on Unix-like systems (e.g., Linux).
```bash
cd target/release
cp libwasmedge_rustls.dylib ~/.wasmedge/plugin/
```
<div align="center">
   <img width="759" alt="img9" src="https://github.com/sandeepB3/LFX-Mentorship-2024-PreTest-3172/assets/107111616/c4ae37d3-0fbc-4aea-b921-b26b829744dc">
</div>

Added the wasm32-wasi as a compilation target for rust.
```bash
cd ~ 
rustup target add wasm32-wasi
```
**Inference:** We are now prepared to execute the examples, demonstrating the utilization of the rustls plug-in and providing a summary of your build process and the results obtained during execution.

### 2. Summary & Execution Result of Example

I have chosen the [wasmedge_hyper_demo/client-https](https://github.com/WasmEdge/wasmedge_hyper_demo/tree/main/client-https) as my example demonstration.
> **Understanding the code:** When we navigate to the `wasmedge_hyper_demo/client-https/src/main.rs`, we see that the provided rust program acts as an HTTP client using the Hyper library and communicates with a server over HTTPS. In this case, the HTTPS communication is facilitated by the wasmedge_hyper_rustls crate, which is designed to be used in WebAssembly (WasmEdge) environments. The wasmedge_hyper_rustls crate uses the Rustls library (an alternative to OpenSSL) under the hood for handling the TLS (Transport Layer Security) encryption.

<div align="center">
   <img width="784" alt="img10" src="https://github.com/sandeepB3/LFX-Mentorship-2024-PreTest-3172/assets/107111616/c58e67cd-93f3-4ad5-8a4f-e0cc86ca26cf">
</div>

**To demonstrate this example we clone the wasmedge_hyper_demo repository, navigate to client-https and build the client-https project with the wasm32-wasi compilation target.**
```bash
git clone https://github.com/WasmEdge/wasmedge_hyper_demo
cd wasmedge_hyper_demo/client-https
cargo build --target wasm32-wasi --release
```
<div align="center">
   <img width="763" alt="img11" src="https://github.com/sandeepB3/LFX-Mentorship-2024-PreTest-3172/assets/107111616/36124d63-13ad-4342-ab44-d31cb3c9c93c">
</div>

Now, we compile the `wasmedge_hyper_client_https.wasm` file present in `target/wasm32-wasi/release` directory as `hyper_client_https.wasm` file to the current `client-https` directory.
```bash
wasmedge compile target/wasm32-wasi/release/wasmedg e_hyper_client_https.wasm hyper_client_https.wasm
```
<div align="center">
   <img width="762" alt="Screenshot 2024-02-19 at 2 22 47 PM" src="https://github.com/sandeepB3/LFX-Mentorship-2024-PreTest-3172/assets/107111616/da86b75c-4f4f-4ebd-972e-201453bc6ffe">
</div>

Finally, we use the wasmedge runtime (installed with the rustls plugin to execute the `hyper_client_https.wasm` file.
```bash
wasmedge hyper_client_https.wasm
```
<div align="center">
   <img width="769" alt="img13" src="https://github.com/sandeepB3/LFX-Mentorship-2024-PreTest-3172/assets/107111616/50ff86ee-7359-4672-bc68-ff9e1403247f">
</div>

**Inference:** As demonstrated above we built and compiled an example rust program that acts as an HTTP client using the Hyper library and communicates with a server over HTTPS, and this communication is facilitated by wasmedge_hyper_rustls crate which utilizes the rustls plugin. Hence, as per my understanding we had already installed the rustls plugin to our wasmedge runtime, so when we executed the `hyper_client_https.wasm` file the rustls plugin was called to extends its use.

## Approach for Building the WASI-NN Burn Backend
### 1. Analogy to Rustls Plugin

In the process of building and installing the plugin, let's delve into the actual implementation of the Rustls plugin. If we examine the `lib.rs` file situated at `WasmEdge/plugins/wasmedge_rustls/src /lib.rs`, we observe that it defines a Rust module functioning as a plugin for the Wasmedge runtime. This module, named "rustls_client," encapsulates functionalities associated with the Rustls TLS library, catering to secure communication needs. Within its scope, the module encompasses error handling mechanisms, TLS connection management, and various I/O operations.

<div align="center">
   <img width="1459" alt="img14" src="https://github.com/sandeepB3/LFX-Mentorship-2024-PreTest-3172/assets/107111616/6ef2f066-c13f-4fcf-bfa9-df25e8489d92">
</div>

From here if we look into the [installation guidelines](https://wasmedge.org/docs/start/install#install-wasmedge-plug-ins-and-dependencies) of WasmEdge, we find a section which tell's us how we can extend WasmEdge's functionality by installing plug-ins and dependencies.

To install **WasmEdge with the TLS plug-in**, we run the following command.
```bash
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugins wasmedge_rustls
```

From here we can conclude that all plugin implementations and installation follows a similar procedure, so keeping this analogy of plugin development we look into the WASI-NN plugin, particularly with the PyTorch backend.

### 2. Analogy to WASI-NN PyTorch Plugin

We will firstly be looking into the plugin implementation for the WASI-NN Pytorch Backend, for which we examine the `torch.h` file and `torch.cpp` file situated at `WasmEdge/plugins/wasi_nn`, the `torch.h` file provids C++ code that defines a set of structures and functions related to WebAssembly System Interface for Neural Networks (WASINN) with a focus on PyTorch integration and the `torch.cpp` file provides code that facilitates the integration of PyTorch models into the WASINN environment, allowing for the execution of neural network computations in a WebAssembly context. It handles loading models, setting inputs, performing computations, and retrieving outputs, with conditional compilation for PyTorch backend support.

Once implemented, one can build and intsall this plugin in a similar fashion to how we built and intsalled the rustls plugin to the WasmEdge runtime. Additionally one can also install the WASI-NN Pytorch backend plugin by running the following command:
```bash
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugins wasmedge_rustls
```

Now we look into a test suite for validating the WASI-NN Pytorch backend plugin, we refer to the following [documentation](https://wasmedge.org/docs/develop/rust/wasinn/pytorch/):
<div align="center">
   <img width="891" alt="img15" src="https://github.com/sandeepB3/LFX-Mentorship-2024-PreTest-3172/assets/107111616/c9855db9-e959-44f3-b55a-ec0c49ae4c02">
</div>

Thus, once the WASI-NN Burn backend is implemented we can build and execute a burn example model in a similar fashion as shown above.

## Miscellaneous
### 1. Applicant Details
I have applied in the LFX Mentorship Portal under WasmEdge for the issue: Integrate of burn.rs as a new backend for WASI-NN, following are my submissions:

**Cover Letter:** https://drive.google.com/file/d/1ih75KwN0RYGGGsUopEbE34drxzVHNFf9/view?usp=sharing

**Resume:** https://drive.google.com/file/d/1_zy1pqf3RQ2bsGObFZDKPv-bpFtU_oRk/view?usp=sharing


### 2. References
https://wasmedge.org/docs/develop/rust/wasinn/pytorch/

https://wasmedge.org/docs/contribute/source/plugin/rusttls/

https://github.com/second-state/WasmEdge-WASINN-examples/

https://wasmedge.org/docs/start/install/

https://github.com/WebAssembly/wasi-nn/

https://github.com/WasmEdge/WasmEdge/

https://github.com/tracel-ai/models/

https://github.com/tracel-ai/burn/

https://burn.dev/book/








