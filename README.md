# Traffic Sign Recognition and Model Compression via Knowledge Distillation

## 📋 Overview
A project focused on maximizing the performance of Traffic Sign Recognition (GTSRB)—a core vision technology for autonomous driving and robotics—while executing **model compression** tailored for resource-constrained robotic environments. By implementing a **Knowledge Distillation (KD)** pipeline, the project successfully transferred knowledge from a high-capacity Teacher model to a lightweight Student model, exploring the optimal trade-off between computational efficiency and predictive accuracy.

## 💻 Jupyter Notebook Source Code
👉 [Click here to view the full interactive notebook via Kaggle](https://www.kaggle.com/code/dongmyungpark/gtsrb)

## 🛠 Tech Stack
- **Language:** Python
- **Frameworks & Libraries:** PyTorch, Torchvision
- **Architecture:** CNN, Knowledge Distillation (KD) Pipeline
- **Dataset:** GTSRB (German Traffic Sign Recognition Benchmark)
- **Domain Concepts:** Model Compression, Robotic Perception, Embedded/On-Device AI

## 💡 Key Features & Engineering

### 1. Knowledge Distillation Pipeline Implementation
- **Teacher-Student Architecture:** Guided the training of a lightweight Student model utilizing *soft labels* extracted from a highly complex, pre-trained Teacher model.
- **Custom Loss Optimization:** Implemented and optimized a custom KD Loss function, enabling the Student model to learn inter-class relationships and underlying distributions (*Dark Knowledge*) rather than merely memorizing hard targets.

### 2. Robotic Perception System Optimization & Analysis (`34212-Lab-S-Report.pdf`)
- **SOTA Trend Analysis:** Conducted an in-depth review of Deep Neural Network (DNN) trends in modern robotics, actively addressing critical bottlenecks such as annotation scarcity and domain shift.
- **Visual-Semantic Evaluation:** Analyzed the root causes of misclassifications through the lens of visual similarity and semantic structure, proposing robust reliability measures for safety-critical robotic systems.

### 3. Class-Specific Performance Enhancement
- **Targeted Speed Limit Analysis:** Specifically monitored and optimized performance metrics for speed limit signs (0~120km/h), which are hyper-critical for autonomous driving safety.
- **Quantitative Improvement:** Empirically verified significant accuracy percentage point (pp) gains in speed limit classes for the KD Student model compared to a standard lightweight baseline model.

## 🔥 Core Engineering Competencies

### 1. Model Compression for On-Device AI
- Successfully executed model reduction strategies to ensure real-time inference capabilities on resource-constrained hardware (embedded systems/robotics) while strictly minimizing accuracy degradation.

### 2. Granular Model Evaluation & Debugging
- Went beyond baseline accuracy metrics by conducting comprehensive Confusion Matrix analyses and visual structural evaluations to identify model vulnerabilities and engineer technically robust solutions.

## 📊 Results & Impact
- **Teacher Model:** Achieved peak baseline accuracy by leveraging deep architectural complexity.
- **Standard Student (Baseline):** Exhibited expected accuracy limitations and capacity bottlenecks due to reduced parameters.
- **KD Student:** **Significantly outperformed the Standard baseline**, demonstrating exceptional complex reasoning capabilities and achieving near-Teacher accuracy, particularly in critical autonomous driving classes (e.g., speed limits).
