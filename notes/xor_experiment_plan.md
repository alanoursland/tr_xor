**Brainstormed Outline for `xor_experiment_plan.md`**

1.  **Title:** Experiment Plan: Investigating Prototype Surface Learning with the XOR Problem

2.  **Introduction / Overview**
    * **1.1. Background:**
        * Briefly recap the core hypothesis of Prototype Surface Learning (PSL) theory: Neural networks (with linear layers + ReLU) learn prototype surfaces/regions, where classification is based on proximity/inclusion, and y=0 indicates maximal feature membership.
        * Mention the XOR problem as a classic, non-linearly separable task that requires a hidden layer, making it a simple yet informative testbed for geometric theories of NN function.
    * **1.2. Experiment Goal & Rationale:**
        * Primary Goal: To empirically investigate and demonstrate how a Multi-Layer Perceptron (MLP) trained on the XOR problem learns representations and performs classification in a manner consistent with the PSL theory.
        * Rationale: XOR's low dimensionality allows for direct visualization of learned geometric structures (hyperplanes, half-spaces/prototype regions) and their relation to input data points.
        * XOR is the classic problem motivating the need for linear separation. Applying PSL to it is a first step in changing the dominate viewpoint.
        * We perform small experiments in controlled environments to gain insight into learning properties of neural networks.
    * **1.3. Scope:** 
    This experiment focuses on analyzing the internal geometric representations of a successfully trained MLP, specifically the hyperplanes and zero-activation regions of its hidden neurons.

    This is exploritory work. It is not an attempt to prove PSL theory. It is an attempt to provide supporting evidence and explanitory examples. We perform a series of experiments with different architectures and activations. We gather data on learning effectiveness and representations. We also explore error functions, optimizers, initialization strategies and data normalization. These experiments aim to raise more questions than it answers. New experiments will be performed in response to questions.

3. **PSL Theory Applied to XOR**

4. **Predicted Learning Properties**

5. **Overview of Methodology**
    * GPU accelerated Pytorch.
    * Exploration of the XOR and higher dimensional Parity problem.
    * Analysis and Evaluation
    * Experiment configuration:
        - Experiment goals (theory being tested; questions being answered; problems being solved; )
        - Training Procedure: architecture; loss function; optimizer; initialization; batch size; epochs; run count.
        - Data Collection: final models; training curves; accuracy histogram across runs; 
        - Metrics: point distance from surface; mirror pairs; 
        - Visualizations: graph 2d hyperplanes against training data
        - Discussion and observations of results
    

6. **Experiment Enumeration**

    This list will be updated as we perform experiments and discover new questions to ask.

