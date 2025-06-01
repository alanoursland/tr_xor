**Outline XOR Experimental Plan**

**1. Title:** Experiment Plan: Investigating Prototype Surface Learning with the XOR Problem

**2. Introduction**
    * **2.1. Core Research Context:**
        * Briefly introduce the Prototype Surface theory: Neural networks (particularly those with linear layers and ReLU activations) are hypothesized to learn "prototype surfaces" or regions.
        * Emphasize the key thesis: Classification is based on proximity/inclusion relative to these surfaces, where 0=Wx+b with an activation defines one or more prototype surfaces. An activation value of zero indicates maximal membership or inclusion within a learned feature/prototype region.
    * **2.2. The XOR Problem as a Testbed:**
        * Highlight the XOR problem as a classic, simple, non-linearly separable task. Its low dimensionality is ideal for visualizing learned geometric structures.
        * Position this investigation as a foundational step: Investigation using XOR, a problem that historically demonstrated the limitations of linear classifiers and motivated the need for non-linear solutions (like MLPs), offers a chance to reinterpret learning mechanisms from the prototype surface perspective.
        * XOR as a primary foundational linear separation theory vs prototype surface theory.
        * Exploritory work. Illustrative evidence.
    * **2.3. Experiment Goals and Exploratory Nature:**
        * Primary Goal: To empirically investigate, visualize, and document how a Multi-Layer Perceptron (MLP) trained on XOR learns representations and performs classification in a manner consistent with the prototype surface theory.
        * Nature of Work: This is exploratory research. It's not an attempt to definitively prove prototype surface theory but rather to provide supporting evidence, generate explanatory examples, and identify new research questions.
        * Iterative Approach: This experiment is part of an anticipated series. Findings and new questions will inform subsequent experiments (e.g., exploring different architectures, activations, error functions, optimizers, initialization strategies, data normalization, and higher-dimensional Parity problems).
    * **2.4. Scope of This Document:**
        * This document details the plan for initial experiments focused on analyzing the internal geometric representations (hyperplanes and "prototype regions" where `ReLU(Wx+b)=0`) of hidden neurons in an MLP successfully trained on the XOR problem.

**3. Theoretical Application: Prototype Surface Theory Applied to XOR**
    * Describe how the general prototype surface theory is expected to manifest specifically for an MLP solving XOR.
    * How hidden neurons are predicted to define prototype regions (`Activation(Wx+b) = 0`).
    * How the combination of these regions (and the signals of inclusion/exclusion) allows the output neuron to solve the XOR logic.
    * Discuss the interpretation of the output layer's function within the prototype surface theory for this binary classification task.
    * ReLU's zero-activation regions encapsulating prototypes (and positive activation signaling non-prototypical inputs) 
    * Abs's zero-activation decision boundary encapsulating prototypes (and positive activation signaling non-prototypical inputs) 
    * Sigmoids's positive and negative extremes.
    * emphasize the geometric interpretation (hyperplanes, half-spaces as prototype regions) rather than explicitly naming them as learned "distance metrics"
    * The metric "point distance from surface" framed carefully as "deviation from the prototype surface (Wx+b=0)" rather than as the network learning a formal "distance metric."
    * 

**4. Predicted Learning Properties and Observations**
    * What specific geometric configurations of hyperplanes and prototype regions are anticipated in the hidden layer(s)?
    * How are these regions expected to relate to the specific XOR input patterns (e.g., (0,0), (0,1), (1,0), (1,1))?
    * Predictions about the state of hidden neurons (active vs. zero-output) for different inputs.
    * Expected consistency or variability of learned geometric solutions across different training runs.
    * The contrast between Abs/ReLU boundaries passing through points and Sigmoid boundaries lying between them. Sigmoids create two prototype surfaces.
    * Mirror pairs of weights where W_i = -W_j supporting ReLU(x) and ReLU(-x)

**5. Experimental Methodology**
    * **5.1. Computational Environment:**
        * Primarily GPU-accelerated PyTorch.
    * **5.2. Core Problem Focus:**
        * Initial experiments: XOR problem.
        * Context for future extension: Higher-dimensional Parity problems.
    * **5.3. Experiment Configuration Parameters (to be detailed for each specific experiment):**
        * **5.3.1. Model Architecture:**
            * Type: MLP. Specific architectures will be described in each experiment.
        * **5.3.2. Dataset:** Standard XOR truth table. Normalized vs Unnormalized.
        * **5.3.3. Training Procedure:**
            * Loss Function (e.g., Binary Cross-Entropy).
            * Optimizer (e.g., Adam, SGD with specific parameters).
            * Weight Initialization strategy.
            * Batch Size (e.g., full batch for XOR).
            * Number of Epochs (or convergence criteria).
            * Number of training runs (with different random seeds).
    * **5.4. Data Collection:**
        * Final trained model parameters (weights and biases).
        * Training curves (loss and accuracy over epochs).
        * Accuracy histogram across multiple runs.
        * Activations (pre and post-ReLU for hidden, pre-sigmoid and post-sigmoid for output) for all XOR inputs.
    * **5.5. Analysis, Metrics, and Visualization:**
        * **Geometric Visualization:** Plot 2D hyperplanes (`Wx+b=0`) of hidden neurons against training data. Identify the "prototype regions" for different activations.
        * **Quantitative Metrics (Examples):**
            * Distance of each input point to each hidden neuron's hyperplane (`Wx+b=0`).
            * Analysis of weight "mirror pairs" (W_0 = -W_1).
        * **Representation Analysis:** Examine hidden layer activation patterns for each XOR input.
    * **5.6. Discussion of Results (for each experiment):**
        * Observations on learned representations.
        * Alignment (or divergence) with prototype surface theory predictions.
        * New questions or insights gained.

**6. Experiment Enumeration and Log**
    * This section will list specific experiments conducted under this plan.
    * Each experiment entry will detail its specific configuration (from 5.3), link to results/visualizations, and summarize key observations and discussion points.
    * This list will be updated iteratively as experiments are performed and new questions arise.
