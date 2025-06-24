# Prototype Surface Learning in Neural Networks

## Overview

This research project investigates a fundamental reinterpretation of how neural networks learn: rather than constructing decision boundaries that separate classes, neural networks learn **prototype surfaces** that characterize where each class naturally exists in feature space. Classification then emerges from measuring geometric deviation from these learned surfaces.

The key insight is that **zero activation indicates membership** in a learned prototype region, while non-zero activation measures distance from it. This inverts the traditional interpretation where high activation indicates strong feature presence.

## Table of Contents

### Part I: Theoretical Foundation

1. **[Project Overview](notes/overview.md)** - Research goals and approach
   - Core hypothesis and research methodology
   - Significance for understanding neural network behavior
   - Approach through controlled experiments

2. **[Core Theory](notes/core_theory.md)** - The mathematical framework of prototype surface learning
   - How linear transformations encode signed distances to hyperplanes
   - ReLU as one-sided prototype region detection
   - Absolute value as two-sided distance measurement
   - Networks as hierarchical prototype surface composers

3. **[The XOR Problem](notes/xor_problem.md)** - Historical context and theoretical importance
   - Minsky & Papert's analysis of perceptron limitations
   - Why XOR is the perfect test case for prototype surface theory
   - From linear separability to geometric learning

4. **[XOR Experiment Plan](notes/xor_experiment_plan.md)** - Detailed experimental methodology
  - Hypothesis-driven experimental design
  - Geometric metrics and visualization approaches
  - Iterative refinement based on findings


### Part II: Experiments & Results

#### Experiment 1: Single Absolute Value Unit

Can a minimal neural network—just one neuron with absolute value activation—solve XOR by learning a prototype surface?

- **[Experiment Overview](reports/abs1/abs1_overview.md)** - Architecture and methodology
  - Model: `y = |w₁x₁ + w₂x₂ + b|`
  - 250 runs across 5 initialization strategies
  - Geometric metrics for prototype surface analysis

- **[Results](reports/abs1/abs1_results.md)** - Empirical findings
  - **100% success rate** across all initialization strategies
  - Universal convergence to optimal geometry
  - Prototype surface consistently intersects False class points
  - True class points at exactly √2 distance

- **[Discussion](reports/abs1/abs1_discussion.md)** - Interpretation and implications
  - Analytical proof of optimal surface configuration
  - Scale vs. orientation in learning dynamics
  - The neuron as a calibrated distance function

Raw data and diagrams are available in project report directories.

#### Experiment 2: Symmetric ReLU Pair

Can two ReLU units learn to mimic absolute value through discovered symmetry? What failure modes emerge?

- **[Experiment Overview](reports/relu1/relu1_overview.md)** - Testing structural vs learned symmetry
  - Model: `y = ReLU(w₁·x + b₁) + ReLU(w₂·x + b₂)`
  - Hierarchy of failure modes discovered through iterative experiments
  - From 42% failure rate to 100% success through geometric insights

- **[Results](reports/relu1/relu1_results.md)** - Failure modes and interventions
  - Standard initialization: 58% success rate
  - Dead data point elimination: 98% success rate
  - Margin constraints: 100% success rate
  - Mirror initialization: 98.4% success rate (perpendicular trap discovered)

- **[Discussion](reports/relu1/relu1_discussion.md)** - Geometric insights
  - Dead data points starve gradient flow
  - Small margins lead to training instability
  - Perpendicular initialization creates zero-gradient traps
  - Structural bias vs. discovered symmetry

Raw data and diagrams are available in project report directories.

## Key Findings

1. **Neurons are distance computers, not feature detectors** - They measure deviation from learned prototype surfaces rather than detecting feature presence.

2. **Zero activation indicates prototype membership** - The regions where neurons output zero define the learned prototypes, not where they activate strongly.

3. **Geometric initialization determines learning success** - Failures arise from specific geometric configurations (dead points, small margins, perpendicular orientations) rather than random chance.

4. **Structural bias provides robustness** - Absolute value's enforced symmetry eliminates failure modes that ReLU pairs must overcome through learning.

5. **Local minima have geometric signatures** - Failed configurations correspond to specific geometric arrangements (horizontal/vertical hyperplanes for XOR).

## Getting Started

### Requirements
- Python 3.8+
- PyTorch
- NumPy, Matplotlib, Scikit-learn
- CUDA-capable GPU (recommended)

### Running Experiments
```bash
cd src
python run.py [experiment_name]  # e.g., abs1_normal, relu1_mirror
python analyze.py [experiment_name]
```

### Exploring Results
- Raw results are stored in `src/results/`
- Analyzed summaries in `reports/[experiment]/data/`
- Visualizations in `reports/[experiment]/figures/`

## Contributing & Future Work

### Open Questions
- How do these principles scale to deeper networks and higher dimensions?
- Can we design initialization schemes that guarantee prototype surface formation?
- What is the relationship between prototype surfaces and adversarial examples?
- How do different activation functions affect prototype geometry?

## Status

🚧 **This is active research in progress** 🚧

The theories, experiments, and conclusions presented here are evolving. Results and interpretations may change as the research develops. Feel free to explore and discuss, but please note that this is not yet a finished work.

## License

- **Code**: MIT License - see [LICENSE](LICENSE)
- **Research content** (text, figures, theoretical framework): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

This means:
- You may use, modify, and share the **code** freely under MIT terms.
- You may reuse or adapt the **research content**, provided you give proper attribution.

## Citation

If reusing research materials, please credit:
```bibtex
@misc{oursland2025prototype,
  author       = {Oursland, Alan},
  title        = {Prototype Surface Learning in Neural Networks: Evidence from XOR (Work in Progress)},
  year         = {2025},
  howpublished = {\url{https://github.com/alanoursland/tr_xor}},
  note         = {GitHub repository, accessed 2025-06-24}
}
```

---

*This research provides empirical evidence for a geometric theory of neural network learning, challenging conventional interpretations and opening new directions for understanding deep learning.*