# Init Loss Sampling for Test Partitioning

## Overview

**Init Loss Sampling for Test Partitioning** is a proposed dataset splitting strategy
that uses the loss distribution from a *randomly initialized model* to guide how
examples are allocated into training and test sets.

Rather than relying purely on label proportions or random splits, this method
leverages early model behavior as a proxy for **sample difficulty**.

## Motivation

In standard machine learning practice:

- Dataset partitions are created independently of the model, often using fixed
  official splits or random sampling.
- Stratification is applied based on labels to preserve class balance, not model performance.

This approach ignores potentially useful information about how "hard" each example
appears to a model **before any training has occurred**.

### Key Insight

If a random network's *initial loss* on a sample correlates with its eventual
learning dynamics, then the loss distribution at initialization could be used to:

- Ensure the test set is *balanced in difficulty* relative to the training set.
- Construct harder or easier test sets intentionally.
- Reduce evaluation noise due to unbalanced sample difficulty.

## Method

1. **Random Initialization**
   - Select an architecture and randomize weights using a standard initialization
     (e.g., Kaiming Normal, Xavier).
   - Freeze model parameters.

2. **Loss Sampling**
   - Evaluate all samples in the dataset (train + test candidates) with the
     untrained model in `eval()` mode.
   - Compute the per-sample loss (e.g., cross-entropy loss for classification).

3. **Difficulty Stratification**
   - Treat the initial loss as a *difficulty score*.
   - Stratify the dataset by difficulty score and label simultaneously.
   - Create partitions that match the loss distribution between train and test sets,
     or skew intentionally for experimental purposes.

4. **Optional: Multiple Initializations**
   - Repeat the above for `k` random seeds to reduce noise in the difficulty
     estimate.
   - Average per-sample losses to form a more stable difficulty score.

## Advantages

- **Difficulty balance**: Prevents the test set from being accidentally easier or
  harder than the training set.
- **Custom test design**: Enables construction of targeted evaluation sets.
- **Model-specific**: Accounts for inductive biases of the architecture.

## Risks and Considerations

- **Potential bias**: Tailoring the test set to the architecture may reduce
  generality of results.
- **Overfitting risk**: Using model-informed partitioning could introduce subtle
  leakage if hyperparameters are tuned on the same model type.
- **Compute overhead**: Requires evaluating the entire dataset at least once per
  initialization.

## Applications

- **Fairer benchmarks**: Especially in small datasets where random splits can
  significantly shift difficulty.
- **Curriculum learning research**: Understanding the relationship between
  perceived difficulty at init and learning progression.
- **Robustness testing**: Designing stress-test splits.

## Conclusion

Init Loss Sampling for Test Partitioning introduces a novel dimension to dataset
splitting: difficulty-awareness from a model's untrained state. By leveraging
initial loss statistics, we can aim for more balanced and intentional train/test
splits, potentially improving reproducibility and interpretability of results.

