# Repivoting: From Theory to Practice - Data-Driven vs Random Pivot Selection

## Introduction: The Geometry of Stuck Hyperplanes

Gradient-based training of affine units can encounter a degenerate condition where the hyperplane becomes unable to rotate effectively. This occurs when the error signals from misclassified examples align with the pivot point (typically the origin), creating a configuration where gradients only scale the weight vector W rather than rotating it.

The fundamental observation is that during training, a hyperplane pivots around a fixed point when the weight vector W is updated. For the affine function y = Wx + b, this pivot point is at (x=0, y=b) - the only point unaffected by changes to W. This becomes clearer when written in point-slope form: y - b = W(x - 0). When misclassified examples happen to line up with this pivot point at the origin (when x=0), the gradient provides no rotational "torque," leaving the hyperplane stuck in its current orientation.

## The Random Repivoting Approach

To address this issue, we implemented a technique that randomly repositions the pivot point during training. The approach works by maintaining a movable pivot point x0 and reformulating the linear transformation as:

y = W(x - x0) + b

At each training step, the algorithm:
1. Samples a new pivot location x0_new from a Gaussian distribution with standard deviation σ
2. Computes the shift delta = x0_new - x0_old
3. Adjusts the bias to maintain the same hyperplane: b_new = b_old + W * delta
4. Updates the stored pivot to x0_new

This process preserves the model's predictions (since W(x - x0_new) + b_new = W(x - x0_old) + b_old) while changing the point around which future weight updates will rotate the hyperplane. The standard deviation σ controls the magnitude of pivot movements.

This document presents experimental results from testing random repivoting, revealing important insights about when pivot manipulation helps or harms training.

## Experimental Investigation: Random Pivoting

We test random pivot selection using a small controlled model and problem to explore its potential.

### Experimental Setup

We tested on the XOR problem using neural networks with varying architectures:
- Simple: Linear(2,1) -> Abs
- Complex: Linear(2,2) -> ReLU -> Sum

Each forward pass, a new pivot is randomly selected near the origin from N(0, σ). The bias is then adjusted so that the hyperplane geometry is unchanged. 

## Key Findings

### Finding 1: Optimizer Compatibility Matters

```
Results with Linear(2,1) -> Abs architecture:

Optimizer | Layer Type           | Median Convergence (Epochs)
----------|---------------------|---------------------------
Adam      | Standard nn.Linear  | 155
Adam      | RandomPivotLinear   | 191
SGD       | Standard nn.Linear  | 392
SGD       | RandomPivotLinear   | 394
```

**Critical Insight**: Random pivoting interferes with Adam's adaptive mechanisms but has minimal impact with SGD. The manual bias adjustments desynchronize Adam's internal momentum and learning rate estimates, leading to slower convergence.

### Finding 2: Conditional Benefits in Difficult Landscapes

```
Results with Linear(2,2) -> ReLU -> Sum (SGD optimizer):

Layer Type         | Success Rate | Median Convergence | Fastest Run
-------------------|--------------|-------------------|-------------
Standard nn.Linear | 25/50 (50%) | 607 epochs       | 304 epochs
RandomPivotLinear  | 23/50 (46%) | 580 epochs       | 177 epochs
```

When SGD faces a challenging landscape with local minima, random pivoting showed modest benefits for successful runs, though it may hurt the overall success rate (we didn't perform t-tests).

### Finding 3: Harmful Effects with Good Initialization

Previous experiments showed that "mirror initialization" improved the ReLU model performace from 48% success to 98.4% with the holdout failures being the "no rotation trap" described above. Mirror initialzation just ensures that the second half of nodes have parameters that are negatives of the first half.

When using mirror initialization, random pivot selection hurt the success rate:

```
Pivot Configuration | Success Rate
--------------------|-------------
No pivoting (σ=0.0) | 99.8%
Random pivot (σ=1.0)| 90.0%
```

**Critical Insight**: Random perturbations destabilize well-initialized networks, knocking them out of good basins of attraction.

### Finding 4: Dose-Response Relationship

We thought there might be a "sweet spot" in the pivot distance from the origin and compared performance across several values σ for N(0, σ)

```
Effect of Pivot Distance (with mirror initialization):

Pivot Sigma | Success Rate | Failed Runs
------------|--------------|-------------
0.00        | 99.8%       | 1/500
0.10        | 99.6%       | 2/500
0.25        | 99.4%       | 3/500
0.50        | 98.0%       | 10/500
1.00        | 90.0%       | 50/500
```

The degradation in performance shows a clear monotonic relationship with the magnitude of random perturbations. If there is a peak, it is narrow and we didn't find it.

## Implications for Data-Driven Repivoting

These experiments with random pivoting provide valuable insights for the original data-driven proposal:

### Strengths Validated

1. **Parameterization Impact**: Even zero-loss reparameterizations significantly affect optimization trajectories, confirming that pivot location matters
2. **Rotational Acceleration**: In challenging landscapes, pivot manipulation can accelerate convergence when successful
3. **Mechanism Validity**: The powerful effects observed validate that pivot-based interventions can meaningfully influence training

### Concerns Highlighted

1. **Optimizer Interference**: Adam does not expect the gradient landscape to change. Changing the pivot slows Adam down. The gradient momentum that Adam tracks may average out the gradients of different pivots.
2. **Stability Risks**: Random pivots pushed our model into degenerate states. 


## Practical Recommendations

Based on these findings:

1. **For Simple Optimizers (SGD)**: Both random and data-driven pivoting may provide benefits in challenging landscapes
2. **For Adaptive Optimizers (Adam)**: Careful integration would be needed to avoid disrupting internal state
3. **For Well-Initialized Networks**: Any pivoting mechanism should include safeguards against destabilizing good solutions
4. **For Implementation**: Consider exponential moving averages of pivot locations to reduce mini-batch variance

## Conclusion

The experimental investigation of random pivoting validates the core hypothesis that pivot manipulation can significantly influence neural network training. While random perturbations showed limited benefits and significant risks, they demonstrated the potency of the mechanism itself.

