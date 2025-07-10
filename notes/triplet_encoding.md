### Core geometry & activations

```
Prototype Surface     —defines→   Class Region
ReLU Neuron           —extends→   Prototype Region
|x| Activation        —encodes→   Unsigned Distance
Sigmoid Activation    —has→       Two Moving Prototype Surfaces
Tanh Activation       —has→       Two Moving Prototype Surfaces
Smooth Activation     —decouples→ Prototype Surface  (from) Decision Boundary
Hyperplane            —anchors→   Prototype Surface
Zero Activation       —signals→   Prototype Membership
Positive Activation   —signals→   Prototype Deviation
Distance (ℓ₂)         —proportional→ Convergence Time
Loss Magnitude        —proxy for→ Distance From Solution
```

### Failure modes & geometry

```
Dead Data Point       —causes→    Zero Gradient
Small Margin          —risks→     Dead Data Point
Out-of-Bounds Neuron  —removes→   Useful Gradient
Perpendicular Trap    —creates→   Local Minimum (50 % acc)
Pathology            —detected by→ Composite Temperature
```

### Monitors & interventions

```
DeadSampleMonitor     —detects→   Dead Data Point
BoundsMonitor         —detects→   Out-of-Bounds Neuron
AnnealingMonitor      —computes→  Composite Temperature
Composite Temperature —controls→  Noise Injection
Error-Driven Annealing —injects→   Corrective Noise
Corrective Noise      —revives→    Dead Data Point
Mirror Initialiser    —enforces→  Hyperplane Symmetry
BHS Initialiser       —guarantees→ No Dead Data
Loss-Based Sampling   —selects→    Low-Loss Initial State
Low-Loss Init         —reduces→    Convergence Epochs
Magnitude Eater       —regularises→ Feature Norm
Monitoring Hooks      —supply→     Real-Time Geometry
```

### Empirical findings

```
Distance-Time Law     —observed-in→ Abs1 Model
Distance-Time Law     —extends-to→ ReLU1 Model
Distance-Time Law     —weakens-in→ ReLU2 Model
Linear Regression     —predicts→   Epochs Remaining
Random Forest         —perfectly-predicts→ Epochs Remaining
Loss < 0.072          —implies→    100 % Convergence (ReLU1)
Temperature ≈ 0       —implies→    Healthy Training
Temperature ≫ 0       —triggers→   Noise Injection
```

### Architectural links

```
Linear Layer          —feeds→      ReLU Activation
ReLU Activation       —decomposes→ |x| via  (ReLU + ReLU(−x))
Absolute-Value Layer  —feeds→      Static Scale
Static Scale          —isolates→   Weight Norms
First-Layer Distance  —drives→     Second-Layer Logits
```

### Methodological scaffold

```
Prototype Theory      —informs→    Initialisation Design
Geometry Logging      —supports→   Empirical Diagnosis
Toy XOR Dataset       —exposes→    Pathologies Early
Monitoring Framework  —bridges→    Theory and Practice
```

*This graph isn’t exhaustive, but it captures the main nodes and edges that have accumulated across your experiments and discussions.*
