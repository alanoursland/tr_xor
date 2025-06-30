### 1  What you are visualising → “sawtooth-ReLU”

A **sawtooth-ReLU** is just a ReLU ramp with a small-amplitude, high-frequency triangular (or saw) wave super-imposed on its positive branch

$$
\text{SawReLU}_{A,f}(z)\;=\;
\underbrace{\max(0,z)}_{\text{plain ReLU}}
\;+\;
A\;{\sf saw}\!\bigl(f\,z\bigr)\;\;\;\;(\;0<A\ll 1,\;f\ge 1\;),
$$

where
${\sf saw}(t)=2\!\bigl(t-\lfloor t+\tfrac12\rfloor\bigr)\in[-1,1]$ is a unit-amplitude sawtooth.
Visually it looks like gold-panning riffles on a shallow slope.

---

### 2  Do people actually use something like this?

* **Hopfield-style networks.**
  Discontinuous sawtooth activations were designed to induce **multistability** and fast associative recall in continuous-time Hopfield nets .
* **Theory of depth separation.**
  In expressivity proofs (Telgarsky 2016; Corlay et al. 2020) a *k-sawtooth* ReLU—ReLU composed with $k$ equally-spaced folds—appears as a *target* function that shallow networks cannot approximate efficiently ([arXiv][1]).
  Those papers do not use the sawtooth as an **activation**, but the math you quoted,
  $\text{SepOrd}=2k+1$, comes from that line of work.
* **Forums & toy demos.**
  There are scattered hobby experiments (e.g. a “sawtooth ReLU” demo for Nengo ) but nothing mainstream in deep-learning practice yet.

So the idea is *out there*, just not (yet) as a standard Keras/PyTorch layer.

---

### 3  Why it matters in the “separation-order” lens

| Property                                                 | Monotone ReLU | SawReLU with $k$ teeth |
| -------------------------------------------------------- | ------------- | ---------------------- |
| # strict monotone pieces                                 | 1             | $k+1$                  |
| **SepOrd**                                               | 1             | $k+1$                  |
| Half-spaces one neuron can union after a final threshold | 1             | $k+1$                  |

*With $k\ge1$ the activation is non-monotone ⇒ SepOrd ≥ 2 ⇒ a **single neuron can implement 2-nd-order predicates like XOR**.*
Increase $k$ and you can, in principle, fold ever more intricate decision sets into one unit—exactly your intuition about “lots of shallow basins”.

---

### 4  Optimisation-side pros & cons

| 👍 potential upsides                                                                                                | 👎 risks & caveats                                                                                                                                                     |                                                                                  |                                                                                                                            |
| ------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **Extra ridges ≈ implicit weight decay.**  Small-amplitude teeth keep gradients (                                   | \partial\sigma/\partial z                                                                                                                                              | \approx1) but add a periodic *sign* flip that discourages runaway weight growth. | **Spurious local minima.**  Each tooth inserts a small plateau in weight-space; too large an amplitude can trap SGD early. |
| **Controlled quantisation.**  Activations are nudged towards the ridge bottoms → natural clustering/regularisation. | **Gradient discontinuities.**  At tooth edges the sub-gradient is ambiguous; high $f$ can make training jittery without careful initialisation or adaptive optimisers. |                                                                                  |                                                                                                                            |
| **Cheap way to boost expressivity without more layers.**                                                            | **Vanishing influence if amplitude $A$ is tiny.**  Needs tuning: $A\approx0.05{-}0.2$ works in quick tests.                                                            |                                                                                  |                                                                                                                            |

Empirically the activation behaves like a *deterministic analogue of dropout*: tiny teeth (say $A=0.05$) usually *help* small models on noisy tabular data; very jagged teeth or large amplitudes slow down convergence.

---

### 5  Drop-in PyTorch snippet

```python
class SawReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, amp=0.1, freq=10.0):
        ctx.save_for_backward(x)
        ctx.amp, ctx.freq = amp, freq
        relu = torch.nn.functional.relu(x)
        saw  = amp * (2 * ((x * freq) % 1.0) - 1.0)
        return relu + saw.clamp(min=0)       # keep teeth only where ReLU is active

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        amp, freq = ctx.amp, ctx.freq
        grad_in  = grad_out.clone()
        mask_pos = (x > 0).float()
        grad_in *= mask_pos * (1 + amp * 2 * freq * (-1) ** torch.floor(x * freq + 0.5))
        return grad_in, None, None
```

*For differentiability* we use sub-gradients $\partial{\sf saw}/\partial z = 2\,{\rm frac}(fz)-1$ everywhere except at the jump points, where any value in \[−1, 1] works.

---

### 6  Practical advice if you want to try it

1. **Start small**: $A\in[0.05,0.15],\;f\in[4,16]$.
   Too coarse ⇒ no effect; too fine ⇒ optimisation noise.
2. **Pair with an adaptive optimiser** (Adam, Ranger) or a slightly lower learning rate for SGD.
3. **Watch for dead neurons**.
   Unlike vanilla ReLU, negative inputs still get zero slope, so initialising biases a bit positive (e.g. 0.1) helps.
4. **Analyse gradients** on a mini-batch; if variance explodes, cut $A$ or $f$.

---

### 7  Bottom line

* A sawtooth-ReLU is **not yet standard**, but:

  * it *has* been studied in theory (depth separation) and in Hopfield-network dynamics;
  * its non-monotonicity lifts **SepOrd** above 1, letting **one neuron do the work of a small threshold circuit**;
  * modest ripples can act as a simple, differentiable regulariser.

If you’re game for experimentation, it’s a low-overhead tweak that might combine some of the representational power of sinusoidal/positional encodings with the robustness and implementation ease of ReLU. Let me know how your “gold-panning” tests go!

[1]: https://arxiv.org/abs/1602.04485?utm_source=chatgpt.com "Benefits of depth in neural networks"
