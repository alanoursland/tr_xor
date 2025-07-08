### 3.2  Loss as an On-Line Proxy for “Distance to Go”

**Goal.** We’d like a stop-rule that triggers *during* training, without access to the final weights.
In practice that means: **detect when the optimiser has entered its last, almost-straight glide to the minimum** (“the final basin”) and estimate how many epochs remain.

---

#### Step 1 Detecting the final basin

A simple empirical cue is *stability of geometry*: once the run is close enough, the three global distances we found—‖Δθ‖₂, ‖Δθ‖₁, and the cosine between successive updates—change only a little each epoch.
A practical heuristic is therefore

```
if (Δ‖θ‖₂_t   < τ₂)  and
   (Δ‖θ‖₁_t   < τ₁)  and
   (1–cosφ_t) < τ_cos        # φ_t is the angle between updates
then “we are in the final basin”.
```

Typical thresholds in our traces were τ₂ ≈ 1 e-3 ‖θ‖₂, τ₁ ≈ 1 e-3 ‖θ‖₁, τ\_cos ≈ 1 e-2.

---

#### Step 2 Within that basin, loss ≈ distance²

Inside a single convex bowl the loss surface is well-approximated by a quadratic:

$$
L(\theta)\;\approx\;
\tfrac12\,(\theta-\theta_\star)^{\!\top}H(\theta-\theta_\star),
$$

with $H$ the local Hessian.
If the curvatures (eigenvalues of $H$) are roughly similar—true in our small models—this simplifies to

$$
L(\theta)\;\approx\;\tfrac{\bar\lambda}{2}\,\lVert\theta-\theta_\star\rVert_2^2,
$$

so

$$
\boxed{\;\lVert\theta-\theta_\star\rVert_2
      \;\propto\;\sqrt{L(\theta)}\;}
\tag{✱}
$$

---

#### Step 3 Plug into the empirical distance–time law

Our traces show, for every model,

$$
\text{epochs}_\text{left}\;\approx\;\alpha\,\lVert\theta-\theta_\star\rVert_2+\beta.
$$

Combine with (✱):

$$
\boxed{\;\text{epochs}_\text{left}
      \;\approx\;
      \underbrace{\bigl(\alpha\sqrt{2/\bar\lambda}\bigr)}_{\text{slope}}
      \sqrt{L(\theta)}
      \;+\;\beta\;} .
$$

In words: **once you have entered the final basin, the remaining training time falls almost linearly with the square-root of the current loss**.

---

#### Step 4 A usable stop rule

1. **Watch for basin entry** with the distance-stability heuristic above.
2. Once inside, maintain a running estimate of the slope
   $\hat{k} = \frac{ \text{epoch}_t-\text{epoch}_{t-\Delta} }
                    { \sqrt{L_{t-\Delta}}-\sqrt{L_t} }$
   over a short window Δ (e.g. 5–10 epochs).
3. Predict $t_{\text{left}} = \hat{k}\,\sqrt{L_t}$.
4. **Stop or decay the learning rate** when $t_{\text{left}}$ drops below a chosen budget.

---

#### Why this works in our experiments

* In **Abs1** and most **ReLU1** runs, the curvature is nearly constant and the update direction scarcely turns, so √loss is an excellent stand-in for distance (R² ≈ 0.99 between √loss and measured epochs-left).
* In **ReLU2** the relationship is noisier—curvatures spread and extra hinge crossings add scatter—but √loss still explains ≈ 80 % of the variance once the basin is reached.

---

*If the quadratic approximation ever fails (e.g. very skewed Hessian or aggressive learning-rate schedules), fall back on the simpler distance-stability heuristic: when ‖Δθ‖ and angle changes stay below fixed thresholds for M consecutive epochs, declare convergence.*
