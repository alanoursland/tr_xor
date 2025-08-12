# Energy-Conserving Optimizer with Spin: A Physics-Inspired Optimization Framework

## Overview

This document presents a novel discrete-time optimizer inspired by physical dynamics. The system models a particle moving on a loss surface under gravity-like forces, governed by energy conservation principles, and extended with angular "spin" dynamics to penalize curvature in the optimization path.

This optimizer integrates:

Potential energy from a scalar loss function.
Translational kinetic energy from particle motion.
Angular kinetic energy (spin) from changes in motion direction.
Damping of both linear and angular velocity to ensure convergence.
The goal is to provide an intuitive, physically motivated optimizer that naturally balances exploration and convergence, with interpretable dynamics.

---

## Core Concepts

### Particle Dynamics

Position:                      \( \mathbf{x}_k \in \mathbb{R}^n \)
Velocity:                      \( \mathbf{v}_k \in \mathbb{R}^n \)
Loss surface (height):          \( f(\mathbf{x}) \)
Gradient:                      \( \nabla f(\mathbf{x}) \)
The particle moves under the influence of a “gravity” force proportional to the negative gradient.

---

### Energy Terms

Potential energy:   \( U_k = f(\mathbf{x}_k) \)
Kinetic energy:    \( K_k = \frac{1}{2} \|\mathbf{v}_k\|^2 \)
Angular (spin) energy: \( K_k^{\text{ang}} = \frac{1}{2} I \|\mathbf{s}_k\|^2 \)
Total energy:     \( E_k = U_k + K_k + K_k^{\text{ang}} \)
The system evolves while approximately conserving total energy, except when damping is applied.

---

## System Parameters

\( g \): gravitational coefficient (gradient scaling)
\( \Delta t \): time step size
\( I \): moment of inertia (for spin energy)
\( \gamma \): linear damping coefficient
\( \beta \): angular (spin) damping coefficient
---

## Update Equations (Per Time Step)

### 1. Compute Gradient and Acceleration

\[
\mathbf{a}_k = -g \nabla f(\mathbf{x}_k)
\]

---

### 2. Update Position (Kinematic Equation)

\[
\mathbf{x}_{k+1} = \mathbf{x}_k + \mathbf{v}_k \Delta t + \frac{1}{2} \mathbf{a}_k (\Delta t)^2
\]

---

### 3. Displacement Direction

\[
\hat{\mathbf{d}}{k+1} = \frac{\mathbf{x}{k+1} - \mathbf{x}k}{\|\mathbf{x}{k+1} - \mathbf{x}_k\|}
\]

---

### 4. Spin Update

Let \( \hat{\mathbf{d}}_k \) be the previous direction. Then:

#### In 2D:

\[
\Delta s = \frac{ \hat{\mathbf{d}}k \times \hat{\mathbf{d}}{k+1} }{ \Delta t }
\quad (\text{scalar spin})
\]

#### In 3D:

\[
\Delta \mathbf{s} = \frac{ \hat{\mathbf{d}}k \times \hat{\mathbf{d}}{k+1} }{ \Delta t }
\quad (\text{vector spin})
\]

\[
\mathbf{s}_{k+1,\text{pre}} = \mathbf{s}k + \Delta \mathbf{s}
\]
\[
\mathbf{s}{k+1} = \mathbf{s}_{k+1,\text{pre}} \cdot (1 - \beta \Delta t)
\]

---

### 5. Update Energies

#### a. Angular Kinetic Energy

\[
K{k+1}^{\text{ang}} = \frac{1}{2} I \|\mathbf{s}{k+1}\|^2
\]

#### b. Potential Energy

\[
U{k+1} = f(\mathbf{x}{k+1})
\]

#### c. Translational Kinetic Energy (Pre-Damping)

\[
K_{k+1,\text{pre}} = Ek - U{k+1} - K_{k+1}^{\text{ang}}
\]

If \( K_{k+1,\text{pre}} < 0 \), the step is invalid and must be corrected (e.g., by reducing step size).

#### d. Apply Linear Damping

\[
K{k+1} = K{k+1,\text{pre}} \cdot (1 - \gamma \Delta t)
\]

---

### 6. Update Velocity

\[
\|\mathbf{v}{k+1}\| = \sqrt{2 K{k+1}}
\]
\[
\mathbf{v}{k+1} = \|\mathbf{v}{k+1}\| \cdot \hat{\mathbf{d}}_{k+1}
\]

---

## Initialization

Set initial position \( \mathbf{x}_0 \)
Set initial velocity \( \mathbf{v}_0 \) (often zero)
Set initial spin \( \mathbf{s}_0 = 0 \)
Compute \( U_0 = f(\mathbf{x}_0) \), \( K_0 = \frac{1}{2}\|\mathbf{v}_0\|^2 \), \( K_0^{\text{ang}} = 0 \)
Total energy: \( E_0 = U_0 + K_0 + K_0^{\text{ang}} \)
---

## Interpretation

Spin stores curvature: Turning “charges up” spin, reducing forward kinetic energy.
Straight motion conserves spin: Energy stays in forward motion unless the path bends.
Damping causes convergence: Linear and angular damping gradually dissipate energy, allowing the optimizer to settle in minima.
Lyapunov-like behavior: Though not strictly decreasing, total energy + damping ensures system convergence to stable equilibria.
---

## Potential Applications

Gradient-based optimization in machine learning
Alternative to SGD or Adam for smoother, physically interpretable dynamics
Toy examples and smooth losses, especially where convergence and path geometry matter
---

## Next Steps

Prototype on simple 2D loss surfaces
Compare trajectories against SGD, momentum, and Adam
Explore convergence rates and sensitivity to parameters \( g, I, \gamma, \beta \)
Visualize spin energy and how it modulates forward progress
---

## Credits

Inspired and developed from first principles as a physics analog for optimization. Novel ideas include:

Treating spin as an energy-reservoir for curvature
Using energy conservation + damping as a convergence mechanism
Explicitly coupling geometry (turning) to optimization dynamics

---

## Updates and Commentary

Here’s a version that keeps your **energy budget** idea, but makes **spin scalar and per-parameter** (no vectors, no cross products). It behaves like: if a parameter’s update direction keeps flipping (or wiggling), its spin builds up and steals energy from its step; if the direction stays steady, spin decays and returns energy.

# Sketch of the update

* You still use any base optimizer to propose a raw step $\Delta\theta^{\text{raw}}$ (SGD, momentum, Adam—whatever).
* Maintain, per parameter $i$:

  * $s_i \ge 0$: scalar “spin”
  * $\Delta\theta^{\text{prev}}_i$: last step value
* Global scalars:

  * $C_t$: total energy cap (decays slowly)
  * $U_t$: potential = current loss value (just use the scalar loss)
* Cheap **turn signal** per parameter:

  $$
  \text{turn}_i = \frac{|\Delta\theta^{\text{raw}}_i - \Delta\theta^{\text{prev}}_i|}{|\Delta\theta^{\text{raw}}_i| + |\Delta\theta^{\text{prev}}_i| + \varepsilon}
  $$

  (≈ 0 when direction/magnitude steady; ≈ 1 on sharp reversals; smoother than a hard sign flip.)
* Spin update (directionless):

  $$
  s_i \leftarrow (1-\beta)\, s_i + \kappa \cdot \text{turn}_i
  $$
* Angular energy per param:

  $$
  K^{\text{ang}}_i = \tfrac{1}{2} I\, s_i^2
  $$
* Raw linear kinetic energy (before scaling):

  $$
  K^{\text{lin}}_{\text{pre}} = \tfrac{1}{2}\sum_i (\Delta\theta^{\text{raw}}_i)^2
  $$
* Global budget (cap minus current potential minus spin):

  $$
  B_t = \max\big(0,\; C_t - U_t - \sum_i K^{\text{ang}}_i\big)
  $$
* Compute a single **global scale** $\alpha \in [0,1]$ applied to all params:

  $$
  \alpha = 
  \begin{cases}
  1, & K^{\text{lin}}_{\text{pre}} \le B_t\\[2mm]
  \sqrt{\dfrac{B_t}{K^{\text{lin}}_{\text{pre}} + \varepsilon}}, & \text{otherwise}
  \end{cases}
  $$
* Final step:

  $$
  \Delta\theta_i = \alpha \cdot \Delta\theta^{\text{raw}}_i
  $$
* Bookkeeping:

  * $\Delta\theta^{\text{prev}}_i \leftarrow \Delta\theta_i$
  * $C_{t+1} \leftarrow (1-\gamma)\, C_t$ (slow global decay)
  * Optionally let **spin decay faster** than the cap: $\beta \gg \gamma$

# Why this matches your intent

* **Directionless spin:** it only cares about *change* in update direction/magnitude; it’s cheap and per-parameter.
* **Energy transfer:** big turns ⇒ $s_i$ grows ⇒ $K^{\text{ang}}_i$ grows ⇒ less budget for linear motion ⇒ global scale $\alpha$ shrinks; steady direction ⇒ spin decays ⇒ budget returns to linear motion.
* **Global coupling stays:** although spin is per-parameter, the **cap is global**, so a few wiggly coordinates can slow the whole system (your “one object in parameter space” vibe), but the accounting is simple.
* **Drop-in:** wrap any optimizer; just rescale its step and keep tiny buffers.

# Pseudo-PyTorch hook

```python
class SpinEnergyWrapper(torch.optim.Optimizer):
    def __init__(self, base_opt, I=1.0, kappa=0.1, beta=0.05, gamma=1e-3,
                 C0=1.0, eps=1e-12):
        self.base_opt = base_opt
        self.I = I
        self.kappa = kappa
        self.beta = beta      # spin decay (faster)
        self.gamma = gamma    # cap decay (slower)
        self.C = C0           # energy cap
        self.eps = eps
        # per-param state: s (spin), prev_step
        self.state = {}

    @torch.no_grad()
    def step(self, loss_value):
        # 1) Let base optimizer compute raw step: stash .grad, do not apply yet
        # Instead, compute "raw step" as what the base optimizer WOULD do.
        raw_steps = []
        params = []
        for group in self.base_opt.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                params.append(p)
                # Approximate raw step as -lr * grad or pull from momentum buffers if available
                lr = group.get('lr', 1e-3)
                raw = -lr * p.grad
                raw_steps.append(raw.clone())

        # 2) Spin update per parameter & accumulate energies
        K_ang = 0.0
        K_lin_pre = 0.0
        turns = []
        for p, raw in zip(params, raw_steps):
            st = self.state.setdefault(p, {})
            prev = st.get('prev_step', torch.zeros_like(raw))
            s = st.get('spin', torch.zeros_like(raw))  # per-element spin (or use scalar: one value per tensor)

            turn = (raw - prev).abs() / (raw.abs() + prev.abs() + self.eps)
            s = (1 - self.beta) * s + self.kappa * turn

            # Energies (sum over all elements)
            K_ang += 0.5 * self.I * (s * s).sum().item()
            K_lin_pre += 0.5 * (raw * raw).sum().item()

            st['spin'] = s
            st['prev_step'] = prev  # will update after scaling
            turns.append(turn)

        # 3) Budget and global scale alpha
        U = float(loss_value.detach().item())
        B = max(0.0, self.C - U - K_ang)
        if K_lin_pre <= B or K_lin_pre == 0.0:
            alpha = 1.0
        else:
            alpha = (B / (K_lin_pre + self.eps)) ** 0.5

        # 4) Apply scaled step, update prev_step, decay cap
        idx = 0
        for group in self.base_opt.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                raw = raw_steps[idx]
                st = self.state[p]
                step_final = alpha * raw
                p.add_(step_final)  # apply
                st['prev_step'] = step_final.clone()
                idx += 1

        self.C *= (1.0 - self.gamma)
```

**Notes**

* I showed **per-element spin** (most granular, still cheap). If you want even cheaper, keep **one scalar spin per tensor** by replacing `s` with a single value (e.g., take mean turn per tensor).
* You can wrap **Adam** as the base optimizer by letting it update its internal moments but intercepting the actual step (use `torch.no_grad()`: compute Adam’s *proposed* update vector, then rescale by $\alpha$ before applying).
* Stability knobs:

  * Lower `kappa` if steps shrink too much on noisy gradients.
  * Raise `beta` to make spin forget faster.
  * Set `C0` relative to initial loss (e.g., `C0 = c * loss.item()`, `c≈2–5`) and decay `gamma` gently (e.g., `1e-4–1e-3`).

This gives you the **feel** of your physics model (energy budget + spin) with **Adam-like cost**. If you want, I can tailor this wrapper to your CIFAR code so we can A/B it against Adam on a small run.
