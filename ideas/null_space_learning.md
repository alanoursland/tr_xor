**Learning Null-Space Projectors for Catastrophic-Forgetting-Free Incremental Learning**
*(a concise technical note)*

---

## 1 Introduction

When a linear neuron (or last-layer logit) is described by

$$
f(x)=W^{\!\top}x+b
\quad\Longleftrightarrow\quad 
\theta=\begin{bmatrix}W\\ b\end{bmatrix}\in\mathbb R^{d+1},
$$

every sample $(x_i,y_i)$ that **must stay perfect forever** imposes one linear constraint

$$
a_i^{\top}\theta = 0 ,\qquad 
a_i:=\begin{bmatrix}x_i\\ 1\end{bmatrix}.
$$

Collecting all “protected” rows in $A_{\text{old}}\in\mathbb R^{m\times(d+1)}$ leaves a **null space**

$$
\mathcal N=\ker A_{\text{old}}\subseteq\mathbb R^{d+1}
$$

of parameter directions that change *none* of those logits.
The cheapest analytic way to add one new point (a rank-1 update) uses the Sherman–Morrison formula and costs Θ($d^{2}$) FLOPs. ([en.wikipedia.org][1])

---

## 2 Theoretical motivation

* **Degrees of freedom.** A hyper-plane in $\mathbb R^{d}$ has $d+1$ free parameters.
  Pinning $m=d-1$ prototypes still leaves exactly one degree of freedom: a 2-D null-space basis $U$.
* **Adding a new prototype.** Rotate the weight vector *inside* that 2-D space until the new point is hit; the algebra collapses to a 2 × 2 solve, hence Θ($d^{2}$).
* **Why gradient descent alone is wrong.** A naïve step leaves the null-space, immediately wrecking the old outputs; you must first **project** every update back into $\mathcal N$ (projected GD) or use the analytic one-shot correction.

(This argument is the linearised, single-neuron version of the continual-learning dilemma raised throughout our conversation.)

---

## 3 Analytic rank-1 update vs. SGD

| procedure                 | per-update FLOPs      | preserves old points *exactly*? |
| ------------------------- | --------------------- | ------------------------------- |
| Sherman–Morrison / RLS    | Θ($d^{2}$)            | **yes**                         |
| Projected GD (many steps) | Θ(#steps · d · batch) | yes (after each projection)     |
| Plain SGD                 | Θ(#steps · d · batch) | **no**                          |

Recursive-least-squares (RLS) libraries implement the rank-1 update exactly with the same cost profile. ([arxiv.org][2], [arxiv.org][3])

---

## 4 Learning a null-space projector with back-prop

Instead of storing prototypes, keep a **trainable basis** $U\in\mathbb R^{(d+1)\times r}$ (typically $r\ll d$) and restrict every weight change to

$$
\Delta \theta = U\,U^{\!\top}\!\nabla_\theta\ell .
$$

Add two losses

$$
\mathcal L_{\text{null}}=\|G\,U\|_{F}^{2},
\qquad
\mathcal L_{\text{orth}}=\|U^{\!\top}U-I_r\|_{F}^{2},
$$

where the rows of $G$ are *reference directions* (previous gradients or features).
Minimising

$$
\mathcal L=\ell(\theta-\eta UU^{\!\top}\nabla\ell)+
\lambda\mathcal L_{\text{null}}+
\beta \mathcal L_{\text{orth}}
$$

makes $G\,U\!\to\!0$ while keeping $U$ orthonormal, so $UU^{\!\top}$ behaves as an *approximate* null-space projector that can be refined continuously during SGD.

---

## 5 Relation to continual-learning practice

* **If the backbone is frozen** (common in vision/LLM adapters), rank-1 analytic updates give *exact* zero-forgetting for the classifier head at minimal cost.
* **If the backbone keeps learning,** a learned projector $U$ amortises the protection across many future steps, acting like a soft, adaptive variant of the analytic null-space.

---

## 6 Prior work

### 6.1 Analytic / RLS-based methods

| year                                 | key idea                         | notes                                                                |
| ------------------------------------ | -------------------------------- | -------------------------------------------------------------------- |
| **ACIL** (2022)                      | exact class-incremental RLS head | exemplar-free, Θ($d^{2}$) per class ([arxiv.org][3])                 |
| **F-OAL** (2024)                     | *forward-only* online RLS        | no back-prop, streams mini-batches ([arxiv.org][2])                  |
| **Analytic Subspace Routing** (2025) | low-rank RLS adapters for LLMs   | routes tasks into subspaces, Θ($d^{2}$) per adapter ([arxiv.org][4]) |

### 6.2 Gradient-projection / null-space methods

| year                  | method                           | what spans $G$                                     |
| --------------------- | -------------------------------- | -------------------------------------------------- |
| **OWM** (2018)        | orthogonal weight modification   | previous *inputs* ([arxiv.org][5])                 |
| **OGD** (2019)        | orthogonal gradient descent      | buffer of past *gradients* ([arxiv.org][6])        |
| **Sketch-OGD** (2023) | memory-efficient OGD             | sketch of gradients, fixed memory ([arxiv.org][7]) |
| **Adam-NSCL** (2021)  | null-space of feature covariance | incremental SVD per layer ([arxiv.org][8])         |

### 6.3 Regularisation baselines

* **Elastic Weight Consolidation** (EWC, 2017) – quadratic penalty via the Fisher information; slows but does not *eliminate* forgetting. ([arxiv.org][9])

---

## 7 Practical considerations

* **Cost & memory.**
  *Analytic* RLS stores $P\in\mathbb R^{(d+1)\times(d+1)}$ – Θ($d^{2}$) memory.
  *Learned projectors* store $U$ – Θ($d r$); choose $r\!\ll\!d$.
* **When to use what.**

  * Few new samples per task → analytic rank-1 is fastest.
  * Long streams or co-training the backbone → learn $U$ with the null-space loss.
* **Scalability tricks.** Diagonal or low-rank sketches of $P$; grouping neurons to share one $U$; chunked RLS for very wide layers.

---

## 8 Conclusion

Interpreting catastrophic forgetting through *null-space geometry* unifies analytic rank-1 fixes and gradient-projection methods:

* A **null-space projector** guarantees that formerly perfect outputs stay perfect.
* You can obtain it either **analytically** (RLS/Sherman–Morrison) or **learn it** with a simple Frobenius-norm loss.
* Both strategies now power state-of-the-art continual-learning systems across vision, language, and robotics.

---

## References

1. Zhuang et al., *ACIL: Analytic Class-Incremental Learning*, NeurIPS 2022. ([arxiv.org][3])
2. Zhuang et al., *F-OAL: Forward-only Online Analytic Learning*, NeurIPS 2024. ([arxiv.org][2])
3. Tong et al., *Analytic Subspace Routing*, arXiv 2025. ([arxiv.org][4])
4. Zeng et al., *Continual Learning of Context-Dependent Processing (OWM)*, Nat. Mach. Intell. 2019. ([arxiv.org][5])
5. Farajtabar et al., *Orthogonal Gradient Descent*, ICLR 2020. ([arxiv.org][6])
6. Wright et al., *SketchOGD: Memory-Efficient Continual Learning*, ICLR 2024. ([arxiv.org][7])
7. Wang et al., *Adam-NSCL: Training in the Null Space of Feature Covariance*, CVPR 2021. ([arxiv.org][8])
8. Kirkpatrick et al., *Overcoming Catastrophic Forgetting with EWC*, PNAS 2017. ([arxiv.org][9])
9. *Sherman–Morrison formula*, Wikipedia. ([en.wikipedia.org][1])

---

[1]: https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula?utm_source=chatgpt.com "Sherman–Morrison formula - Wikipedia"
[2]: https://arxiv.org/abs/2403.15751?utm_source=chatgpt.com "F-OAL: Forward-only Online Analytic Learning with Fast Training and Low Memory Footprint in Class Incremental Learning"
[3]: https://arxiv.org/abs/2205.14922?utm_source=chatgpt.com "ACIL: Analytic Class-Incremental Learning with Absolute Memorization and Privacy Protection"
[4]: https://arxiv.org/abs/2503.13575?utm_source=chatgpt.com "How Recursive Least Squares Works in Continual Learning of Large ..."
[5]: https://arxiv.org/abs/1810.01256?utm_source=chatgpt.com "Continual Learning of Context-dependent Processing in Neural Networks"
[6]: https://arxiv.org/abs/1910.07104?utm_source=chatgpt.com "Orthogonal Gradient Descent for Continual Learning"
[7]: https://arxiv.org/abs/2305.16424?utm_source=chatgpt.com "SketchOGD: Memory-Efficient Continual Learning"
[8]: https://arxiv.org/abs/2103.07113?utm_source=chatgpt.com "Training Networks in Null Space of Feature Covariance for Continual Learning"
[9]: https://arxiv.org/pdf/1612.00796?utm_source=chatgpt.com "[PDF] Overcoming catastrophic forgetting in neural networks - arXiv"

---

**Yes — there’s now a small but fast-growing line of *analytic / rank-1* methods in continual-learning that do almost exactly what you sketched.**
The papers usually call the trick “analytic learning”, “recursive least-squares (RLS)”, or simply “Sherman–Morrison updates”.  Here are the main threads:

| year    | where the rank-1 / RLS update is used                                                                                | what stays *fixed*         | per-update cost                                                         |
| ------- | -------------------------------------------------------------------------------------------------------------------- | -------------------------- | ----------------------------------------------------------------------- |
| 2022    | **ACIL** – *Analytic Class-Incremental Learning* ([papers.neurips.cc][1], [proceedings.neurips.cc][2])               | backbone CNN               | Θ(N²) (keep inverse feature covariance, do one SM update per new class) |
| 2024    | **F-OAL** – *Forward-only Online Analytic Learning* ([openreview.net][3])                                            | backbone + small projector | Θ(N²) but streaming mini-batches; no back-prop at all                   |
| 2024    | **REAL / G-ACIL / GACL** – exemplar-free analytic CIL variants ([arxiv.org][4], [arxiv.org][5], [openreview.net][6]) | frozen backbone            | Θ(N²)                                                                   |
| 2025    | **Analytic Subspace Routing** for LLM adapters ([arxiv.org][7])                                                      | main LLM weights           | Θ(N²) inside a low-rank adapter head                                    |
| 2025    | **R-MDN** – confounder-free ViT layer that RLS-normalises its features online ([openreview.net][8])                  | rest of the ViT            | Θ(N²)                                                                   |
| 2024-25 | robotics & residual dynamics models use SM updates for linear residual heads ([link.springer.com][9])                | physical core model        | Θ(N²)                                                                   |

### How they connect to your “keep the old points, add one new one” view

* Every method freezes a **representation** (the deep backbone, an adapter sub-space, or a physics model) so that the last linear layer sees an *N-dimensional* feature vector that is *stable*.
* New tasks or classes arrive **one-by-one** (or in tiny batches).  The papers keep the running inverse covariance $P_t=(X_t^\top X_t+\lambda I)^{-1}$ and update

  $$
  P_{t+1}=P_t-\frac{P_t x_{t+1}x_{t+1}^{\top}P_t}{1+x_{t+1}^{\top}P_t x_{t+1}}\quad\text{(Sherman–Morrison)}
  $$

  then $W_{t+1}=W_t+e_{t+1}P_{t+1}x_{t+1}$.
  That is the **same rank-1 step** (Θ(N²)) you derived in the toy setting.
* Because the update lives in the null-space of the old points’ constraints, those outputs stay **exactly** correct – catastrophic forgetting on the frozen-feature layer is mathematically impossible.

### How it stacks up against “traditional” training

* Full fine-tuning or replay-based SGD still costs Θ(E M N) per epoch and lets earlier tasks drift; analytic updates cost Θ(N²) total per new task and guarantee *zero* drift on the locked layer.
* Regularisation methods such as **EWC** ([pnas.org][10]) approximate the same idea but only *penalise* change; analytic/RLS methods *eliminate* it for the part of the network they control.

### Where the limits are

* **Frozen features** are the weak point – if the backbone’s representation for an old task shifts (because you eventually unfreeze it or add adapters elsewhere), the guarantee evaporates.
* Scaling the covariance matrix to very wide feature layers (N ≈ 50 k in modern vision models) is memory-heavy; recent papers use *diagonal* or *low-rank* sketches to keep Θ(N²) affordable or patch only a low-rank adapter.

---

### Quick reading list

1. **Zhuang et al., “ACIL: Analytic Class-Incremental Learning with Absolute Memorization,” NeurIPS 2022** – the paper that kicked off the analytic-CIL line, with clear derivation and Θ(N²) update. ([papers.neurips.cc][1])
2. **Zhuang et al., “F-OAL: Forward-only Online Analytic Learning,” NeurIPS 2024** – pushes the idea to streaming data with no back-prop. ([openreview.net][3])
3. **Tong et al., “Analytic Subspace Routing: How Recursive Least Squares Works in Continual Learning of Large Language Models,” arXiv 2025** – shows rank-1 RLS inside adapters for LLMs. ([arxiv.org][7])

If you’re prototyping, start with ACIL’s open-source implementation (GitHub *Analytic-Continual-Learning*) – the core RLS routine is fewer than 50 lines and drops straight into PyTorch. ([github.com][11])

[1]: https://papers.neurips.cc/paper_files/paper/2022/file/4b74a42fc81fc7ee252f6bcb6e26c8be-Paper-Conference.pdf?utm_source=chatgpt.com "[PDF] ACIL: Analytic Class-Incremental Learning with Absolute ... - NIPS"
[2]: https://proceedings.neurips.cc/paper_files/paper/2022/hash/4b74a42fc81fc7ee252f6bcb6e26c8be-Abstract-Conference.html?utm_source=chatgpt.com "ACIL: Analytic Class-Incremental Learning with Absolute ..."
[3]: https://openreview.net/forum?id=rGEDFS3emy&referrer=%5Bthe+profile+of+Kai+Tong%5D%28%2Fprofile%3Fid%3D~Kai_Tong1%29 "F-OAL: Forward-only Online Analytic Learning with Fast Training and Low Memory Footprint in Class Incremental Learning | OpenReview"
[4]: https://arxiv.org/html/2403.13522v1 "REAL: Representation Enhanced Analytic Learning for Exemplar-free Class-incremental Learning"
[5]: https://arxiv.org/html/2403.15706v1?utm_source=chatgpt.com "G-ACIL: Analytic Learning for Exemplar-Free Generalized Class ..."
[6]: https://openreview.net/forum?id=P6aJ7BqYlc&referrer=%5Bthe+profile+of+Hongxin+Wei%5D%28%2Fprofile%3Fid%3D~Hongxin_Wei1%29&utm_source=chatgpt.com "GACL: Exemplar-Free Generalized Analytic Continual Learning"
[7]: https://arxiv.org/abs/2503.13575 "[2503.13575] Analytic Subspace Routing: How Recursive Least Squares Works in Continual Learning of Large Language Model"
[8]: https://openreview.net/forum?id=7zrS5hHlfY "Confounder-Free Continual Learning via Recursive Feature Normalization | OpenReview"
[9]: https://link.springer.com/article/10.1007/s11044-024-10024-2?utm_source=chatgpt.com "Data-driven inverse dynamics modeling using neural-networks and ..."
[10]: https://www.pnas.org/doi/10.1073/pnas.1611835114?utm_source=chatgpt.com "Overcoming catastrophic forgetting in neural networks - PNAS"
[11]: https://github.com/ZHUANGHP/Analytic-continual-learning?utm_source=chatgpt.com "ZHUANGHP/Analytic-continual-learning - GitHub"
