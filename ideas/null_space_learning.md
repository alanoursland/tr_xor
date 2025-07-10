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
