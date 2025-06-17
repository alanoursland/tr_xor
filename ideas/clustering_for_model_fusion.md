# 🧠 Clustering-Based Model Construction from Small Network Ensembles

## 1. Conceptual Overview

Neural networks are often trained from scratch using random initializations, despite the fact that many independently trained models rediscover similar internal features. In this work, we propose a **layerwise clustering method** to extract and reuse **shared neuron-level structures** from many small models. These **recurring hyperplanes**—i.e., neurons with similar weights across models—are treated as functional subunits. We use their **clusters** to guide the **construction of larger models** with aligned, interpretable, and reusable internal representations.

Unlike traditional model averaging or ensembling, our method treats **individual hyperplanes as the unit of learning**, enabling compositional reuse across architectures. This procedure effectively bootstraps large model structure from many smaller, fast-to-train networks, providing a scalable method for structural distillation and initialization.

---

## 2. Method: Bottom-Up Network Synthesis via Layerwise Hyperplane Clustering

### 2.1 Step 1: Train a Population of Small Models

We train a set of small neural networks (e.g., small MLPs, CNNs, or transformers) independently on the same task. Each model provides a complete set of learned hyperplanes (neurons) at each layer. These models may be trained on different data subsets or the full dataset.

Let each model $M^{(k)}$ consist of layers $L_1^{(k)}, L_2^{(k)}, \dots, L_n^{(k)}$, with weight matrices $W_l^{(k)} \in \mathbb{R}^{d_l \times d_{l-1}}$ for layer $l$.

---

### 2.2 Step 2: Cluster Hyperplanes per Layer

We extract all individual neuron weight vectors (rows of each $W_l^{(k)}$) from every model and cluster them to find **recurring neuron types**.

Let $\mathcal{C}_l = \{c_{l,1}, \dots, c_{l,K_l}\}$ be the clusters for layer $l$.

For early layers (e.g., convolutional or first MLP layers), clustering can be done directly in weight space using cosine similarity or L2 distance. For deeper layers, see below.

---

### 2.3 Step 3: Cluster Middle Layers **Relative to Lower-Layer Structure**

Neurons in middle layers compute functions **conditioned on the outputs of earlier layers**. Thus, their meaning depends on **which basis they receive input from**.

To normalize this across models:

* First, cluster layer $l-1$ and fix a canonical ordering of its cluster centroids.
* For each neuron in layer $l$, project its incoming weight vector onto the **cluster basis** of $l-1$ (i.e., in the space of centroid activations, not raw input).
* Cluster in this normalized space.

This ensures that neurons are grouped based on **functional behavior relative to a shared input basis**, not arbitrary neuron orderings.

---

### 2.4 Step 4: Normalize Neuron Order via Cluster Frequency

To enable cross-model weight merging or statistical aggregation:

* Reorder neurons in each layer according to the **frequency of their cluster** (i.e., how often a given cluster appears across models).
* This provides a consistent neuron index space, aligning neurons by semantic role rather than training artifacts.

---

### 2.5 Step 5: Assemble a Larger Model from Cluster Centroids

Using the bank of clusters $\mathcal{C}_l$ for each layer $l$, we construct a larger model with:

* A greater number of neurons per layer, selected from the **most frequent and stable clusters**.
* Weights initialized to the **cluster centroids**.
* Layer connectivity preserved via consistent cluster ordering.

This results in a **structured, compositional network** whose layers are made from **statistically common, functionally aligned subunits** distilled from the small model population.

---

## 3. Architectural and Theoretical Implications

### 3.1 Hyperplanes as the Unit of Knowledge

Each neuron defines a hyperplane, partitioning its input space. By clustering these, we identify **which partitions consistently recur**—these are the true atomic "features" across models.

### 3.2 Emergence of Neural Primitives

The clustering procedure uncovers **emergent neural motifs**: commonly used transformations that span across architectures. These motifs can be reused, weighted, or modified, creating a **vocabulary of features** that supports flexible architecture design.

### 3.3 Representation Grounding

Middle-layer clustering grounded on lower-layer clusters provides a consistent basis for deeper representations. This allows semantically aligned representations across networks, making transfer and merging more viable.

---

## 4. Advantages

* **Parallelizable**: Training many small models is cheap and parallel.
* **Composable**: Clusters can be mixed, pruned, or reused across tasks.
* **Interpretability**: Neurons acquire identities based on cluster semantics.
* **Scalable**: Larger models can be grown by aggregating learned structure.

---

## 5. Extensions

* **Hierarchical clustering**: Identify multi-layer motifs or subnetworks, not just individual neurons.
* **Cluster routing**: Train lightweight routers that dynamically weight or select clusters based on input.
* **Transformers**: Apply clustering to attention heads, MLP projections, and normalization patterns.
* **Cross-task generalization**: Use cluster frequency across multiple tasks to identify domain-invariant structure.

---

## 6. Summary

This method offers a principled framework for building large neural networks from reusable parts. By mining, clustering, and composing neuron-level structures from small models, we generate models with grounded representations and improved initialization. The result is a **bottom-up construction method** where the building blocks are **hyperplanes**, not just parameters—offering a novel paradigm for both interpretability and architecture design.
