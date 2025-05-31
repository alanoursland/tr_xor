# **Experiment Results: Single Absolute Value Unit for XOR Classification (abs1)**

## **1. Introduction**

This document presents the results of the `abs1` experiment, which investigated the capability of a minimal neural network—a single neuron with an absolute value-based activation function—to solve the XOR problem. The detailed experimental setup, model architecture (including the clamping of the output to $[0,1]$), and objectives are described in `abs1_overview.md`. The theoretical underpinnings, including the "Prototype Surface Learning Theory" and "Separation Order Theory," can be found in `core_theory.md` and `separation_order_theory.md`, respectively.

This report summarizes the performance of this single-unit model when trained on the XOR dataset using various weight initialization strategies. The primary goals were to validate theoretical predictions about the model's capability, analyze the impact of initialization, characterize failure modes, and gather evidence for the prototype surface learning mechanism. All experiments were conducted over 500 independent training runs per initialization strategy.

---

## **2. Overall Performance Summary**

The experiment demonstrates that a single neuron equipped with an absolute value activation function **can indeed solve the XOR problem**. However, the success and reliability of convergence are **highly dependent on the weight initialization strategy** employed, largely due to the behavior of the clamping operation in the model's output stage.

Across the five tested initialization strategies, performance varied significantly. Smaller initial weight magnitudes (specifically, the "Tiny" initialization with $\sigma=0.1$) led to perfect convergence and 100% accuracy across all runs. As the variance of the initial weights increased, the convergence rate and the proportion of perfectly accurate models generally decreased. This highlights the critical role of initialization in enabling this minimal architecture to find a valid solution, primarily by avoiding pre-activation values that lead to zero gradients due to clamping.

---

## **3. Detailed Results per Initialization Strategy**

The performance metrics for each of the five weight initialization strategies, tested over 500 independent training runs, are summarized in the table below. This data includes the convergence rate (percentage of runs achieving a final Mean Squared Error loss < 0.01), perfect accuracy rate (100% classification accuracy, assuming output < 0.5 is class 0 and $\ge$ 0.5 is class 1), the distribution of final accuracies, average final loss, the minimum and maximum final losses observed across all runs, and the average training time per run.

| Metric                          | Tiny (σ=0.1)                             | Normal (σ=0.5)                           | Xavier (σ ≈ 0.707)                         | Kaiming (σ=1.0)                          | Large (σ=4.0)                            |
| :------------------------------ | :--------------------------------------- | :--------------------------------------- | :----------------------------------------- | :--------------------------------------- | :--------------------------------------- |
| **Experiment Name**             | `abs1_tiny`                              | `abs1_normal`                            | `abs1_xavier`                              | `abs1_kaiming`                           | `abs1_large`                             |
| **Perfect Accuracy Rate (%)**   | 100%                                     | 80.4%                                    | 63.0%                                      | 53.4%                                    | 12.6%                                    |
| **Accuracy Distribution (runs)**|                                          |                                          |                                            |                                          |                                          |
| 100% Accurate                   | 500                                      | 402                                      | 315                                        | 267                                      | 63                                       |
| 75% Accurate                    | 0                                        | 0                                        | 0                                          | 0                                        | 0                                        |
| 50% Accurate                    | 0                                        | 98                                       | 185                                        | 233                                      | 437                                      |
| 25% Accurate                    | 0                                        | 0                                        | 0                                          | 0                                        | 0                                        |
| 0% Accurate                     | 0                                        | 0                                        | 0                                          | 0                                        | 0                                        |
| **Avg. Final Loss**             | 3.41 × 10⁻¹⁰                              | 0.0980                                   | 0.1850                                     | 0.2330                                   | 0.4370                                   |
| **Min. Final Loss**             | 0.0                                      | 0.0                                      | 0.0                                        | 1.78 × 10⁻¹⁵                              | 0.0                                      |
| **Max. Final Loss**             | 3.07 × 10⁻⁹                               | 0.5                                      | 0.5                                        | 0.5                                      | 0.5                                      |
| **Avg. Training Time (s)**      | 0.194                                    | 0.195                                    | 0.196                                      | 0.199                                    | 0.194                                    |

*(Note: Total runs for each strategy = 500. "Perfect Accuracy Rate" refers to runs achieving 100% classification accuracy on the 4 XOR patterns. Min/Max Final Loss are the best and worst final losses observed across all 500 runs for that strategy.)*

---

## **4. Comparative Analysis of Initialization Strategies**

The choice of weight initialization strategy had a pronounced effect on the model's ability to solve the XOR problem. The table below summarizes key performance metrics across the different strategies:

| Initialization Strategy | Std. Dev. ($\sigma$) | Convergence Rate (Loss < 0.01) | Perfect Accuracy Rate (100%) | Average Final Loss |
| :---------------------- | :------------------: | :----------------------------: | :--------------------------: | :----------------: |
| Tiny                    | 0.1                  | 100%        | 100%      | $3.41 \times 10^{-10}$ |
| Normal                  | 0.5                  | 80.4%     | 80.4%   | 0.0980 |
| Xavier                  | Approx. 0.707        | 63.0%     | 63.0%   | 0.1850 |
| Kaiming                 | Approx. 1.0          | 53.4%    | 53.4%  | 0.2330 |
| Large                   | 4.0                  | 12.6%      | 12.6%    | 0.4370 |

*(Note: Standard deviation for Xavier is $\sqrt{1/N_{in}} = \sqrt{1/2} \approx 0.707$. Standard deviation for Kaiming is $\sqrt{2/N_{in}} = \sqrt{2/2} = 1.0$. $N_{in}=2$ for this model.)*

**Observations and Explanation of Performance Differences:**

* There is a clear trend: **smaller initial weight standard deviations lead to better performance**. The Tiny initialization ($\sigma=0.1$) achieved perfect results.
* As the initial weight magnitudes increase (Normal $\rightarrow$ Xavier $\rightarrow$ Kaiming $\rightarrow$ Large), both the convergence rate and the perfect accuracy rate steadily decline.
* The Large initialization ($\sigma=4.0$) performed very poorly, with only 12.6% of runs converging to a perfect solution.

The superior performance of smaller initial weights is a direct consequence of the **clamping operation** $y_{final} = \text{clamp}(|L(x_1, x_2)|, 0.0, 1.0)$ used in the model. The effective activation function $g(L(X)) = \text{min}(\text{abs}(L(X)), 1.0)$ has a derivative of zero with respect to $L(X)$ whenever $\text{abs}(L(X)) \ge 1$.
Let $L(X; w_1, w_2) = w_1 x_1 + w_2 x_2$ (since bias $b$ is initialized to zero).
For the XOR problem with centered inputs `[(-1,-1), (1,-1), (-1,1), (1,1)]` and targets `[0, 1, 1, 0]`:
* Inputs `(-1,-1)` and `(1,1)` are target 0. For these, we need $L(X) \approx 0$. Specifically, for `(1,1)`, we need $w_1+w_2 \approx 0$. If, due to initialization, $\text{abs}(w_1+w_2) \ge 1$, the model will output 1 (due to $\text{abs}(\cdot)$ then clamp), but the gradient for this crucial input will be zero. This prevents the weights from being adjusted to make their sum closer to zero, effectively stalling learning for these cases.
* Weights $w_1, w_2$ are drawn independently from $\mathcal{N}(0, \sigma^2)$, so their sum $w_1+w_2$ is drawn from $\mathcal{N}(0, 2\sigma^2)$.
    * **Tiny init ($\sigma=0.1$):** $\text{std}(w_1+w_2) \approx 0.141$. It is virtually guaranteed that $\text{abs}(w_1+w_2) < 1$ initially, allowing gradients to flow for all XOR conditions. This enables the model to learn $w_1+w_2 \approx 0$ for target 0 inputs, and appropriate magnitudes for $w_1, w_2$ for target 1 inputs.
    * **Normal init ($\sigma=0.5$):** $\text{std}(w_1+w_2) \approx 0.707$. The probability of $\text{abs}(w_1+w_2) \ge 1$ is non-negligible (around 16%), leading to a reduced success rate compared to Tiny init.
    * **Xavier ($\sigma \approx 0.816$) and Kaiming ($\sigma=1.0$):** $\text{std}(w_1+w_2)$ is $\approx 1.15$ and $\approx 1.414$ respectively. These have a higher probability of $\text{abs}(w_1+w_2) \ge 1$ initially, further increasing the risk of hitting the zero-gradient region for the target 0 cases.
    * **Large init ($\sigma=4.0$):** $\text{std}(w_1+w_2) \approx 5.6$. $\text{abs}(w_1+w_2)$ will almost certainly be $\ge 1$, making it very difficult for the model to learn the $w_1+w_2 \approx 0$ condition.

Thus, the observed results are an artifact of the clamping operation. Smaller initial weights ensure that the pre-clamped absolute values remain in a region where gradients are non-zero, allowing the learning dynamics to adjust weights appropriately for all XOR input patterns.

---

## **5. Discussion of Experimental Objectives**

### **5.1. Objective 1: Validate Theoretical Prediction**
The experiment aimed to demonstrate that a single unit with an absolute value activation can reliably solve XOR. The results, particularly with the **Tiny initialization strategy ($\sigma=0.1$) achieving 100% convergence and 100% perfect accuracy across all 500 runs, strongly validate this theoretical prediction.** This aligns with the "Separation Order Theory," which posits that an activation function with Separation Order $k$ can implement a predicate of Minsky-Papert order $k$ within a single unit. Since the absolute value function has $\text{SepOrd}(|\cdot|) = 2$ and XOR has a Minsky-Papert order of 2, this capability is expected.

### **5.2. Objective 2: Analyze Initialization Impact**
A key objective was to quantify how different weight initialization strategies affect convergence and accuracy. As detailed in Section 4, the impact is substantial and primarily driven by the interaction of initial weight magnitudes with the output clamping mechanism.
* **Tiny initialization** ($\sigma=0.1$) was optimal, ensuring that initial pre-clamped values $\text{abs}(L(X))$ were likely less than 1, thus preserving gradient flow for all XOR conditions, especially the critical $w_1+w_2 \approx 0$ requirement for target 0 inputs.
* **Larger initializations** (Normal, Xavier, Kaiming, and especially Large) increasingly risked that $\text{abs}(L(X)) \ge 1$ for inputs that should target 0 (like `(1,1)` or `(-1,-1)`), leading to zero gradients for these cases due to clamping and preventing the model from learning the correct weight configurations. This explains the observed decline in performance as $\sigma$ increased.

### **5.3. Objective 3: Characterize Failure Modes**
For runs that did not achieve 100% accuracy, the predominant outcome was **50% accuracy**. For the XOR problem, 50% accuracy means the model correctly classifies exactly two of the four patterns.
This failure mode is strongly linked to the **clamping operation and the resultant zero-gradient issue** discussed in Section 4 and 5.2. If initial weights are too large, causing $\text{abs}(w_1x_1+w_2x_2+b) \ge 1$ for an input where the target is 0 (e.g., for `(1,1)` or `(-1,-1)` where $b=0$ and ideally $w_1+w_2 \approx 0$), the model outputs 1 but receives no gradient to correct the weights for that input pattern. The model might still learn to satisfy the target 1 conditions (e.g., making $\text{abs}(w_1-w_2)$ large for inputs `(1,-1)` or `(-1,1)`), but fails on one pair of target 0 inputs, leading to 50% accuracy. This suggests the model gets stuck in a local minimum where the clamp prevents learning for a subset of the input patterns.

### **5.4. Objective 4: Validate Prototype Surface Theory**
This objective aimed to verify that for successfully trained models, data points for class 0 (e.g., `(-1,-1)` and `(1,1)`) lie on or near the learned prototype surface $L(x_1,x_2) = w_1x_1 + w_2x_2 + b = 0$, while points for class 1 (e.g., `(1,-1)` and `(-1,1)`) are characterized by $|L(x_1,x_2)| \ge c \approx 1.0$ (before clamping).
The summary statistics provided in the JSON files do not include the learned weights ($w_1, w_2, b$) or the specific $L(x_i)$ values for each input pattern from the individual runs. Therefore, a **full quantitative validation of the Prototype Surface Theory requires further analysis** of the saved model parameters from successful runs. The current results (showing the model *can* solve XOR) are consistent with the theory's predictions, but direct evidence of the learned surface properties is pending.

---

## **6. Conclusion**

The `abs1` experiment successfully demonstrated that a single neuron employing an absolute value activation function is capable of solving the XOR problem. This empirical result supports the theoretical framework of Separation Order, which predicts this capability due to the absolute value function having a Separation Order of 2.

The experiment unequivocally highlighted the **critical importance of weight initialization**, with the observed performance differences being a direct artifact of the **clamping operation** in the model's output stage. Initializing weights with a small standard deviation ("Tiny" init, $\sigma=0.1$) led to 100% success. This is because such an initialization keeps the initial pre-clamped absolute values ($\text{abs}(L(X))$) in a region ($<1$) where gradients can flow for all XOR conditions, particularly for learning that $w_1+w_2 \approx 0$ for target 0 inputs. Larger initial weight magnitudes progressively reduced the success rate, as they increased the likelihood of $\text{abs}(L(X)) \ge 1$ for target 0 inputs, leading to zero gradients due to clamping and stalling learning for those critical cases.

Common failure modes resulted in 50% accuracy, likely due to this clamping-induced gradient issue for specific input patterns. Further analysis of learned weights from successful runs is necessary to fully validate the specific mechanisms of the Prototype Surface Learning Theory for this model.

---

## **7. Next Steps / Further Analysis**

* **Analyze Learned Parameters:** Extract and analyze the learned weights ($w_1, w_2, b$) from a sample of successfully trained models under different initialization strategies (especially "Tiny" and "Normal").
* **Verify Prototype Surface Properties:** For successful runs, calculate $L(x_i) = w_1x_{i1} + w_2x_{i2} + b$ for all four XOR input patterns ($x_i$). Confirm that for inputs with target label 0, $|L(x_i)| \approx 0$, and for inputs with target label 1, $|L(x_i)| \ge c$ (where $c$ is ideally close to 1.0), as predicted by the Prototype Surface Theory.
* **Investigate Failed Runs:** Examine the weights and decision boundaries for runs that achieved only 50% accuracy to better understand the specific local minima or learning dynamics that led to these suboptimal solutions, particularly how the clamping function contributed.

