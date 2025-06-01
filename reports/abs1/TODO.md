# TODO

Planned improvements and analysis extensions for the `abs1` XOR experiments.

---

## 🧠 Training Logic

* [ ] **Replace fixed epoch count with convergence-based training**

  * Use a stopping condition based on final loss or accuracy.
  * Retain a high max epoch limit (e.g., 5000) to allow for convergence.

* [ ] **Store the actual number of epochs completed**

  * Log this value per run in the stats object for later analysis.

---

## 💾 Experiment State Logging

* [ ] **Log the initial model state before training**

  * Save the initial values of weights and bias (`w0`, `w1`, `b`) before training.
  * Include this in the per-run output directory (`init_state.pt` or JSON).

---

## 📈 Geometry and Trajectory Analysis

* [ ] **Plot final (w0, w1, b) values as a 3D point cloud**

  * Color-code points by outcome (e.g., green = 100% accuracy, red = failure).
  * View projections (e.g., (w0, w1), (w0, b), etc.) to understand clustering.

* [ ] **Plot starting (w0, w1, b) values the same way**

  * Use the same color scheme to show which starting states lead to successful or failed outcomes.

* [ ] **Visualize relationships between starting and ending states**

  * Arrows or lines from initial to final state for each run.
  * Possibly animate or segment by outcome class.

---

## 🔍 Mode and Cluster Analysis

* [ ] **Cluster the final states**

  * Use a clustering algorithm (e.g., DBSCAN, KMeans) to identify modes in the solution space.
  * Count number of clusters and map them to accuracy levels.

* [ ] **Draw prototype surfaces for each cluster mode**

  * For each cluster centroid (w0, w1, b), draw the corresponding hyperplane in input space.
  * Include failed-mode planes in visualizations to understand typical failure geometries.

