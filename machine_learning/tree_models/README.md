# Tree Models

Machine learning project covering decision trees, random forests, and related tree-based models using Scikit-learn.

## Tasks

- **Task 0 — Build the tree:** Before any data was provided, we wrote a template function that says "if data ever arrives, build a decision tree with these rules" (`build_decision_tree`). It was configured to use Gini impurity with unlimited depth.
- **Task 1 — Train the tree:** We trained the empty tree we built on the real wine dataset (`train_tree`). From this point on, the tree had learned the patterns in the data.
- **Task 2 — Display the tree's rules:** We converted the trained tree's internal "if-then" logic (e.g., `proline <= 755`) into readable text and printed it (`draw`).
- **Task 3 — Generate predictions:** We gave the trained tree test data it had never seen and asked "which class do you think these belong to?", then collected the predictions (`generate_predictions`).
- **Task 4 — Measure performance:** We compared the predictions against the true labels and produced a report containing precision, recall, and f1-score (`evaluate`).
- **Task 5 — Pre-pruning:** Before growing the tree, we used grid search to find the best limits (`max_depth`, `min_samples_leaf`, `min_samples_split`). This was meant to prevent the tree from overfitting in advance. Accuracy rose from 0.75 to 0.83.
- **Task 6 — Extract the pruning path:** This time we took the opposite approach — first grow the tree without limits, then prune it (post-pruning). We extracted the list of "alpha" (pruning strength) values we could try.
- **Task 7 — Try every alpha:** For each value in the alpha list from Task 6, we trained a separate tree and measured both its train and test scores. The chart showed that light pruning helps, while excessive pruning oversimplifies the model and hurts performance.
- **Task 8 — Pick the best alpha:** Among the results from Task 7, we wrote logic that automatically selects the alpha with the best balance (first the highest test score, then the smallest train-test gap, then the largest alpha).
- **Task 9 — Random Forest:** Instead of relying on a single tree, we built a "forest" of 20 different trees. Accuracy rose to 0.94 — we saw the power of combining many trees.
- **Task 10 — Feature importance:** We measured and ranked how often and how effectively each feature (e.g., `proline`, `color_intensity`) was used across the trees in the forest. The most important feature turned out to be `proline`.
- **Task 11 — Boosting:** We tried boosting, a different method where trees sequentially correct each other's errors — we wrote a function that can choose between AdaBoost, GradientBoosting, XGBoost, and LightGBM, then compared the four in terms of speed and accuracy.

**Summary:** Starting from a single simple tree, we improved it through pruning, then moved on to multi-tree methods (forest, boosting) to progressively increase performance.
