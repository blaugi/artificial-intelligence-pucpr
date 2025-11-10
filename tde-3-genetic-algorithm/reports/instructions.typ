#set text(font: "Atkinson Hyperlegible")
#set text(lang: "pt")
#set text(region: "br")
#set par(
  justify: true,
  leading: 0.52em,
)
== TDE 3 - Optimization: Feature Selection using Genetic Algorithms
=== Activity Theme:
Implementation of a genetic algorithm to optimize feature selection in a classification task using the Breast Cancer Dataset.

=== Objective:
Apply evolutionary computing techniques to optimize feature selection. The goal is to improve classifier accuracy while reducing feature dimensionality, using a wrapper-based strategy guided by a Genetic Algorithm (GA).
=== Description:
Each Breast Cancer instance has 30 features. Students (in groups of up to 4) must implement a Genetic Algorithm to search for an optimal subset of features that maximizes classification accuracy.
The selected features will be evaluated using a KNN (from Scikit-Learn, with default parameters).
The Breast Cancer dataset should be split into three subsets: training (60%), validation (20%), and test (20%). In each generation of the GA, individuals (feature subsets) will be used to train classifiers on the training portion and evaluated on the validation set.
Additionally, the results must be compared to:
-	Classifier with all features
Deliverables:
-	PDF Report including:
  -	Description of the GA parameters and implementation
  -	Comparative results using the table below
-	Source Code (Python)


#table(
  columns: (auto, auto, auto, auto),
    table.header([*Feature Selection Method*],[*\# of Features*],[*Accuracy Test Set (%)*], [*Execution time (s)*]),
    align: (x, y) =>
    if y == 0 { center } else { top },
    [Without Selection], [], [], [],
    [Genetic Algorithm], [], [], [],

  )

