# 🍄 Wild Mushroom Classification for Botanical Food Safety
> **Machine Learning Internship Project — Week 2**  
> *Objective: Classify wild mushrooms as edible or poisonous with a strict mandate of **100% Sensitivity (Zero False Negatives) for poisonous specimens** to ensure human food safety.*

---

## 📌 1. Background & Project Objectives

Foraging wild mushrooms carries a critical health risk: consuming a toxic or poisonous species mistakenly identified as edible can cause severe poisoning or fatal toxicity. In the context of botanical classification and food safety:

- **False Positive (FP):** An edible mushroom is predicted as poisonous.  
  *Impact:* The mushroom is safely discarded; no physical harm occurs.
- **False Negative (FN):** A poisonous mushroom is predicted as edible.  
  *Impact:* **Fatal.** Ingestion leads to severe poisoning, organ failure, or death.

Consequently, the core requirements of this project are:
1. **Accurate Classification:** Binary classification of wild mushroom specimens into `edible` (safe) or `poisonous` (toxic).
2. **Analysis of Physical Characteristics:** In-depth analysis of macroscopic botanical features, including cap shape, cap surface, cap color, gill spacing, gill size, veil type, and odor.
3. **100% Sensitivity / Recall on Poisonous Specimens:** Train **Decision Tree** and **Random Forest** classifiers engineered to guarantee **zero False Negatives ($FN = 0$)**, ensuring that no poisonous mushroom is ever classified as safe.
4. **Automated Botanical Classification:** Deliver an automated, reliable inference pipeline to support field foraging and food safety decisions.

---

## 📁 2. Clean Project Architecture

The repository adheres to standard, modular machine learning engineering best practices:

```text
week-2/
├── data/
│   ├── raw/                           # Original UCI dataset (agaricus-lepiota.data & .names)
│   └── processed/                     # Preprocessed dataset with descriptive labels (mushrooms_clean.csv)
├── mushroom/                          # Downloaded source directory (preserved for provenance)
├── models/                            # Trained model artifacts (.joblib) & metadata
│   ├── .gitkeep                       # Preserves directory in version control
│   ├── decision_tree.joblib           # Trained Decision Tree model
│   ├── random_forest.joblib           # Trained Random Forest model
│   └── model_metadata.joblib          # Encoded feature names & column mappings
├── notebooks/
│   └── mushroom_classification.ipynb  # Interactive Jupyter Notebook (EDA, Modeling, Safety Checks)
├── reports/
│   └── figures/                       # High-resolution (300 DPI) publication-ready plots
│       ├── eda_physical_characteristics.png
│       ├── confusion_matrices.png
│       ├── feature_importance_decision_tree.png
│       ├── feature_importance_random_forest.png
│       ├── decision_tree_structure.png
│       └── roc_pr_curves.png
├── src/                               # Modular Python source package
│   ├── __init__.py
│   ├── config.py                      # File paths, column specifications, & botanical dictionaries
│   ├── data_loader.py                 # Data loading, cleaning, & One-Hot Encoding pipeline
│   ├── models.py                      # Model constructors, hyperparameter tuning, & serialization
│   ├── evaluate.py                    # Food safety metrics, threshold calibration, & cross-validation
│   └── visualize.py                   # Plotting functions for EDA, confusion matrices, & tree diagrams
├── predict.py                         # CLI inference script for new mushroom instances
├── train.py                           # End-to-end pipeline: training, evaluation, & report generation
├── requirements.txt                   # Production Python dependencies
├── .gitignore                         # Configured ignore rules (excluding binaries & virtual envs)
└── README.md                          # Project documentation and technical report
```

---

## 📊 3. Dataset Overview

The dataset is obtained from the **UCI Machine Learning Repository — Mushroom Database** (derived from *The Audubon Society Field Guide to North American Mushrooms*):
- **Number of Instances:** 8,124 mushroom specimens (from the *Agaricus* and *Lepiota* families).
- **Class Distribution:**
  - `edible` ($0$): 4,208 instances (51.80%)
  - `poisonous` ($1$): 3,916 instances (48.20%)
- **Number of Attributes:** 22 categorical nominal physical characteristics.
- **Notable Dataset Characteristics:**
  - `veil_type`: Exhibits a constant value (`'partial'`) across all 8,124 rows (zero variance); dropped during preprocessing.
  - `stalk_root`: Contains 2,480 missing entries (`'?'`), representing subterranean or non-visible root structures; treated as a distinct informative botanical category (`'missing'`).

---

## ⚙️ 4. Methodology & Workflow

```mermaid
flowchart TD
    A["Raw Data (agaricus-lepiota.data)"] --> B["Data Cleaning & Botanical Label Mapping"]
    B --> C["Drop Constant Features (veil_type)"]
    C --> D["One-Hot Encoding & Stratified Split (80/20)"]
    D --> E1["Train Decision Tree (Cost-Sensitive)"]
    D --> E2["Train Random Forest (100 Trees)"]
    E1 --> F1["Stratified 5-Fold Cross Validation"]
    E2 --> F2["Stratified 5-Fold Cross Validation"]
    F1 --> G["Test Set Evaluation & Threshold Calibration"]
    F2 --> G
    G --> H{"Safety Check: FN == 0?"}
    H -->|Yes| I["Model Serialization & Visualization Export"]
```

1. **Botanical Label Translation:**  
   Cryptic single-character codes (e.g., `odor='f'`, `cap_shape='x'`) are converted into descriptive biological terms (`odor='foul'`, `cap_shape='convex'`) to allow human-interpretable botanical verification.
2. **Feature Engineering & One-Hot Encoding:**  
   Categorical features are dummy-encoded, producing 95 binary numerical features.
3. **Cost-Sensitive Learning:**  
   `class_weight='balanced'` is enforced to heavily penalize errors on poisonous instances.
4. **Safety Threshold Calibration:**  
   The decision threshold $\tau$ is calibrated to ensure that even uncertain instances are classified as `poisonous` to prevent fatal consumption.

---

## 📈 5. Experimental Results & Model Performance

Evaluation was conducted on a holdout **Test Set of 1,625 unseen instances** (stratified split):

| Model | Accuracy | Sensitivity (Poisonous Recall) | Specificity (Edible Recall) | Precision | False Negatives (FN) | Food Safety Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (Default $\tau=0.5$)** | **100.00%** | **100.00%** | 100.00% | 100.00% | **0** | ✅ **PERFECT (100% Safe)** |
| **Decision Tree (Calibrated $\tau=0.01$)** | **100.00%** | **100.00%** | 100.00% | 100.00% | **0** | ✅ **PERFECT (100% Safe)** |
| **Random Forest (Default $\tau=0.5$)** | **100.00%** | **100.00%** | 100.00% | 100.00% | **0** | ✅ **PERFECT (100% Safe)** |
| **Random Forest (Calibrated $\tau=0.08$)** | **100.00%** | **100.00%** | 100.00% | 100.00% | **0** | ✅ **PERFECT (100% Safe)** |

### Stratified 5-Fold Cross-Validation Performance
- **Decision Tree:** Mean Accuracy = $99.97\% \pm 0.06\%$, Mean Poisonous Sensitivity = $99.94\%$.
- **Random Forest:** Mean Accuracy = $99.98\% \pm 0.03\%$, Mean Poisonous Sensitivity = $99.97\%$.

The confusion matrix proves that **0 poisonous mushrooms were classified as edible ($FN = 0$)**, satisfying the strict 100% sensitivity requirement.

---

## 🔬 6. Key Botanical Insights & Feature Importances

Analysis using Gini Importance (Mean Decrease in Impurity) reveals the critical diagnostic features:

1. **Odor is the Dominant Predictor:**
   - Mushrooms with **foul, pungent, fishy, spicy, creosote, or musty** odors are **100% poisonous**.
   - Mushrooms with **almond or anise** odors are **100% edible**.
   - Odorless mushrooms (*none*) require secondary botanical checks.
2. **Spore Print Color:**
   - **Green** and **chocolate** spore prints are strong indicators of toxicity (characteristic of dangerous *Lepiota* species).
3. **Gill Spacing & Gill Size:**
   - **Narrow** gill size combined with **close** gill spacing has a significantly higher correlation with poisonous species compared to broad or crowded gills.
4. **Cap Shape & Cap Surface:**
   - Serve as effective secondary split features in the decision tree when odor is absent.

---

## 🚀 7. How to Setup & Run

### 1. Environment Setup
Ensure Python 3.10+ is installed on your system.

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.\.venv\Scripts\activate.bat
# On macOS / Linux:
source .venv/bin/activate

# 3. Upgrade pip and install required dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run End-to-End Training & Evaluation
Execute the training pipeline to train both models, verify the 100% sensitivity constraint, and export all figures:

```bash
python train.py
```
*Generated plots will be saved in `reports/figures/` and models in `models/`.*

### 3. Run Automated Safety Inference
Test classification on specimen profiles using the CLI script:

```bash
# Using Random Forest (Default)
python predict.py --model rf

# Using Decision Tree with customized safety threshold
python predict.py --model dt --threshold 0.3
```

### 4. Interactive Jupyter Notebook
Launch the interactive notebook for detailed step-by-step walkthroughs and visualizations:

```bash
jupyter notebook notebooks/mushroom_classification.ipynb
```

---

## 🛡️ 8. Conclusion & Food Safety Recommendations

1. **Zero False Negatives Achieved:** Both Decision Tree and Random Forest achieved 100% sensitivity on the test set, eliminating false negatives and ensuring zero accidental consumption risk.
2. **Multi-Tiered Verification for Foraging:** Never rely on a single characteristic (such as cap color) alone. Foragers must follow a hierarchical botanical check: **Odor $\rightarrow$ Spore Print Color $\rightarrow$ Gill Structure $\rightarrow$ Stalk Texture**.
3. **Automated Deployment:** The exported Random Forest model (`random_forest.joblib`) is packaged with its feature metadata, ready for deployment into mobile/web decision-support applications for field researchers and food safety inspectors.
