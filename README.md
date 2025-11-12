# Food101 Conformal Prediction on Databricks

A production-ready implementation of conformal prediction for food image classification using Vision Transformer (ViT) on the Food101 dataset. This project implements RAPS (Regularized Adaptive Prediction Sets) to provide uncertainty-aware predictions with statistical coverage guarantees.

**Key Innovation: Adaptive Prediction Sets**
Unlike traditional classifiers that output a single prediction, this model generates *adaptive prediction sets* that grow or shrink based on uncertainty. High-confidence predictions yield small sets (1-2 classes), while uncertain predictions expand to include more plausible options (up to 6 classes), all while maintaining a rigorous 90% coverage guarantee.

![Food101 Conformal Prediction Examples](assets/images/food101_3_classes.png)

*Three examples showing conformal prediction with 90% coverage guarantee. **Left (High Confidence):** Clean breakfast burrito image produces a single-class prediction set. **Center (Medium Confidence):** More ambiguous image yields 3 possible classes. **Right (Low Confidence):** Difficult image requires 6 classes to maintain 90% coverage. The prediction set size adapts automatically based on model uncertainty.*

---

## Overview

This project demonstrates **uncertainty quantification** for deep learning image classification using conformal prediction. Instead of forcing the model to commit to a single prediction, this approach returns **prediction sets** — multiple plausible classes that come with rigorous statistical guarantees about correctness.

### Why Conformal Prediction?

Traditional classifiers face a fundamental limitation: they must always choose one answer, even when multiple options are equally plausible. Conformal prediction solves this by being honest about uncertainty.

**Standard Classification:**
- Forces a single prediction: "This is a breakfast burrito (97.9% confidence)"
- Confidence scores can be misleading and uncalibrated
- No guarantees about how often predictions are correct

**Conformal Prediction:**
- Returns flexible prediction sets: "The true label is in {breakfast_burrito, tacos, huevos_rancheros} with 90% confidence"
- Provides mathematical coverage guarantees that hold regardless of model or data distribution
- Adapts set size to uncertainty: confident predictions → small sets (1-2 classes), uncertain predictions → larger sets (3-6 classes)

This approach is particularly valuable in high-stakes applications like medical diagnosis, food safety, or quality control where knowing when the model is uncertain is as important as the prediction itself.

---

## Understanding the Results

The image at the top shows three prediction examples that demonstrate how prediction sets adapt to model confidence:

| Confidence Level | Set Size | Interpretation |
|-----------------|----------|----------------|
| **High** (1-2 classes) | 1 | Strong evidence for one class - the model can confidently exclude all other options |
| **Medium** (3-5 classes) | 3 | Ambiguous features suggest multiple legitimate candidates - honest uncertainty |
| **Low** (6+ classes) | 6 | Insufficient evidence to narrow down - could be out-of-distribution or genuinely difficult |

### What Does "90% Coverage" Mean?

The coverage guarantee is the key insight: **across many predictions, at least 90% of the prediction sets will contain the true label.**

This is fundamentally different from traditional "confidence scores":
- ❌ Traditional: "90% confident this is class A" (no calibration guarantee)
- ✅ Conformal: "90% of my prediction sets include the correct answer" (mathematically guaranteed)

The model adapts its honesty: when uncertain, it returns larger sets rather than guessing. This transparency allows downstream decisions to account for uncertainty appropriately.

---

## Key Benefits

- ✅ **Statistical guarantees** - 90% coverage provably maintained
- ✅ **Uncertainty quantification** - Know when the model is uncertain
- ✅ **Distribution-free** - No assumptions about data distribution
- ✅ **Model-agnostic** - Works with any classifier
- ✅ **Production-ready** - Deployed on Databricks Model Serving
- ✅ **Reproducible** - Pinned dependencies and MLflow tracking

---

## Quick Start

### Prerequisites

```bash
# Databricks Runtime 13.0+ with ML support
# Python 3.10+
# GPU recommended for notebook 01 (training):
#   - Serverless GPU compute (recommended - zero setup)
#   - OR GPU cluster (g4dn.xlarge or larger)
```

### Installation

1. **Clone this repository directly into Databricks**
   - Open your Databricks workspace
   - Navigate to Repos in the left sidebar
   - Click "Add Repo" or "Create Repo"
   - Enter the repository URL: `https://github.com/yourusername/food101_conformal_prod.git`
   - Click "Create Repo"

2. **Attach to a cluster**
   - Open either notebook
   - Attach to a cluster with ML runtime (GPU recommended for notebook 01)
   - Dependencies will be installed automatically via `%pip install` commands in the notebooks

### Running the Notebooks

**Step 1: Fine-tune the base model**
```
Open: 01-Fine-Tune-CV-Model.ipynb
Run all cells (takes ~20-25 minutes with GPU)
Output: Trained ViT model in Unity Catalog
```

**Step 2: Add conformal prediction**
```
Open: 02-Conformal-Model.ipynb
Run all cells (takes ~10-15 minutes)
Output: Conformal model with prediction sets
```

**Step 3: Deploy to production** (optional)
```
Last cell in notebook 02 creates a Model Serving endpoint
Test via REST API
```

---

## Notebooks

### [01-Fine-Tune-CV-Model.ipynb](01-Fine-Tune-CV-Model.ipynb)

Fine-tunes a Vision Transformer (ViT) model on the Food101 dataset and registers it to Unity Catalog.

**What you'll learn:**
- Transfer learning with Hugging Face Transformers
- MLflow experiment tracking on Databricks
- Unity Catalog model registration
- Model validation and inference testing

**Requirements:**
- Databricks cluster with GPU (recommended for training)
  - Option 1: [Serverless GPU compute](https://docs.databricks.com/aws/en/compute/serverless/gpu) (fastest setup - no cluster config needed)
  - Option 2: GPU cluster with ML runtime (g4dn.xlarge or larger)
- Unity Catalog enabled workspace
- Food101 dataset in Delta table format

**Key steps:**
1. Install pinned dependencies for reproducibility
2. Load Food101 data from Delta table (15,150 train images, 5,050 test images)
3. Fine-tune Vision Transformer with transfer learning
4. Track training with MLflow (accuracy, loss, F1 score)
5. Register model to Unity Catalog
6. Test inference with sample predictions

---

### [02-Conformal-Model.ipynb](02-Conformal-Model.ipynb)

Wraps the base classifier with conformal prediction using RAPS (Regularized Adaptive Prediction Sets).

**What you'll learn:**
- Conformal prediction with RAPS algorithm
- Calibration-based uncertainty quantification
- MLflow PyFunc custom model wrapping
- Model serving endpoint deployment

**Key concepts:**
- **Calibration:** Uses 5,050 images to compute conformal thresholds
- **Coverage guarantee:** 90% of prediction sets contain the true label
- **Adaptive sets:** Set size varies from 1-6 classes based on uncertainty
- **RAPS parameters:** λ=0.01, k=5 for regularization

**Key steps:**
1. Load base Vision Transformer from Unity Catalog
2. Generate calibration scores on held-out data
3. Compute RAPS conformal quantile (qhat)
4. Create PyFunc wrapper for prediction sets
5. Validate empirical coverage (target: 90%)
6. Deploy to Databricks Model Serving endpoint

---

## Technical Details

### Model Architecture
- **Base Model:** Vision Transformer (google/vit-base-patch16-224-in21k)
- **Dataset:** nateraw/food101 (101 food categories)
- **Fine-tuning:** 5 epochs, learning rate 2e-4, batch size 8
- **Evaluation Results:**
  - **Accuracy:** 84.53%
  - **Loss:** 0.6771

### Conformal Prediction
- **Algorithm:** RAPS (Regularized Adaptive Prediction Sets)
- **Coverage target:** 90% (α = 0.1)
- **Empirical coverage:** 92.1% (validated on 5,050 test images)
- **Average set size:** 1.83 classes
- **Median set size:** 1 class

### Package Versions (Pinned)
```
transformers==4.46.3
torch==2.5.1
torchvision==0.20.1
mlflow==2.18.0
pillow==11.0.0
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.5.2
matplotlib==3.9.2
```

See [requirements.txt](requirements.txt) for complete dependency list.

---

## Project Structure

```
food101_conformal_prod/
├── 01-Fine-Tune-CV-Model.ipynb    # Fine-tune Vision Transformer
├── 02-Conformal-Model.ipynb       # Add conformal prediction
├── requirements.txt               # Pinned Python dependencies
├── assets/
│   ├── images/
│   │   ├── food101_3_classes.png # Example prediction sets
│   │   ├── food101_plots.png     # Training metrics
│   │   └── image.jpg             # Test image for API or App
│   └── MLmodel                   # MLflow model metadata
└── README.md                      # This file
```

---

## References

This implementation is based on:

**Uncertainty Sets for Image Classifiers using Conformal Prediction**
Anastasios N. Angelopoulos, Stephen Bates, Jitendra Malik, and Michael I. Jordan
📄 Paper: [arXiv:2009.14193](https://arxiv.org/abs/2009.14193)
*International Conference on Learning Representations (ICLR), 2021*

**A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification**
Anastasios N. Angelopoulos and Stephen Bates
📄 Paper: [arXiv:2107.07511](https://arxiv.org/abs/2107.07511)
💻 Code: [github.com/aangelopoulos/conformal-prediction](https://github.com/aangelopoulos/conformal-prediction)

**Food101 Dataset:**
Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc
*Food-101 -- Mining Discriminative Components with Random Forests*
European Conference on Computer Vision, 2014

---

## License

MIT License - See LICENSE file for details

This project uses:
- Food101 dataset (research/educational use)
- Hugging Face Transformers (Apache 2.0)
- PyTorch (BSD-style license)

---

## Author

**Jonathan Whiteley**
Databricks Solutions Architect
Built on Databricks Platform

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

For bugs or feature requests, open an issue on GitHub.

---

**Built with ❤️ using Databricks, PyTorch, Hugging Face Transformers, and MLflow**
