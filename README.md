# Food101 Conformal Prediction on Databricks

A production-ready implementation of conformal prediction for food image classification using Vision Transformer (ViT) on the Food101 dataset.

![Food101 Conformal Prediction Examples](images/food101_3_classes.png)

*Three examples showing conformal prediction with 90% coverage guarantee. **Left (High Confidence):** Clean breakfast burrito image produces a single-class prediction set. **Center (Medium Confidence):** More ambiguous image yields 3 possible classes. **Right (Low Confidence):** Difficult image requires 6 classes to maintain 90% coverage. The prediction set size adapts automatically based on model uncertainty.*

## Overview

This project demonstrates **uncertainty quantification** for deep learning image classification using conformal prediction. Instead of just returning a single prediction, the model returns **prediction sets** with statistical coverage guarantees.

**What makes this different from standard classification?**
- Standard models: "I predict this is a breakfast burrito (97.9% confidence)"
- Conformal prediction: "I guarantee with 90% probability the true label is in this set: {breakfast_burrito}"

The prediction set size adapts based on model uncertainty - confident predictions get smaller sets, uncertain predictions get larger sets.

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

### [02-Conformal-Wrapper.ipynb](02-Conformal-Wrapper.ipynb)

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

## Quick Start

### Prerequisites

```bash
# Databricks Runtime 13.0+ with ML support
# Python 3.10+
# GPU recommended for notebook 01 (training)
```

### Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/yourusername/food101_conformal_prod.git
   ```

2. **Install dependencies** (outside Databricks)
   ```bash
   pip install -r requirements.txt
   ```

3. **Upload to Databricks Workspace**
   - Import notebooks to your Databricks workspace
   - Attach to a cluster with ML runtime

### Running the Notebooks

**Step 1: Fine-tune the base model**
```
Open: 01-Fine-Tune-CV-Model.ipynb
Run all cells (takes ~30-40 minutes with GPU)
Output: Trained ViT model in Unity Catalog
```

**Step 2: Add conformal prediction**
```
Open: 02-Conformal-Wrapper.ipynb
Run all cells (takes ~10-15 minutes)
Output: Conformal model with prediction sets
```

**Step 3: Deploy to production** (optional)
```
Last cell in notebook 02 creates a Model Serving endpoint
Test via REST API or Gradio interface
```

---

## Technical Details

### Model Architecture
- **Base:** Vision Transformer (google/vit-base-patch16-224-in21k)
- **Fine-tuning:** 8 epochs, learning rate 2e-4, batch size 8
- **Performance:** 84.5% top-1 accuracy on Food101 test set

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
├── 02-Conformal-Wrapper.ipynb     # Add conformal prediction
├── requirements.txt               # Pinned Python dependencies
├── requirements-app.txt           # Gradio app dependencies
├── app.py                         # Gradio interface (optional)
├── images/
│   └── food101_3_classes.png     # Example predictions
├── DEPLOYMENT_SUMMARY.md          # Deployment guide
├── GRADIO_APP_GUIDE.md           # Gradio app instructions
└── README.md                      # This file
```

---

## Key Benefits

✅ **Statistical guarantees** - 90% coverage provably maintained
✅ **Uncertainty quantification** - Know when the model is uncertain
✅ **Distribution-free** - No assumptions about data distribution
✅ **Model-agnostic** - Works with any classifier
✅ **Production-ready** - Deployed on Databricks Model Serving
✅ **Reproducible** - Pinned dependencies and MLflow tracking

---

## Understanding the Results

The image at the top shows three prediction examples:

| Confidence Level | Set Size | Interpretation |
|-----------------|----------|----------------|
| **High** (1-2 classes) | 1 | Model is very confident - clear features match one class |
| **Medium** (3-5 classes) | 3 | Model sees ambiguous features - multiple plausible answers |
| **Low** (6+ classes) | 6 | Model is uncertain - image is difficult or out-of-distribution |

The 90% coverage guarantee means: **across many predictions, at least 90% of the prediction sets will contain the true label.**

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

