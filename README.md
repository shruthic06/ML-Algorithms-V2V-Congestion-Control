
# Congestion Control for Traffic Regulation in Vehicular Networks


## Overview

This project addresses congestion control in Vehicular Ad Hoc Networks (VANETs) by combining **machine learning prediction** with an **adaptive transmission algorithm (P&A-A)**.  

- **Module 1 – Traffic Prediction:** Predict short-term local density (traffic state) using classification algorithms.  
- **Module 2 – Channel Busy Ratio (CBR) Prediction:** Use regression algorithms to estimate CBR from vehicular and packet-level parameters.  
- **Module 3 – Prediction & Adaptation Algorithm (P&A-A):** Dynamically tune transmission rate (TR) and power (TP) based on predicted traffic and CBR to maintain channel load under the threshold (40%).  

This pipeline achieves higher reliability, awareness, and efficient channel utilization compared to conventional Decentralized Congestion Control (DCC).

---

## Project Objectives

- Apply supervised ML classification to predict **traffic density states** (smooth, congested, blockage, etc.).  
- Predict **channel busy ratio (CBR)** with regression models for accurate congestion assessment.  
- Implement **Prediction & Adaptation Algorithm (P&A-A)** to tune beacon transmission parameters.  
- Compare performance against existing methods (e.g., DCC).  
- Provide an extensible codebase for vehicular communication research.

---

## Algorithms Implemented

### Traffic Classification (Module 1)
| Algorithm | Description |
|-----------|-------------|
| **Logistic Regression** | Parametric classifier for categorical traffic states |
| **KNN** | Distance-based classifier with K-nearest neighbors |
| **Decision Tree** | Tree-based model with Gini criterion |
| **Random Forest** | Ensemble of decision trees for robust classification |
| **Naïve Bayes** | Probabilistic model based on Bayes’ theorem |

### CBR Regression (Module 2)
| Algorithm | Description |
|-----------|-------------|
| **Ridge Regression** | L2-regularized linear regression, handles multicollinearity |
| **Lasso Regression** | L1-regularized regression, induces sparsity |
| **Elastic Net** | Combination of L1 + L2 penalties for balanced regression |
| **Decision Tree Regression** | Non-linear regression using tree splits |
| **Random Forest Regression** | Ensemble regression with high accuracy and stability |

### P&A-A Algorithm (Module 3)
- Adjusts **Transmission Rate (TR)** and **Transmission Power (TP)** jointly.  
- Uses predicted **local density** and **CBR**.  
- Ensures strict beaconing requirements (≥5 Hz) while keeping CBR < 40%.  
- Outperforms DCC in terms of utilization and awareness.

---

## Features

### End-to-End Pipeline
- Unified scripts for **classification**, **regression**, and **PA&A tuning**.  
- Modular design allows independent experimentation with each stage.

### Configurable
- CSV-based datasets (`train_set.csv`, `test_set.csv`, `cbrdataset.csv`).  
- Easy to swap in extended features or real vehicular traces.

### Evaluation Metrics
- **Classification:** Accuracy, Cohen’s Kappa, Precision/Recall, F1 Score.  
- **Regression:** MAE, MSE, RMSE, R², Adjusted R².  
- **Congestion Control:** Transmission rate, Transmission power, CBR over vehicle densities.

---

## Project Structure

```
.
├── classification/              # Module 1: Traffic prediction
│   └── allalg_clean.py
├── regression/                  # Module 2: CBR prediction
│   └── cbr_regression_clean.py
├── paa/                         # Module 3: Prediction & Adaptation
│   └── pa_a_clean.py
├── data/                        # Datasets (not committed; add to .gitignore)
│   ├── train_set.csv
│   ├── test_set.csv
│   └── cbrdataset.csv
├── requirements.txt             # Python dependencies
├── USED_FILES_REPORT.md         # Audit of legacy vs cleaned scripts
└── README.md
```

---

## Install Requirements

```bash
pip install -r requirements.txt
```

---

## 🔧 How to Run

### 1. Traffic Classification
```bash
# Interactive menu
python classification/all_algprithms_traffic_prediction.py

# Run specific algorithm
python classification/all_algprithms_traffic_prediction.py --algo rf
```

### 2. CBR Regression
```bash
python regression/regression_algorithms_cbr.py --algo ridge --csv data/cbrdataset.csv --target CBR
python regression/regression_algorithms_cbr.py --algo lasso --csv data/cbrdataset.csv --target CBR
python regression/regression_algorithms_cbr.py --algo enet  --csv data/cbrdataset.csv --target CBR
python regression/regression_algorithms_cbr.py --algo rfr   --csv data/cbrdataset.csv --target CBR
```

### 3. PA&A Tuning
```bash
python paa/pa_a.py --csv cbrdataset.csv --data-dir data --cbr-col CBR
# Output: pa_a_tuned.csv
```

---

## Weather Codes (Traffic Dataset)

Weather is encoded numerically (0–47). Example:  
```
0 → tornado, 1 → tropical storm, 2 → hurricane, …, 32 → sunny, 46 → snow showers, 47 → isolated thundershowers
```

---

## Future Scope

- Integrate **neural networks** for traffic prediction (e.g., ANN, LSTM).  
- Include **collision rate** parameter in PA&A for higher awareness.  
- Extend to hybrid **V2V + V2I** environments.  
- Explore real-world datasets from **SUMO / Veins** mobility simulators.  
- Build dashboards for **interactive visualization** of TR, TP, CBR.

---

## References

- Zemouri, Djahel, Murphy. *Altruistic Prediction-Based Congestion Control for Strict Beaconing Requirements in Urban VANETs*. IEEE TSMC, 2019.  
- Bansal et al. *Comparing LIMERIC and DCC approaches for VANET channel congestion control*. IEEE WiVeC, 2014.  
- [Vehicular Traffic Dataset – NIT Patna](https://data.mendeley.com/)  
- Appendix & implementation details (Anna University BE Dissertation, 2021).  

---

## License

This project is licensed under the [MIT License](./LICENSE.md).
