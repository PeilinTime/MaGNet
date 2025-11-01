# MaGNet: A Mamba Dual-Hypergraph Network for Stock Prediction via Temporal-Causal and Global Relational Learning

## Introduction 📖
MaGNet is a Mamba dual-hypergraph network that integrates advanced temporal modeling with dual hypergraph relational learning to capture both causal and global market dependencies.

### Framework Overview
![MaGNet Framework](https://github.com/PeilinTime/MaGNet/blob/main/figrues/MaGNet%20Framework.png)

---




## Datasets & Model Weights 📦

All datasets and model weights are available on Google Drive (anonymized already):
👉 [Download Link](https://drive.google.com/drive/folders/1fh3NTVLAF3GE00iHVng7HojNcr1-W5Du?usp=sharing)

Included datasets:

* **DJIA**
* **NASDAQ 100**
* **CSI 300**

---

##  How to Run MaGNet 🚀

### 1. Download this repository

Download or clone this code repository to your local machine.

### 2. For Training and Prediction

Download one of the datasets from the Google Drive link above:

* DJIA: `djia_alpha158_alpha360.pt`
* NASDAQ100: `nas100_alpha158_alpha360.pt`
* CSI300: `csi300_alpha158_alpha360.pt`

Place the downloaded file in the same directory as the codebase.
Run the following command to train the model and make predictions (including training, validation, and test sets):

```bash
python train.py
```

### 3. For Backtesting

Download the dataset **and** its corresponding model weight from the same link:

* DJIA: `djia_alpha158_alpha360.pt` & `djia_weight.pt`
* NASDAQ100: `nas100_alpha158_alpha360.pt` & `nas100_weight.pt`
* CSI300: `csi300_alpha158_alpha360.pt` & `csi300_weight.pt`

Place the downloaded files in the same directory as the codebase.
Run the following command to perform backtesting and results:

```bash
python backtest.py
```

---

##  Backtesting Results 📈

Below are the backtesting performance charts of MaGNet on all datasets:

![Backtesting_result_DJIA](https://github.com/PeilinTime/MaGNet/blob/main/figrues/Backtesting_result_DJIA.png)
![Backtesting_result_NASDAQ100](https://github.com/PeilinTime/MaGNet/blob/main/figrues/Backtesting_result_NASDAQ100.png)
![Backtesting_result_CSI300](https://github.com/PeilinTime/MaGNet/blob/main/figrues/Backtesting_result_CSI300.png)
![Backtesting_result_HSI](https://github.com/PeilinTime/MaGNet/blob/main/figrues/Backtesting_result_HSI.png)
![Backtesting_result_SP100](https://github.com/PeilinTime/MaGNet/blob/main/figrues/Backtesting_result_SP100.png)
![Backtesting_result_Nikkei225](https://github.com/PeilinTime/MaGNet/blob/main/figrues/Backtesting_result_Nikkei225.png)
