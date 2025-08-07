# IonNTxPred

A computational framework for predicting and designing **ion channel-impairing proteins** using alignment-based, machine learning, and protein language model-based methods.

---

## 📌 Introduction
**IonNTxPred** is developed to help researchers identify proteins and peptides that modulate ion channels such as **sodium (Na⁺)**, **potassium (K⁺)**, **calcium (Ca²⁺)**, and **others**. It integrates traditional ML models, motif discovery, and state-of-the-art protein language models (PLMs) to deliver accurate predictions and insightful biological analysis.
It employs large language model for predicting toxic activity against ion channel. The final model offers **Prediction, Protein-Scanning, and Design** modules, implemented using protein language models.

🔗 Visit the web server for more information: [IonNTxPred Web Server](http://webs.iiitd.edu.in/raghava/ionntxpred)

🔗 Visit the Hugging Face: [IonNTxPred](https://huggingface.co/raghavagps-group/IonNTxPred)

📖 Please cite relevant content for complete details, including the algorithm behind the approach.

---

## 📚 Reference
**Rathore et al.** _LLM-based Prediction and Designing of Ion Channel Impairing Proteins._ **#Coming Soon#**

---
### 🖼️ IonNTxPred Workflow Representation
![IonNTxPred Workflow](https://raw.githubusercontent.com/saloni21098/IonNTxPred/main/images/IonNTxPred.png)


## 🧪 Quick Start for Reproducibility

Follow these steps to replicate the core results of our paper:

```bash
# 1. Clone the repository
git clone https://github.com/raghavagps/IonNTxPred.git
cd IonNTxPred

# 2. Set up the environment (conda recommended)
conda env create -f environment.yml
conda activate IonNTxPred

# 3. Download pre-trained models
# Visit: https://webs.iiitd.edu.in/raghava/IonNTxPred/download.html
# Download the model ZIP and extract it in the root directory

# 4. See the available optiopns
python ionntxpred.py -h

# 5. Run prediction on sample input
python ionntxpred.py -i example.fasta -o output.csv  -j 1 -m 1 -wd working_direcotory_path

```

## 🛠️ Installation Options


###  🧰 Pip Installation  ![PyPI Logo](images/PyPI_logo.svg.png)
To install IonNTxPred via PIP, run:
```bash
pip install ionntxpred
```
To check available options, type:
```bash
ionntxpred -h
```

### 🔹 Standalone Installation
IonNTxPred is written in **Python 3** and requires the following dependencies:

#### ✅ Required Libraries
```bash
python=3.10.7
pytorch
```
Additional required packages:
```bash
pip install scikit-learn==1.5.2
pip install pandas==1.5.3
pip install numpy==1.25.2
pip install torch==2.1.0
pip install transformers==4.34.0
pip install joblib==1.4.2
pip install onnxruntime==1.15.1
Bio (Biopython): 1.81
tqdm: 4.64.1
torch: 2.6.0
```

### 🔹 Installation using environment.yml
1. Create a new Conda environment:
```bash
conda env create -f environment.yml
```
2. Activate the environment:
```bash
conda activate IonNTxPred
```

---

## ⚠️ Important Note
- Due to the large size of the model file, the model directory has been compressed and uploaded.
- Download the **zip file** from [Download Page](https://webs.iiitd.edu.in/raghava/ionntxpred/download.html).
- **Extract the file** before using the code or model.

---

## 🔬 Classification
**IonNTxPred** classifies peptides and proteins as **ion channel impairing or non-impairing** based on their primary sequence.

🔹 **Model Options**
- **ESM2-t33**
- **Hybrid model (ESM2-t33+MERCI)**: Default Mode **

---

## 🚀 Usage

### 🔹 Minimum Usage
```bash
ionntxpred.py -h
```
To run an example:
```bash
ionntxpred.py -i example.fasta
```

### 🔹 Full Usage
```bash
usage: ionntxpred.py [-h]
                   [-i INPUT]
                   [-o OUTPUT]
                   [-t THRESHOLD]
                   [-j {1,2,3,4}]
                   [-c Channel]
                   [-m {1,2,3}]
                   [-d {1,2}]
                   [-wd WORKING DIRECTORY]
```
#### Required Arguments
| Argument | Description |
|----------|-------------|
| `-i INPUT` | Input: Peptide or protein sequence (FASTA format or simple format) |
| `-o OUTPUT` | Output file (default: `outfile.csv`) |
| `-t THRESHOLD` | Threshold (0-1, default: `0.5`) |
| `-j {1,2,3,4}` | Job type: 1-Prediction, 2-Protein Scanning, 3-Design all possible mutants, 4- Motif Scanning |
| `-c {1,2,3,4}` | Ion channel type: 1: Na+, 2: K+, 3: Ca+, 4: Other |
| `-m {1,2,3}` | Model selection: 1: ESM2-t33, 2: Hybrid (ESM2-t33 + MERCI) |
| `-wd WORKING` | Working directory for saving results |

---

## 📂 Input & Output Files

### ✅ **Input File Format**
IonNTxPred supports two formats:
1. **FASTA Format:** (Example: `example.fasta`)
2. **Simple Format:** (Example: `example.seq`, each sequence on a new line)

### ✅ **Output File**
- Results are saved in **CSV format**.
- If no output file is specified, results are stored in `outfile.csv`.

---

## 🔍 Jobs & Features

### 🔹 **Job Types**
| Job | Description |
|-----|-------------|
| 1️⃣ **Prediction** | Predicts whether the input peptide/protein is an ion channel impairing or not. |
| 2️⃣ **Protein Scanning** | Identifies toxic regions in a protein sequence. |
| 3️⃣ **Design** | Generates and predicts **all possible mutants**. |
| 4️⃣ **Motif Scanning** | Identifies motifs using MERCI |



### 🔹 **Additional Options**
| Option | Description |
|--------|-------------|
| `-p POSITION` | Position to insert mutation (1-indexed) |
| `-r RESIDUES` | Mutated residues (single/double letter amino acid codes) |
| `-w {8-20}` | Window length (Protein Scan mode only, default: 12) |
| `-d {1,2}` | Display: 1-Ion channel impairing only, 2-All peptides (default) |

---

## 📑 Package Contents

| File | Description |
|------|-------------|
| **INSTALLATION** | Installation instructions |
| **LICENSE** | License information |
| **README.md** | This file |
| **IonNTxPred.py** | Python program for classification |
| **example.fasta** | Example file (FASTA format) |

---

## 📦 PIP Installation (Again for Reference)
```bash
pip install ionntxpred
```
Check options:
```bash
ionntxpred -h
```

---

🚀 **Start predicting toxicity with IonNTxPred today!**

🔗 Visit: [IonNTxPred Web Server](http://webs.iiitd.edu.in/raghava/ionntxpred/)

