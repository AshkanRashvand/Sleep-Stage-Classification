# Automatic Sleep Stage Classification using Single-Channel EEG

## Description
This project focuses on the automatic classification of sleep stages using single-channel EEG data. We developed and compared multiple deep learning models, including:

- CNNTransformer
- CNNLSTM
- Multi-Scale CNN-based Representation Learning
- Feature-Enhanced Multi-Scale Convolutional Transformer (FEMSCTNet)

The goal was to improve the accuracy and efficiency of sleep stage classification. This project uses the Sleep-EDF20 dataset, which contains polysomnographic (PSG) files and hypnograms stored in the `Data` folder.

## Features
- Preprocessing of EEG signal for model input.
- Implementation of four different deep models.
- Evaluation metrics: Accuracy, F1 Score, and Loss.
- Classification report for detailed performance analysis.

## Folder Structure
```
├── data/
PSG files               # Contains raw PSG 
├── output/             # Processed data and model results
├── src/                # Source code for preprocessing and models
│   ├── CNNLSTM.py
│   ├── CNNTransformer.py
│   ├── data_loaderswindow.py
│   ├── MRCNNT.py
│   ├── MRCNNT3.py
│   ├── preprocessing.py
│   ├── psgreader.py
│   └── windowed_model.py
├── README.md           # Project documentation
├── requirements.txt    # List of dependencies
```

## Getting Started
### Prerequisites
- Python 3.10 or later
- Required Python libraries (see `requirements.txt`):
  - matplotlib==3.9.2
  - mne==1.8.0
  - mpmath==1.3.0
  - numpy==1.25.2
  - pandas==2.2.2
  - pillow==10.4.0
  - pytz==2024.2
  - scikit_learn==1.5.2
  - scipy==1.14.1
  - seaborn==0.13.2
  - torch==2.1.2+cu118

   

Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Step 1: Preprocessing
The Sleep-EDF20 dataset is available on Physionet website. First download the dataset and then preprocess the raw data before training the models. Run the following command:
```bash
python src/preprocessing.py
```
The preprocessed data will be saved in the `output` folder.

### Step 2: Model Training and Evaluation
Train and evaluate the models using the preprocessed data. To run a specific model, use the following commands:

- For CNNLSTM:
  ```bash
  python src/CNNLSTM.py
  ```

- For CNNTransformer:
  ```bash
  python src/CNNTransformer.py
  ```

- For Multi-Scale CNN-based Representation Learning:
  ```bash
  python src/MRCNNT.py
  ```

- For : Feature-Enhanced Multi-Scale Convolutional Transformer (FEMSCTNet)
  ```bash
  python src/MRCNNT3.py
  ```

### Step 3: Results
After running a model, the following outputs will be generated:
- Accuracy, F1 Score, and Loss metrics
- A classification report displayed in the terminal





---

Thank you for exploring this project! We hope our models contribute to advancing sleep stage classification research.

