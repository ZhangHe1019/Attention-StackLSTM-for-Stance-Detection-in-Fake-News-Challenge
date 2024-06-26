## Prerequisites
- Python 3.9 installed
- Required Python libraries installed (refer to `requirements.txt`)

#### Data Downloading
Please visit [here](https://drive.google.com/drive/folders/1FpmW9SXt3loD_Cna0C33X_L83wJQDW6n?usp=drive_link) to download the dataset and the extracted features (approximately 1GB) and unzip them in the current directory.
![Screenshot 2024-06-26 210736](https://github.com/ZhangHe1019/Attention-based-Model-for-Stance-Detection-in-Fake-News-Challenge/assets/103262469/747d2621-05bf-4cd9-a732-4ec3d7d1b1a4)


#### Data Preprocessing
```bash
./run.sh preprocessing
```
This command executes three Python scripts for generating different types of features: bag-of-words, baseline features, and topic features. All scripts are configured for the 'train' mode using the 'fnc-1' corpus.

#### Feature Selection and Model Configuration
```bash
./run.sh selection [subcommand]
```
Subcommands for selection include:

| Subcommand      | Description                             |
|-----------------|-----------------------------------------|
| **embedding**   | Selects the best word embedding model.  |
| **feature**     | Executes feature selection.             |
| **layer**       | Adjusts the model structure.            |
| **hyperparameter** | Tunes the hyperparameters.           |

For example, to execute feature selection:
```bash
./run.sh selection feature
```

#### Model Training
```bash
./run.sh train
```
Trains models on both full and partial subsets of data.

#### Testing
```bash
./run.sh test
```
Tests models on both full and partial subsets of data.
