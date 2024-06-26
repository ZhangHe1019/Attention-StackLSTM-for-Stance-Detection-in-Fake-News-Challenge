## Prerequisites
- Python 3.9 installed
- Required Python libraries installed (refer to `requirements.txt`)


#### Data Preprocessing
```bash
./script_name.sh preprocessing
```
This command executes three Python scripts for generating different types of features: bag-of-words, baseline features, and topic features. All scripts are configured for the 'train' mode using the 'fnc-1' corpus.

#### Feature Selection and Model Configuration
```bash
./script_name.sh selection [subcommand]
```
Subcommands for selection include:
embedding: Selects the best word embedding model.
feature: Executes feature selection.
layer: Adjusts the model structure.
hyperparameter: Tunes the hyperparameters.

```bash
./script_name.sh selection feature
```

#### Model Training
```bash
./script_name.sh train
```
Trains models on both full and partial subsets of data.

#### Testing
```bash
./script_name.sh test
```
Tests models on both full and partial subsets of data.
