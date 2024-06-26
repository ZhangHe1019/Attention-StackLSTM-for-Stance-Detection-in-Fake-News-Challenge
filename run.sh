#!/bin/bash
export PYTHONPATH=$(pwd)

# Check the input argument and execute the corresponding python script
if [ "$1" == "preprocessing" ]; then
    python3 ./data/bow_feature_generation.py --mode train --corpus fnc-1
    python3 ./data/baseline_feature_generation.py --mode train --corpus fnc-1
    python3 ./data/topic_feature_generation.py --mode train --corpus fnc-1
elif [ "$1" == "selection" ]; then
    if [ "$2" == "embedding" ]; then
        python3 ./scripts/word_embedding_selection.py
    elif [ "$2" == "feature" ]; then
        python3 ./scripts/feature_selection.py
    elif [ "$2" == "layer" ]; then
        python3 ./scripts/structure_improvement.py
    elif [ "$2" == "hyperparameter" ]; then
        python3 ./scripts/hyperparameter_tuning.py
    else
        echo "Invalid second argument for 'selection'. Use 'embedding', 'feature', 'layer', or 'hyperparameter'."
    fi
elif [ "$1" == "train" ]; then
        python3 ./scripts/train.py --subset full
        python3 ./scripts/train.py --subset partial
elif [ "$1" == "test" ]; then
        python3 ./scripts/test.py --subset full
        python3 ./scripts/test.py --subset partial
else
    echo "Invalid first argument. Please use 'preprocessing', 'selection', 'train', or 'test'."
fi
