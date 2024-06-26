#!/bin/bash
export PYTHONPATH=$(pwd)

# Check the input argument and execute the corresponding python script
if [ "$1" == "preprocessing" ]; then
    python ./data/bow_feature_generation.py --mode train --corpus fnc-1
    python ./data/baseline_feature_generation.py --mode train --corpus fnc-1
    python ./data/topic_feature_generation.py --mode train --corpus fnc-1
elif [ "$1" == "selection" ]; then
    if [ "$2" == "embedding" ]; then
        python ./scripts/word_embedding_selection.py
    elif [ "$2" == "feature" ]; then
        python ./scripts/feature_selection.py
    elif [ "$2" == "layer" ]; then
        python ./scripts/structure_improvement.py
    elif [ "$2" == "hyperparameter" ]; then
        python ./scripts/hyperparameter_tuning.py
    else
        echo "Invalid second argument for 'selection'. Use 'embedding', 'feature', 'layer', or 'hyperparameter'."
    fi
elif [ "$1" == "train" ]; then
        python ./scripts/train.py --subset full
        python ./scripts/train.py --subset partial
elif [ "$1" == "test" ]; then
        python ./scripts/test.py --subset full
        python ./scripts/test.py --subset partial
else
    echo "Invalid first argument. Please use 'preprocessing', 'selection', 'train', or 'test'."
fi
