The following .py files work for generating network input:
Data Preparation.py file is designed for obtaining the concatenation representation of headline-article pairs and its corresponding stance.
The generated .csv file can be found at the dataset  file of fnc-1 file. 
bow_feature_generation.py, baseline_feature_generation.py, topic_feature_generation.py are utilized to carry on the the generation of 1 bow feature,
4 baseline features and 4 topic model features and combine each type of features into one single .pkl file.

The following .py file works for evaluation metrics:
FNC_score.py works as an external package designed as the evaluation metrics.

The following .py files work for model selection process: 
Attention-LSTM-Word-Embedding-Selection.py for selecting word vector types. Attention-StackLSTM-Feature-Selection 
for choosing help feature set. Attention-StackLSTM-Structure-Improvement.py for the application of bidirectional LSTM, the selection of number of LSTM hidden layers and number 
of Fully connected layers. Attention-StackLSTM-Hyperparameter-Tuning.py for tuning the learning rate and dropout rate.

The following .py files work for model training: 
Attention-StackLSTM-Model-Training-Full-Dataset.py for training the Attention-StackLSTM classifier designed for 4 categories.
Attention-StackLSTM-Model-Training-Partial-Dataset.py for training the Attention-StackLSTM classifier designed for agree and disagree class.  
StackLSTM-Model-Training-Partial-Dataset.py for training the StackLSTM classifier designed for agree and disagree class.
Attention-LSTM-Model-Training-Partial-Dataset.py for training the Attention-LSTM classifier designed for agree and disagree class.

The following .py files work for model evaluation through voting scheme: 
Attention-StackLSTM-Model-Voting-For-Test-Full-Dataset.py for evaluating the Attention-StackLSTM classifier designed for 4 categories.
Attention-StackLSTM-Model-Voting-For-Test-Partial-Dataset.py for evaluating the Attention-StackLSTM classifier designed for agree and disagree class.  
StackLSTM-Model-Training-Voting-For-Test-Partial-Dataset.py for evaluating the StackLSTM classifier designed for agree and disagree class.
Attention-LSTM-Model-Voting-For-Test-Partial-Dataset.py for evaluating the Attention-LSTM classifier designed for agree and disagree class.
The voting result will be generated as a single .csv file named voting_result.csv