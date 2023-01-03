Feature Generation:
1.Data Preparation.py file is designed for obtaining the concatenation representation of headline-article pairs and its corresponding stance. The generated .csv file can be found at the dataset  file of fnc-1 file. 
2.bow_feature_generation.py, baseline_feature_generation.py, topic_feature_generation.py are utilized to carry on the the generation of 1 bow feature,
  4 baseline features and 4 topic model features and combine each type of features into one single .pkl file.

Evaluation metrics:
1.FNC_score.py works as an external package designed as the evaluation metrics.

Model selection process: 
1. Attention-LSTM-Word-Embedding-Selection.py ---->selecting word vector types. 
2. Attention-StackLSTM-Feature-Selection.py ----> choosing helpful feature set. 
3. Attention-StackLSTM-Structure-Improvement.py -----> applying bidirectional LSTM, the selection of number of LSTM hidden layers and number 
of Fully connected layers. 
4. Attention-StackLSTM-Hyperparameter-Tuning.py -----> tuning the learning rate and dropout rate.

Model training: 
1. Attention-StackLSTM-Model-Training-Full-Dataset.py ----> training the Attention-StackLSTM classifier designed for 4 categories.
2. Attention-StackLSTM-Model-Training-Partial-Dataset.py -----> training the Attention-StackLSTM classifier designed for agree and disagree class.  
3. StackLSTM-Model-Training-Partial-Dataset.py ----> training the StackLSTM classifier designed for agree and disagree class.
4. Attention-LSTM-Model-Training-Partial-Dataset.py ----> training the Attention-LSTM classifier designed for agree and disagree class.

Model Evaluation through voting scheme: 
1. Attention-StackLSTM-Model-Voting-For-Test-Full-Dataset.py ----> evaluating the Attention-StackLSTM classifier designed for 4 categories.
2. Attention-StackLSTM-Model-Voting-For-Test-Partial-Dataset.py ----> evaluating the Attention-StackLSTM classifier designed for agree and disagree class.  
3. StackLSTM-Model-Training-Voting-For-Test-Partial-Dataset.py ----> evaluating the StackLSTM classifier designed for agree and disagree class.
4. Attention-LSTM-Model-Voting-For-Test-Partial-Dataset.py ----> evaluating the Attention-LSTM classifier designed for agree and disagree class.
5. The voting result will be generated as a single .csv file named voting_result.csv


The full data and extracted features can be found here (fnc-1 and .vecter_cache file): https://1drv.ms/u/s!AvFAYFbdSN6phmo78TMOyUO-cu2-?e=24LfL2
