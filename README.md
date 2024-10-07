<h1>
  EmotionAI: An AI Approach to EEG-based Emotion Recognition using a Consumer Device
</h1>

EmotionAI is a repository where a binary emotion prediction model or numerical predictor can be trained using data recorded with the MUSE EEG-Powered Meditation & Sleep Headband.<br>

In this repository you can find:<br>
-concat -> a python file where is possible to concatenate all your csv files containing the muse registrations.<br>
-eegBandSeparator -> a python file where is possible to convert your raw data in the alph, beta, gamma, delta and theta frequencies.<br>
-eegFreqsConvertr -> a python file where is possible to convert data from a frequency f1 to a frequency f2.<br>
-elaborazione_dati -> a python file where the pre_elaboration and the convertion in feature vectors is done to your dataset.<br>
-fv_generator -> a python file where your pre elaborated data is converted into a feature vector.<br>
-postElaboration -> a python file where the post elaboration of the predictions of the model is done.<br>
-preElaboration -> a python file containing the operations of pre elaboration.<br>
-training_all_subject_post -> the trainer of the binary classification model using a "leave one subject out" approach, with a post elaboration operations.<br>
-training_all_subject_regression -> the trainer of the regression model using a "leave one subject out" approach, with a post elaboration operations.<br>
-training_one_subject -> the trainer of the binary classification model using a k-fold cross validation on a single subject.<br>

