![UTA-DataScience-Logo](https://user-images.githubusercontent.com/89792487/207612409-6342a227-58d1-41a8-8277-96d45e92b5e5.png)

# BillionWordImputation

* This repository holds all attempts made/models tried to complete the [Billion Word Imputation Kaggle Challenge](https://www.kaggle.com/competitions/billion-word-imputation/overview). 

## Overview

  The task defined in the Kaggle challenge is, given a "billion" words forming senctences pulled from the [Chelba et al.](https://arxiv.org/abs/1312.3005) training dataset and a testing dataset holding a lesser amount of senctences with one word removed from each, you are to train something (no suggestions given) to both find the location of the missing word and replace it with the correct word. My approach to this challenge began with omitting the task of locating the index of the missing word because the two ways to complete this subtask (N-gram Model or Word Distance Statistics) were too computationally expensive for my machine and difficult to properly implement in a pipeline. My next step was to take my training set and create my own set of missing words where the indicies were given by either a "[MASK]" symbol or a  "103" symbol. 
  
  After this new training set was created, I could now approach the task of filling in the correct word using a Masked Language Model (MLM). Three strategies were attempted to use MLM: [Masked Language Modeling with BERT](https://keras.io/examples/nlp/masked_language_modeling/), [Masked Language Modeling with BERT and HuggingFace](https://www.youtube.com/watch?v=R6hcxMMOrPE&ab_channel=JamesBriggs), and [Next Word Prediction BI-LSTM](https://www.kaggle.com/code/ysthehurricane/next-word-prediction-bi-lstm-tutorial-easy-way).
  
  My best and only working model (Masked Language Modeling with BERT) was able to predict masked words with probabilities on the order of 0.02 +-0.02. Comparing to the scores attained on the Kaggle leaderboard was not applicable because I could not make a submission, given I had to greatly alter the challenge and my approaches.

## Summary of Work Done

### Data

* Data:
  * Type:
    * Input: Full sentences in English ranging from 3 to 25 words
    * Input: Sentences with approximately 15% of their words masked with placeholders.
  * Size: 4.2GB
  * Instances (Train, Test, Validation Split): 5000 sentences for training, and 1250 masked sentences for testing. Giving an 80/20 split. 

#### Preprocessing / Clean up

* Skim the appropriate amount of sentences off the top of each dataset, format into lists, remove punctuation and change all cases to lowercase, then place into dataframe.

#### Data Visualization

* Next Word Prediction BI-LSTM (epochs, accuracy)


![image](https://user-images.githubusercontent.com/89792487/207628043-1918c5e8-91f4-4a0e-abc8-4df3b1acc57b.png)

* Next Word Prediction BI-LSTM (epochs, loss)


![image](https://user-images.githubusercontent.com/89792487/207628069-6b883822-2118-4fe1-ac69-01378ce976dd.png)

### Problem Formulation

  For both the Masked Language Modeling with BERT and Masked Language Modeling with BERT and HuggingFace models, the input was the unedited full sentences and the expected output was "unmasked" sentences from the test set. Only the Masked Language Modeling with BERT model was able to give me some kind of tangible output. The other model would kill my kernel every time an attempt was made to train the model, regardless of how many parameters I move around.
  
  For the Next Word Prediction BI-LSTM model the input was sentences that were "chopped off" at the index of the masked word. The input is a bit different because the expectation is for the model predict the next word in the sentence, which can then be spliced back with the remainder of the sentence. The furthest I could get with this model was for it to return to me a list of words it "believed" could be the next word, but I could not get it to actually choose a word.

### Training

  The training was set to run on CPU for the Masked Language Modeling with BERT and Masked Language Modeling with BERT and HuggingFace models, and the last model was not set to run on any device in particular. Training for all models took upwards of 9 hours before any edits were made to the training/testing set, after chopping down, training took around 45 minutes. The only training curves I was able to extract were for the Next Word Prediction BI-LSTM model, and they looked to be very linear because of my machines inability to complete a sufficient amount of epochs without killing the kernel. The decision to stop training was highly regulated by how many epochs my machine could handle. I did try Google Collab to see if I could offload the computaional power needed to a remote computer, and a I was met with a similar demise.

### Conclusions

* Given my attempts did not end in any comparable and inferable results, I will state that MLM pipelined with either the N-gram model or the Word Distance Statistics (WDS) to locate the missing word would be the most effective route.

### Future Work

* Next steps would be to take another stab at integrating the missing word locating models into the picture and look more into HuggingFace's pretrained embedders.

## How to reproduce results

*Results in their current state are not ideal to reproduce.*

### Overview of files in repository

* Describe the directory structure, if any.
* List all relavent files and describe their role in the package.
* An example:
  * utils.py: various functions that are used in cleaning and visualizing data.
  * preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
  * visualization.ipynb: Creates various visualizations of the data.
  * models.py: Contains functions that build the various models.
  * training-model-1.ipynb: Trains the first model and saves model during training.
  * training-model-2.ipynb: Trains the second model and saves model during training.
  * training-model-3.ipynb: Trains the third model and saves model during training.
  * performance.ipynb: loads multiple trained models and compares results.
  * inference.ipynb: loads a trained model and applies it to test data to create kaggle submission.

* Note that all of these notebooks should contain enough text for someone to understand what is happening.

### Software Setup (Packages)

* Pandas
* Numpy
* Pytorch
* HuggingFace Transformers
* Keras
* Tensorflow
* tqdm


## Citations

* [Locating and Filling Missing Words in Sentences](https://stlong0521.github.io/20160305%20-%20Missing%20Word.html)
* [Using pre-trained word embeddings](https://keras.io/examples/nlp/pretrained_word_embeddings/)
* [End-to-end Masked Language Modeling with BERT](https://keras.io/examples/nlp/masked_language_modeling/)
* [Keras NLP](https://keras.io/keras_nlp/)
* [HuggingFace](https://huggingface.co/)
* [Next Word Prediction BI-LSTM](https://www.kaggle.com/code/ysthehurricane/next-word-prediction-bi-lstm-tutorial-easy-way)
* [Pinecone NLP](https://www.pinecone.io/learn/nlp/)
* [Tensorflow Hub Bert](https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3)
* [Training Bert](https://www.youtube.com/watch?v=R6hcxMMOrPE&ab_channel=JamesBriggs)
* [Building a Next Word Predictor in Tensorflow](https://towardsdatascience.com/building-a-next-word-predictor-in-tensorflow-e7e681d4f03f#:~:text=Next%20Word%20Prediction%20or%20what,or%20emails%20without%20realizing%20it.)






