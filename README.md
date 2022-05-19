# NLP_Assignment_2

## Authors:

Group 42:

- Leo Classon
- Mitchell Dior Lobbes
- Paola Feil


## Project Description:

This NLP project was developed as part of the NLP Technology Masters course at the Vrije Universiteit Amsterdam. We work with the OLIDv1 dataset, which contains 13,240 annotated tweets for offensive language detection. This dataset was used in the SemEval 2019 shared 
task on offensive language detection (OffensEval 2019).

We will focus on Subtask A (identify whether a tweet is offensive or not). We preprocessed the 
dataset so that label ‘1’ corresponds to offensive messages (‘OFF’ in the dataset description 
paper) and ‘0’ to non-offensive messages (‘NOT’ in the dataset description paper).


## Instructions to run the code:

- The code for tasks 1-2 can be found in "PartA_Task1-2.py" and can be run without further changes
- The code for tasks 3-7 can be found in "PartA_Tasks3A_&_PartB_Tasks4-7.ipynb". To run this code, please open the file in Google Colab, set the settings of the notebook to using the GPU, and make sure to upload the submitted code including the "TrainedModel" folder (therefore ignoring cells related to connecting the notebook to Google Drive). When loading the model, please change the path to the model as specified in the notebook.