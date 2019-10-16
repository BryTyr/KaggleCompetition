# KaggleCompetition
This is a sequential model I built to predict the values of income based on selected features

for preprocessing I used a mix of:

-One hot encoding  
-Back-fill/mean-fill of missing values  
-irrelevant feature pruning  
-Data visualisation (initData.html and initDataProcessed.html)  
-Outlier removal  
-feature normalization  

Model:
I used Keras Sequential model with 3 layers two of which had 64 nodes and third have 32 nodes, an adam optimizer and an early callback with a patience of 10 epochs.

I would like to give credit to this10 epochs tutorial as it gave me great insight on how to build the model:
https://www.tensorflow.org/tutorials/keras/regression

I achived a RSME of 61,234 on my final submission.
