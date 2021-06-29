LDA
===

LDA linear transformation of input data.

Inputs
------

* Data: input dataset
* Test Data: input dataset

Please make sure to load only Data into the widget and add Test Data after LDA has finished processing Data!If you have this widget incorporated into a workflow
ensure to clear the Input channels before reusing them.

Outputs
-------

* LDA: lda model from Data.

Explanation
------------

Linear discriminant analysis (LDA) computes the LDA linear transformation of the input data and displays the scatter 
plot of the scores of the discriminant functions within the "Scatterplot" tab and the 
confusion matrix of the lda model\'s classification performance within the "Confusion matrix" tab. 

.. image:: icons\\LDAscatter.png
  :width: 600
  :alt: Alternative text

1. Select which discriminant functions (DF) has to be added to the graph.
2. You can decide whether to color your data points by class or not and whether to see a legend or not.
3. When Apply Automatically is ticked, the widget will automatically communicate all changes. Alternatively, click Apply.

.. image:: icons\\LDAconf.png
  :width: 600
  :alt: Alternative text

Within the "Confusion matrix options" box are several evaluation modes for the confusion matrix calculation.

* if "Test on train data" is klicked, the confusion matrix regarding the prediction of the whole Data is calculated while 
  Data serves to build the lda model.
* if "Test on test data" is klicked, Data serves to build the lda model and Test data\'s class membership will be predicted. 
  By comparison of predictions with true class membership the confusion matrix is calculated.
* If "Cross validation" is klicked, Data gets devided into the specified number of folds. Every fold once is removed from Data
  while the remaining Data builds the lda model. The removed fold\'s class membership is predicted and by comparison the confusion matrix
  is calculated. After every fold gets predicted once, the mean of the confusion matrix is presented in the widget.
* If "Stratified cross validation" is klicked, the same calculations take place as if "Cross validation" is klicked but the algorithm
  ensures that every fold contains every class in equivalent numbers.
* If "Leave one out" is klicked, cross validation takes place but every fold contains only one element. Every element is predicted once.

