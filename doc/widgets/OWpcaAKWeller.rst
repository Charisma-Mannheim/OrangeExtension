PCA
===

PCA linear transformation of input data.

Inputs
------

* Data: imput dataset
* Test Data: imput dataset

Please make sure to load only Data into the widget and add Test Data after PCA has finished processing Data! This widget fails when reloading a workflow. If you have this widget incorporated into a workflow
ensure to clear the Input channels before reusing them.

Outputs
-------

* Data: summary of input Data and pca transformed data.
* Scores: pca transformed data.
* Test Scores: pca transformed Test Data by pca model obtained from Data.
* Loadings: Eigenvectors.
* Explained variance: explained variance per principal component.
* Explained variance cumulative: summed explained variance per principal component
* Reconstruction error of cv row wise: sum of root mean squared error per used number of principal components for 10-fold cross validation.
* Reconstruction error of cv by Eigenvector: sum of root mean squared error per used number of principal components for 10-fold cross validation without overfit.
* Outlier: samples which have Q- or/and T²-values higher then the confidence level and therefore are considered outliers.
* Outlier corrected data: original data set without the outliers.
* PCA: pca model from Data

Explanation
------------

Principal Component Analysis (PCA) computes the PCA linear transformation of the input data and displays the scree 
plot of explained variances within the "Scree" tab, the 
root mean squared errors (rmse) of cross validation within the "Error" tab and the Q-T²-Plot within the "Q residuals vs. Hotellings T squared" tab. 

.. image:: icons\\PCAexplVar.png
  :width: 600
  :alt: Alternative text

1. Select how many principal components you wish in your output. It is best to choose as few as possible with variance covered as high 
   as possible. You can also set how much variance you wish to cover with your principal components.
2. You can standardize data to adjust the values to common scale or just mean-center it.
3. When Apply Automatically is ticked, the widget will automatically communicate all changes. Alternatively, click Apply.
4. Produce a report.
5. Principal components graph, where the red (lower) line is the variance covered per component and the green (upper) line is cumulative variance covered by components.

.. image:: icons\\PCArmsec.png
  :width: 600
  :alt: Alternative text

Principal components graph, where the red line is the rmsecv of cross validation row wise per component and the blue line is the rmsecv of cross validation by Eigenvector per component used.
The number of components of the transformation can be selected either in the "Components Selection" input box or by dragging the vertical cutoff line in the graph.


.. image:: icons\\PCAQTplot.png
  :width: 600
  :alt: Alternative text

To ensure that the PCA model is not skewed by outliers or to detect abnormal behavior in processes two metrics can be used for outlier detection purposes. 
The Q-residual and the T squared-metric. The underlying principle for using those metrics is that outliers are not as good described by the model as normal data points. 
They represent two kinds of errors; The Q-residuals represents the residuals that remain after a specific number of Components is used. Thus, they describe the left-over 
variations that are not explained by the PCA-Model. Since outliers are not as well described as other data points they will have larger Q-residuals.
The T squared metric covers the variations within the model. Where low scores account for a better fit then high scores. The horizontal line represents the confidence level of the
Q-Metric, the vertical line represents the confidence level of the T squared-metric.


Examples
--------

PCA is often used for data reduction purposes when handling large datasets. Below, we used the Irisdataset to show how we can improve the visualization of the dataset with PCA. 
The transformed data in the Scatter Plot show a much clearer distinction between classes than the default settings.
On the left the desired confidence level can be adjusted.

.. image:: icons\\PCAexample.png
  :width: 800
  :alt: Alternative text

The widget provides several outputs: scores and loadings. Scores are weights for individual instances in the new 
coordinate system, while loadings are the system descriptors (weights for principal components). Loadings are useful to decide which original features could be important. 
Explained variance and cumulative explained variance are as well as rmse and rmsecv helpful tools to decide how many principal components are necessary to describe the original data sufficiently.




*Disclaimer: parts of this documentation page are taken from the documentation page of orange\'s original pca widget which served as template for this pca widget. For teaching purposes only!*




