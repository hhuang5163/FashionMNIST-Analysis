# MNIST-Analysis
K-fold cross-validation implemented from scratch to aid in analysis of the MNIST dataset<br>
It was determined that the left side of the numbers 3 and 8 are the important features used in differentiating between handwriting samples of the numbers 3 and 8. In the Hard2ClassifyData folders, we can see that samples with light handwriting are typically hard to classify as well as "scrunched" handwriting, where the lower part of 3 overextends to almost make it look like an 8. An example is below:
<img src="https://github.com/hhuang5163/MNIST-Analysis/blob/main/Hard2ClassifyDataLogReg/Uncertain422.png">

## Results
Important pixels to differentiate between a 3 and an 8 as determined by Logistic Regression (brighter pixels means more important)
<img src="https://github.com/hhuang5163/MNIST-Analysis/blob/main/LogImportantFeatures.png">
<br><br>
Important pixels to differentiate between a 3 and an 8 as determined by Linear SVM (brighter pixels means more important)
<img src="https://github.com/hhuang5163/MNIST-Analysis/blob/main/SVMImportantFeatures.png">
<br><br>
The folders Easy2ClassifyData[MLmodel] and Hard2ClassifyData[MLmodel] contain examples of handwriting that is easy for the ML model to classify and hard for the ML model to classify, respectively.
## To run
<ol>
  <li>Unzip the MNIST.zip file to obtain the MNIST dataset. Ensure the unzipped folder remains in the same folder as the file MNIST.zip.
  <li>To replicate the results, simply open a file and run.
    <ul>
      <li><b>cv_builtin.py</b> will fit a Logistic Regression and Linear SVM model to the data and print the tuned hyperparameters as determined by scikit-learn methods.
      <li><b>cv_scratch.py</b> will run the cross validation implemented from scratch. Feel free to change the number of folds K, which is defined as a global variable to experiment with the method.
      <li><b>important_features.py</b> will use the best lambda (can be changed in the code) as determined by cv_scratch.py to find the pixels in an image that are important in differentiating between a 3 and an 8.
    </ul>
</ol>
