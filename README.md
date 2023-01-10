# Programming-for-Data-Analysis-2

##### *Name: Megan O'Donovan*
##### *Email: G00411435@gmit.ie*

### Aim
Our objective is to identify which features are most helpful in predicting malignant or benign cancer and to classify whether the breast cancer is benign or malignant. The dataset being used in this report was published here. This data set was created by Doctor William H. Wolberg, physician at the University of Wisconsin Hospital at Madison, Wisconsin,USA. Breast Cancer occurs as a result of abnormal growth of cells in the breast tissue, commonly referred to as a Tumour, which can be benign (not cancerous) or malignant (cancerous). Dr. Wolberg recorded results from 10 features, calculating the mean value, extreme  or worst value and standard error from each feature. Additional patient information was also recorded in the dataset: Patient ID and Diagnosis(M= Malignant, B=Benign)
Ten real-valued features were computed for each cell nucleus:<br>

•	a) radius <br>
•	b) texture <br>
•	c) perimeter<br>
•	d) area<br>
•	e) smoothness <br>
•	f) compactness <br>
•	g) concavity <br>
•	h) concave points <br>
•	i) symmetry<br>
•	j) fractal dimension <br>

### Data Preparation <br>

The dataset contains 596 rows and 32 columns, where each row represents an observation for one patient. The tumour is classified as benign or malignant based on the above features. The dataset is check for null or missing values and columns that cannot be used for classification are removed, ID. <br>

### Exploratory Data Analysis (EDA) <br>
Exploratory Data Analysis is used as a method of finding patterns and/or relationships to build on future analysis. There are several packages in python that allow for the data to be visually analysed such as Matplotlib and Seaborn. The variables are split into 3 groups containing 10 variables each. The dataset is conveniently ordered into distinct groups: mean, se and worst. <br>

Based on histogram plots, malignant tumours have higher values for most measurements. Radius, texture, smoothness, compactness and concavity are of particular interest. The distribution of benign features, appear further along the x-axis for several of the features, particularly perimeter, area and concavity and radius. Comparison of standard error distribution by malignancy shows that there is no perfect separation between any of the features, with heavy overlapping existing for features symmetry_se and  smoothness_se. There is good separation for concave_points_worst, concavity_worst, perimeter_worst, radius_worse, area_mean, perimeter_mean.  <br>

Assessing the violin plots for features that would be good for classification, where the medians of the diagnosis groups look separated. The median of texture_mean and concave_point_se would be the better choices for classification. The median of symmetry_mean, smoothness_mean and fractal_dimension_mean for Malignant and Benign appear closely related. The  majority of the medians of the features in the violin charts for almost all Malignant or Benign don't vary much for the standard error. The distribution curve for area_se is skewed. The distribution of datapoints for radius_worst, texture_worst and area_worst all appear to be well separated and should be considered for feature classification.  <br>

### Feature Selection <br>

Feature Selection is the process where variables are selected based on the likelihood of most predictive ones, to build our machine learning models. Within datasets, not all features are relevant, removing these features increases the accuracy of the model and reduces the possibility of an over-fitted model. Four possible feature selection methods are examined (correlation, chi-square, random forest and cross validation) and then each reduced dataset is applied to the machine learning model. 

##### Correlation Matrix 
In general, it is recommended to avoid having correlated features. Highly correlated features will, in general not bring additional information, but will increase the complexity of the algorithm, thus increasing the risk of errors. Because of this, variables that have a high correlation will be removed. A correlation matrix summarises the dataset where the goal is to visualise a pattern where features with a high correlation to be identified. The matrix shows values ranging between -1 and 1, for either negative or positive relationships. A result of zero indicates no relationship at all. Features with a correlation higher than .95 will be removed from the dataset. By eliminating highly correlated features we can avoid a predictive bias for the information contained in these features. <br>

##### Chi square <br>
To increase accuracy of our model we need the dataset to have a high chi-squared statistics. Using python libraries SelectKBest and chi2, SelectKBest selects the features with best chi-square,  calculating Chi-square between each feature and the target based on the number of features, k=10 and fit to the training model where 30% of the data is reserved for testing purposes using SciKit-Learn library in Python for the train_test_split method. <br>

##### Random Forest <br>
“Each tree of the random forest can calculate the importance of a feature according to its ability to increase the pureness of the leaves.” After the importance of each feature is known, feature selection occurs using Recursive Feature Elimination. The random forest regressor is fitted to the training dataset, storing the feature importance’s. <br>

##### Cross Validation <br>
The previous two selection methods used a default best fit of k=10 but this is an assumption. Therefor I will use cross validation to not only find best features but also find how many features do I need for best accuracy. Using feature importance to select the best set of features according to RFE with 5 folds for the cross-validation.<br>

### Machine Learning
Feature scaling is used to ensure all features are at the same level of magnitude or range. The goal of this project is to build a model which classifies tumours as benign or malignant. Different types of classification algorithms in Machine Learning are applied to our dataset ( Logistic Regression, SGD classifier and Random Forest). for each feature selection method ( 1. Correlation, 2. Chi-square, 3. Recursive Feature Elimination (RF), 4. Cross-validation). Sklearn’s library was used to import all the methods of classification algorithms and applied to the training dataset. <br>

The same method was applied to each algorithm. We want an ROC curve that sticks as close to the y-axis as possible. Of the 12 ROC curves, the dataset after cross-validation selection produced the best ROC curves for the classification algorithms and the highest AUC  scores. Although a perfect ROC score of 1 was not achieved, the SGD classifier algorithm had a score of .991. Across the feature selected datasets, SGD classifier and random forest scored higher than logistic regression. Taking the average of the accuracies of the three validation sets, the accuracies appear to be about the same for each algorithm, but the SGD classifier algorithm after cross validation produces the highest accuracy score of roughly 97.5%. <br>

The precision and recall scores for each algorithm was checked since the ROC curves are very similar. Precision is the ratio of true positives to the total number of samples classified as positive by the algorithm: <br>
$\frac{TP}{TP + FP}$

Recall is the number of positive samples that were accurately classified as positive to the number of positive samples in the data set: <br>
$\frac{TP}{TP + FN}$  <br>

Since we care more about recall, than precision when classifying malignant or benign tumours, and all the algorithms perform reasonably well with respect to precision, it seems like the SGD classifier or random forest are the best fits for our model after correlation-based selection. <br>

### Summary <br>
* Having built several classification models, I can see that the SGD Classifier algorithm after Cross Validation feature selection gives the best results for our dataset, with roughly 97.5% accuracy using 30 tumour characteristics. <br>
* Machine learning algorithms that performed the best include models SGD Classifier after Cross Validation feature selection and after Random Forest Selection. <br>
* The most predictive features found with recursive feature elimination are mean texture, perimeter and area, and worst texture, perimeter, and area. <br>
* The most predictive features using random forest classification are mean for texture, perimeter and area, area se, and worst area perimeter. <br>

### References:<br>
•	https://towardsdatascience.com/building-a-simple-machine-learning-model-on-breast-cancer-data-eca4b3b99fa3  <br>
•	https://rstudio-pubs-static.s3.amazonaws.com/344010_1f4d6691092d4544bfbddb092e7223d2.html  <br>
•	https://medium.com/@shahid_dhn/building-ml-model-to-predict-whether-the-cancer-is-benign-or-malignant-on-breast-cancer-wisconsin-d6cf8b47f49a  <br>
•	https://github.com/pkmklong/Breast-Cancer-Wisconsin-Diagnostic-DataSet/blob/master/Breast%20Cancer%20Wisconsin%20(Diagnostic)%20DataSet_Orignal_Data_In_Progress.ipynb  <br>
•	https://github.com/mugdhapai/Cancer_Classification/blob/main/Code/cancer_classification.ipynb  <br>
•	https://chriskhanhtran.github.io/minimal-portfolio/projects/breast-cancer.html <br>
•	https://medium.com/@kathrynklarich/exploring-and-evaluating-ml-algorithms-with-the-wisconsin-breast-cancer-dataset-506194ed5a6a <br>
•	https://www.geeksforgeeks.org/box-plot-in-python-using-matplotlib/ <br>
•	https://www.projectpro.io/recipes/select-features-using-chi-squared-in-python  <br>
•	https://towardsdatascience.com/feature-selection-using-random-forest-26d7b747597f  <br>
•	https://www.yourdatateacher.com/2021/10/11/feature-selection-with-random-forest/  <br>


