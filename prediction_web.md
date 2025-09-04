
I have always been a soccer/football fan.  Most recently I have become a Premier League addict! Being that I alway love to dive into data, I started looking for data and landed at fbref.com.  I could build customer searches against decades of data.  Getting that data locally in a file I could use with my code was tideous, but I got it done.  As I started to explore the data I realized this would be perfect for a classification algorithm.  What I wanted to do is use an algorithm to predict the outcome of a future match. Let's use ML to solve this!

___

# Project Overview

### Context

Okay, so honestly my son and I started making guesses on the results at a game.  Some of our own predictions were based on solid news, but mine was mostly a gut feeling and which team was higher on the table.  I wanted a better way to predict the results.

The overall aim of this work is to accurately predict the *result* (Win, Lose, or Draw) for future matches.  I ended up with results that are probabilities of Home Win, Tie, and Away Win.  From the probabilities I could with some simple averaging predict an outcome most likely to happen with a good level of confidence.

To achieve this, I looked to build out a predictive model that will find relationships between 50+ statistical features and the actual *result*.  From that learning the algorithm would make probabilities of outcomes.

### Actions

Since I had already compiled the dataset to start with my first action was to determine what model would be best.  This meant several separate programs that implemented different types of models and collecting accuracy values of the trained models.  This would direct me to a good model to use.

As I wanted to predict a categorical output (W, L, or D), I ruled out regression models, and focused on classification models:

* Linear Regression
* Decision Tree
* Random Forest

Random Forest is my favorite so I went with that.

Next I will follow the standard steps of building, training, testing, and deploying a model, the very one being used in this web app.

1. Data cleaning
2. Model trainining
3. Assessing the model
4. Feature engineering
5. Retrain model with features that are most important
6. Set up and build web app
8. build the part of the web app that makes use of the model

___

# Data Overview

Admittedly, I started with too many variables (50+) which made understanding feature importance or even creating some basic plots challenging.  That gets sorted out when I tackle feature engineering and selection.

Using pandas in Python, I created a single dataset that I could use for project.

```python

#########################################
# Import required Python packages
#########################################

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV


#########################################
# Import sample data
#########################################

# Import

data_for_model = pd.read_csv("data/match_data_new.csv")

# Drop obvious variables that wont be needed for classification
data_for_model.drop(['Unnamed: 10', 
                     'Comp', 
                     'Rk', 
                     'Date', 
                     'Team',
                     'Opp', 
                     'GD',
                     'GA', # cut here
                     ], axis=1, inplace = True)

```

As you can see I dropped some variables that seem useful.  However, later on in these steps I discovered there was massive data leakage.  By that, I mean variables that directly related to the target variable.  Specifically, Goal Differential (GD) and I needed to pick either Goals Against (GA) or Goals For (GF).  Also, Team and Opponent names were not required for classification training.


# Random Forest

We will again utlise the scikit-learn library within Python to model our data using a Random Forest. The code sections below are broken up into 3 key sections:

* Data Preprocessing
* Model Training
* Performance Assessment

### Data Preprocessing

Random Forests, just like Decision Trees, are not susceptible to the effects of outliers, and highly correlated input variables, so the required preprocessing here is lighter. Fortunately, the data came with no missing values, and for Random Forest feature scaling would not be necessary. The focus was to get the model trained on the starting datatset and process for feature selection based on the importance of features in the model.


##### Split Out Data For Modelling

In exactly the same way done for any modelling, in the next code block we do two things, we firstly split our data into an **X** object which contains only the predictor variables, and a **y** object that contains only our dependent variable.

Once we have done this, we split our data into training and test sets to ensure we can fairly validate the accuracy of the predictions on data that was not used in training.  In this case, we have allocated 80% of the data for training, and the remaining 20% for validation.

```python

# split data into X and y objects for modelling
X = data_for_model.drop(["Results"], axis = 1)
y = data_for_model["Results"]

# split out training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

```
Next I performed a grid search to find the optimal hyperparameters for the model.

```python
#########################################
# Evaluate for best parameters with grid search
#########################################

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [3, 5, 7, 9],
    'max_features': [2, 3, 5, 9, 11]
}

# Create model and grid search
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit and find best parameters
grid_search.fit(X, y)

# Get best parameters
best_params = grid_search.best_params_
best_score = grid_search.best_score_
best_model = grid_search.best_estimator_

best_params
Out[10]: {'max_depth': 5, 'max_features': 11, 'n_estimators': 50}
```
The best_parms output is more interesting to this process and I put those to good use.


### Model Training

Instantiating and training our Random Forest model is done using the below code.  I use the *random_state* parameter to ensure I get reproducible results, and this helps me understand any improvements in performance with changes to model hyperparameters.

I set the other parameters as perscribed above, meaning that this random forest will be made up of 50 decision trees, all with a max depth (splits) of 5 and max features in the leaf nodes of 11.  I am not out of the woods yet becuase the best score for these settings was 1.0 - a perfect accuracy, Impossible!

```python

# instantiate our model object
clf_full = RandomForestClassifier(random_state=42, 
                                  n_estimators=50, 
                                  max_depth=5, 
                                  max_features=11)

# fit our model using our training & test sets
clf.fit(X_train, y_train)

```



### Model Performance Assessment 

These predictions are on the model trained with the full dataset with now feature selection done yet.  That is coming up soon.

##### Predict On The Test Set

To assess how well the model is predicting on new data - I used the trained model object (here called *clf*) and ask it to predict the *results* variable for the test set

```python

# predict on the test set
y_pred = clf.predict(X_test)

# calculate the confusion matrix and plot
conf_matrix = confusion_matrix(y_test, y_pred_class)

plt.style.use("seaborn-v0_8-poster")
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for (i, j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j, i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()
```

The results were discouraging at best, so I wont waste the ink, to share.  It did however lead me to some interesting steps and findings.

(see image at the bottom of this write up)

This confirmed the best accuracy score of 1.0.  Trying different hyperparameters would be ridiculous.  So, I started to think usually, I would see these results if I had not specified the stopping criteria in the form of the hyperparameters.  It must be something else.  I turned back the data.  Yes, I still needed to do feature selection, but there was something else.

I had inadvertantly included features that directly related to the output variable.  In otherwords, these input variables could vary well be the output variable.  This is called data leakage.  I found the culprets and removed them.  After reprocessing, the grid search found different parameters to use number of decisions trees is now 500, max depth will be 7, and max features is 11. The results showed a better confusion matrix - encouraging!

(see image at the bottom of this write up)


### Feature Importance 

The approach I decided to use, as I feel it is more reliable then any other is called **Permutation Importance** which cleverly uses some data that has gone *unused* at when random samples are selected for each Decision Tree (this stage is called "bootstrap sampling" or "bootstrapping")

These observations that were not randomly selected for each Decision Tree are known as *Out of Bag* observations and these can be used for testing the accuracy of each particular Decision Tree.

For each Decision Tree, all of the *Out of Bag* observations are gathered and then passed through.  Once all of these observations have been run through the Decision Tree, we obtain an accuracy score for these predictions, which in the case of a regression problem could be Mean Squared Error or r-squared.

In order to understand the *importance*, we *randomise* the values within one of the input variables - a process that essentially destroys any relationship that might exist between that input variable and the output variable - and run that updated data through the Decision Tree again, obtaining a second accuracy score.  The difference between the original accuracy and the new accuracy gives us a view on how important that particular variable is for predicting the output.

*Permutation Importance* is often preferred over *Feature Importance* which can at times inflate the importance of numerical features. Both are useful, and in most cases will give fairly similar results.


```python

# calculate permutation importance
result = permutation_importance(clf_full, X_test, y_test, n_repeats=10, random_state=42)

permutation_importance = pd.DataFrame(result["importances_mean"])
feature_names = pd.DataFrame(X.columns)
permutation_importance_summary = pd.concat([feature_names, permutation_importance], axis=1)
permutation_importance_summary.columns = ["input variable", "importance"]
permutation_importance_summary.sort_values(by = "importance", inplace=True)

# plot permutation importance
plt.barh(permutation_importance_summary["input_variable"],permutation_importance_summary["permutation_importance"])
plt.title("Permutation Importance of Random Forest")
plt.xlabel("Permutation Importance")
plt.tight_layout()
plt.show()

```

That code gives a *Permutation Importance* plot!


(see chart at the end of this report


The overall story the chart tells is that very few of the variables I was using were most important or impactful input.  I was relieved because fewer variables is just easier all around.

I did the process above again and finally came up with a model I was happy with.

___

# Growth & Next Steps

From a data point of view, trying a completely different set of variables not present in any of the processing above could prove to better accuracy.

The deployment of the model with the web app proved very useful and while test the web app I discovere a few small issues with the data like predicting on team names that don't exist in the dataset because the name was slightly different.