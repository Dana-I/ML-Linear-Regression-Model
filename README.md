# Linear Regression Using Gradient Descent (Part 1)

## Description
This project implements **linear regression from scratch** using **gradient descent** to predict final student grades (`G3`) from the UCI Student Performance dataset. Categorical features are one-hot encoded, and features are standardized.

## How to Run
1. Make sure **Python 3** is installed.
2. Install required packages:
   python3 -m pip install numpy pandas matplotlib scikit-learn ucimlrepo
3. Run the script:
   python3 part1.py

##Part 2 - SGDRegressor model 
   
#description: 
The linear regression is implemented using Scikit-learn's regressor. We are using this model so we can compare it to the results from the model in part 1. 
   
#Libraries used:
 - numpy
 - pandas
 - matplotlib
 - scikit-learn
 - ucimlrepo

#Scikit-learn Classes used:
 - sklearn.linear_model.SGDRegressor
 - sklearn.preprocessing.StandardScaler
 - sklearn.model_selection.train_test_split
   
##How to run Part 2 
1. python3 -m pip install numpy pandas matplotlib scikit-learn ucimlrepo
2. Run the script:
   python3 part2.py
