import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## reusing code from week 09 homework

def normalize_z(df):
    '''Input df, output dfout
    Z Normalizes a dataframe'''
    
    dfout = df.copy()
    dfout = (dfout - df.mean(axis=0)) / df.std(axis=0)
    
    return dfout

# def min_max

def get_features_targets(df, feature_names, target_names):
    '''input: df, ['feature_names'], ['target_names']
    output: df_feature, df_target'''
    
    df_feature = df[feature_names]
    df_target = df[target_names]
    
#     print(type(df_feature))
    
    return df_feature, df_target


def prepare_feature(df_feature):
    '''prepares the feature
    input: df_feature \n output: prepared_df_feature'''
    
    # get the number of columns, number of features
    cols = len(df_feature.columns)
    
    # shape the feature columns
    # why -1 for row? Transpose
    feature = df_feature.to_numpy().reshape(-1, cols)
    rows = feature.shape[0] # get number of rows
    
    # create our ones
    ones = np.ones((rows, 1))
    
    X = np.concatenate((ones, feature), axis=1)
    
    return X


def prepare_target(df_target):
    '''prepares the target
    input: df_target
    output: prepared_target'''
    
    cols = len(df_target.columns)
    target = df_target.to_numpy().reshape(-1, cols)
    return target


def predict(df_feature, beta):
    
    # normalize and prepare the feature
    X = prepare_feature(normalize_z(df_feature))
    
    return calc_linear(X, beta)


def calc_linear(X, beta):
    '''input: X, beta output: y= X cross B'''
    
    return np.matmul(X, beta)


def gradient_descent(X, y, beta, alpha, num_iters):
    '''input: X, y, beta, alpha, num_iters
    output: beta, J_storage(list of all costs computed with using beta)'''
    
    # get number of rows
    m = X.shape[0]
    
    # initialize J storage which stores all the successive iterations of cost function
    J_storage = np.zeros((num_iters, 1))
    
    # compute cost and store in J_storage each succesive iteration of cost function
    for n in range(num_iters):
        deriv = np.matmul(X.T, (calc_linear(X, beta) - y))
        beta = beta - alpha * (1/m) * deriv
        J_storage[n] = compute_cost(X, y, beta)
        
    return beta, J_storage


def compute_cost(X, y, beta):
    
    n = X.shape[0]
    
    # error = yhat - y
    error = calc_linear(X, beta) - y
    error_sq = np.matmul(error.T, error) # transpose the error matrix
    # multiply by itself
    
    J = (1/(2*n)) * error_sq
    
    return J


def split_data(df_feature, df_target, random_state=100, test_size=0.3):
    '''splits data into training and testing
    input: df_feature, df_target, random_state=100, test_size=0.3
    output: df_feature_train, df_feature_test, df_target_train, df_target_test'''
    np.random.seed(random_state)
    n = len(df_feature)
    
    test_num = int(n * (test_size))
    
    test_idx = np.random.choice(n, test_num, replace=False) # all the indexes of the test indexes
    train_idx = [i for i in range(n) if i not in test_idx]
    
    df_feature_test = df_feature.iloc[test_idx]
    df_feature_train = df_feature.iloc[train_idx]
    
    df_target_test = df_target.iloc[test_idx]
    df_target_train = df_target.iloc[train_idx]
    
    return df_feature_train, df_feature_test, df_target_train, df_target_test
  
    
def r2_score(y, ypred):
    '''finds the R^2 score
    input y(np_array) actual values, ypred(np_array) predicted values
    output R^2'''
    
    n = len(y)
    
    y_bar = np.mean(y)
    
    # SS_res = summation (true_y - y_pred)**2
    SS_res = 0
    for true_y, pred_y in zip(y, ypred):
        SS_res += (true_y - pred_y)**2
    
    # SS_tot = summation (true_y - y_bar)**2
    SS_tot = 0
    for true_y in y:
        SS_tot += (true_y - y_bar)**2    
    
    r_2 = 1 - (SS_res / SS_tot)
    
    return r_2
    pass


def mean_squared_error(target, pred):
    '''finds the mean squared error
    input target(np_array), pred(np_array)
    output mean_squared_error'''
    
    n = len(target)
    
    # summation of (actual_y - y_cap)**2
    MSE = 0
    for actual_y, pred_y in zip(target, pred):
        MSE += (actual_y - pred_y)**2
    MSE /= n
    
    return MSE
    pass