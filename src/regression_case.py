import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scs
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split

df = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
# pd.scatter_matrix(df)
# plt.show()

X_cols = ['SalesID','YearMade']

def load_data(df, X_cols, test=False):
    X = df[X_cols]
    if test == False:
        y = df['SalePrice']
        f = pd.concat([X, y], axis=1)
    # Removing all years that are 1000
        f = f[f['YearMade']>1000]
    # Regrouping data
        X = f[X_cols]
    if test == False:
        y = f['SalePrice']

        X_trainf, X_testf, y_trainf, y_testf = train_test_split(X, y, test_size=0.20, random_state=42)
        X_train = X_trainf['YearMade'].reshape(-1,1)
        X_test = X_testf['YearMade'].reshape(-1,1)

        return X_train, X_test, y_trainf, y_testf
    else:
        X = X[X['YearMade'] >1000]
        return X

def fit_model(X_train, X_test, y_train, y_test):
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train,y_train)
    return model

X_train, X_test, y_train, y_test = load_data(df,X_cols)
model = fit_model(X_train, X_test, y_train, y_test)

score = model.score(X_train, y_train)


print "Score: {}".format(score)

def predict_mean(y):
    '''
    Input: y-data from training data (numpy array y)
    Output: mean predictions (numpy array y_pred)
    '''
    y_pred = np.mean(y)
    return y_pred

def submit(df_test, model):
    x = load_data(df_test,X_cols, test=True)
    y_pred = model.predict(X_test)
    y_pred = pd.Series(y_pred, name='SalePrice')
    print len(x['SalesID']), len(y_pred)
    results_df = pd.concat([x['SalesID'],y_pred], axis=1)

    return results_df.to_csv('data/your_predictions.csv',index=False)

submit(df_test, model)
