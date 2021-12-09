import pandas as pd
from .preprocessing import asel_mebaysan_preprocess
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score


def get_test_data(na='group', fix_outlier=True):
    df = pd.read_csv('./data/test.csv').drop(['Unnamed: 0','id'], axis=1)
    df = asel_mebaysan_preprocess(df, na, fix_outlier)
    X = df.drop('satisfaction_satisfied', axis=1)
    y = df['satisfaction_satisfied']
    return df, X, y


def predict(X, y, model):
    y_pred = model.predict(X)
    return {
        'model':model,
        'accuracy': accuracy_score(y, y_pred),
        'f1-score': f1_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'precision': precision_score(y, y_pred)
    }
