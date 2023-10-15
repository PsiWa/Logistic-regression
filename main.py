import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd
import sys
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("austin_weather.csv")
    df = df.drop('Date', axis=1)

    indexes = list()
    for iter,row in df.iterrows():
        for c in df.columns:
            if row[c] == '-' :
                if iter not in indexes: indexes.append(iter)
    df.drop(df.index[indexes], inplace=True)

    predictions_df = df[['Events']]
    predictions_df['Events'] = np.where(df['Events'].str.contains('Rain'), 1,  0)
    samples_df = df.drop(['Events','PrecipitationSumInches'], axis=1)
    
    print(samples_df)
    print(predictions_df['Events'].value_counts())
    
    data_vars=samples_df.columns.values.tolist()
    data_final_vars = list()

    logreg = LogisticRegression(max_iter=10000)
    rfe = RFE(logreg, n_features_to_select=9)
    rfe = rfe.fit(samples_df, predictions_df.values.ravel())
    print(rfe.support_)
    print(rfe.ranking_)

    for idx,val in enumerate(rfe.support_):
        if val: data_final_vars.append(data_vars[idx])

    print(samples_df.columns.values)
    print(data_final_vars)

    final_samples_df = samples_df[data_final_vars]
    print(final_samples_df)

    X_train, X_test, y_train, y_test = train_test_split(final_samples_df, predictions_df, test_size=0.2)
    print(f'Полная выборка: {len(samples_df)}\nТренировочная: {len(X_train)}')

    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)

    np.set_printoptions(threshold=sys.maxsize)
    y_pred = model.predict(X_test)

    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))
    #print("MSE:", mean_squared_error(y_test, y_pred))
    #print("r2_score = %s" % r2_score(y_test, y_pred))

    logit_roc_auc = roc_auc_score(y_test, model.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    main()