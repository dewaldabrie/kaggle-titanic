"""
Compare Algorithms
Much of this taken from 
https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/
"""

import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor
from clean_titanic import clean

def main():
    # load dataset
    df1 = pandas.read_csv("data/train.csv")
    X, Y = clean(df1)
    X = X.values
    Y = Y.values.ravel()
    #url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    #names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    #dataframe = pandas.read_csv(url, names=names)
    #array = dataframe.values
    #X = array[:,0:8]
    #Y = array[:,8]
    # prepare configuration for cross validation test harness
    seed = 7
    # prepare models
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    models.append(('DTreeReg', DecisionTreeRegressor()))
    models.append(('DTreeClass', DecisionTreeClassifier()))
    models.append(('ADABoostDTreeReg', AdaBoostRegressor(DecisionTreeRegressor())))
    # evaluate each model in turn
    cross_val_scores = []
    names = []
    scoring = 'f1'
    for name, model in models:
        kfold = model_selection.KFold(n_splits=5, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        cross_val_scores.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    box_plot(cross_val_scores, names)

def box_plot(score_results, names):
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(score_results)
    ax.set_xticklabels(names)
    plt.show()

def precision_recall_curve(y_test, y_score):
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_test, y_score)

    plt.step(recall, precision, color='b', alpha=0.2,
                     where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision))


if __name__ == "__main__":
    main()
