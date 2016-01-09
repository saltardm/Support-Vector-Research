
import cPickle
import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import operator
import pandas.io.data
from sklearn.qda import QDA
import re
from dateutil import parser
from backtest import Strategy, Portfolio
import pystocks

def getStock(symbol, start, end):
    """
    Downloads Stock from Yahoo Finance.
    Computes daily Returns based on Adj Close.
    Returns pandas dataframe.
    """
    df =  pd.io.data.get_data_yahoo(symbol, start, end)

    df.columns.values[-1] = 'AdjClose'
    df.columns = df.columns + '_' + symbol
    df['Return_%s' %symbol] = df['AdjClose_%s' %symbol].pct_change()
    
    return df

def getStockDataFromWeb(fout, start_string, end_string):
    """
    Collects predictors data from Yahoo Finance and Quandl.
    Returns a list of dataframes.
    """
    start = parser.parse(start_string)
    end = parser.parse(end_string)
    
    nasdaq = getStock('^IXIC', start, end)
    frankfurt = getStock('^GDAXI', start, end)
    london = getStock('^FTSE', start, end)
    paris = getStock('^FCHI', start, end)
    hkong = getStock('^HSI', start, end)
    nikkei = getStock('^N225', start, end)
    australia = getStock('^AXJO', start, end)
    
    djia = getStockFromQuandl("YAHOO/INDEX_DJI", 'Djia', start_string, end_string) 
    
    out =  pd.io.data.get_data_yahoo(fout, start, end)
    out.columns.values[-1] = 'AdjClose'
    out.columns = out.columns + '_Out'
    out['Return_Out'] = out['AdjClose_Out'].pct_change()
    
    return [out, nasdaq, djia, frankfurt, london, paris, hkong, nikkei, australia]
#returns delta-Multiple day returns and delta-Returns moving average
def addFeatures(dataframe, adjclose, returns, n):
 
    
    return_n = adjclose[9:] #+ "Time" + str(n)
    dataframe[return_n] = dataframe[adjclose].pct_change(n)
    
    roll_n = returns[7:]# + "RolMean" + str(n)
    dataframe[roll_n] = pd.rolling_mean(dataframe[returns], n)

#adds features to each dataset and returns augmented list
def applyRollMeanDelayedReturns(datasets, delta):
  
    for dataset in datasets:
        columns = dataset.columns    
        adjclose = columns[-2]
        returns = columns[-1]
        for n in delta:
            addFeatures(dataset, adjclose, returns, n)
    
    return datasets
#2 part join, outer join of predictors UNION and then Left join of predictors
def mergeDataframes(datasets, index, cut):
    """
    merges datasets in the list 
    """
    
    subset = [dataset.iloc[:, index:] for dataset in datasets[1:]]
    
    first = subset[0].join(subset[1:], how = 'outer')
    finance = datasets[0].iloc[:, index:].join(first, how = 'left') 
    finance = finance[finance.index > cut]
    return finance

def applyTimeLag(dataset, lags, delta):
    """
    apply time lag to return columns selected according  to delta.
    Days to lag are contained in the lads list passed as argument.
    Returns a NaN free dataset obtained cutting the lagged dataset
    at head and tail
    """
    
    dataset.Return_Out = dataset.Return_Out.shift(-1)
    maxLag = max(lags)

    columns = dataset.columns[::(2*max(delta)-1)]
    for column in columns:
        for lag in lags:
            newcolumn = column + str(lag)
            dataset[newcolumn] = dataset[column].shift(lag)

    return dataset.iloc[maxLag:-1,:]
#SV Classifier
def prepareDataForClassification(dataset, start_test):
    """
    generates categorical output column, attach to dataframe 
    label the categories and split into train and test
    """
    le = preprocessing.LabelEncoder()
    
    dataset['UpDown'] = dataset['Return_Out']
    dataset.UpDown[dataset.UpDown >= 0] = 'Up'
    dataset.UpDown[dataset.UpDown < 0] = 'Down'
    dataset.UpDown = le.fit(dataset.UpDown).transform(dataset.UpDown)
    
    features = dataset.columns[1:-1]
    X = dataset[features]    
    y = dataset.UpDown    
    
    X_train = X[X.index < start_test]
    y_train = y[y.index < start_test]              
    
    X_test = X[X.index >= start_test]    
    y_test = y[y.index >= start_test]
    
    return X_train, y_train, X_test, y_test

def performClassification(X_train, y_train, X_test, y_test, method, parameters, fout, savemodel):
    """
    performs classification on daily returns using several algorithms (method).
    method --> string algorithm
    parameters --> list of parameters passed to the classifier (if any)
    fout --> string with name of stock to be predicted
    savemodel --> boolean. If TRUE saves the model to pickle file
    """
   
    if method == 'RF':   
        return performRFClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)
        
    elif method == 'KNN':
        return performKNNClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)
    
    elif method == 'SVM':   
        return performSVMClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)
    
    elif method == 'ADA':
        return performAdaBoostClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)
    
    elif method == 'GTB': 
        return performGTBClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)

    elif method == 'QDA': 
        return performQDAClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)
#Random forest classifier
def performRFClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
    """
    Random Forest Binary Classification
    """
    clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    if savemodel == True:
        fname_out = '{}-{}.pickle'.format(fout, datetime.now())
        with open(fname_out, 'wb') as f:
            cPickle.dump(clf, f, -1)    
    
    accuracy = clf.score(X_test, y_test)
    
    return accuracy

#K-Nearest Neighbours Cluster
def performKNNClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
    """
    KNN binary Classification
    """
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)

    if savemodel == True:
        fname_out = '{}-{}.pickle'.format(fout, datetime.now())
        with open(fname_out, 'wb') as f:
            cPickle.dump(clf, f, -1)    
    
    accuracy = clf.score(X_test, y_test)
    
    return accuracy

#SVM
def performSVMClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
    """
    SVM binary Classification
    """
    c = parameters[0]
    g =  parameters[1]
    clf = SVC()
    clf.fit(X_train, y_train)

    if savemodel == True:
        fname_out = '{}-{}.pickle'.format(fout, datetime.now())
        with open(fname_out, 'wb') as f:
            cPickle.dump(clf, f, -1)    
    
    accuracy = clf.score(X_test, y_test)
    
    return accuracy
#Adaptive Boost
def performAdaBoostClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
    """
    Ada Boosting binary Classification
    """
    n = parameters[0]
    l =  parameters[1]
    clf = AdaBoostClassifier()
    clf.fit(X_train, y_train)

    if savemodel == True:
        fname_out = '{}-{}.pickle'.format(fout, datetime.now())
        with open(fname_out, 'wb') as f:
            cPickle.dump(clf, f, -1)    
    
    accuracy = clf.score(X_test, y_test)
    
    return accuracy
#Gradient Tree
def performGTBClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
    """
    Gradient Tree Boosting binary Classification
    """
    clf = GradientBoostingClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    if savemodel == True:
        fname_out = '{}-{}.pickle'.format(fout, datetime.now())
        with open(fname_out, 'wb') as f:
            cPickle.dump(clf, f, -1)    
    
    accuracy = clf.score(X_test, y_test)
    
    return accuracy

#QD Analysis
def performQDAClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
    """
    Quadratic Discriminant Analysis binary Classification
    """
    def replaceTiny(x):
        if (abs(x) < 0.0001):
            x = 0.0001
    
    X_train = X_train.apply(replaceTiny)
    X_test = X_test.apply(replaceTiny)
    
    clf = QDA()
    clf.fit(X_train, y_train)

    if savemodel == True:
        fname_out = '{}-{}.pickle'.format(fout, datetime.now())
        with open(fname_out, 'wb') as f:
            cPickle.dump(clf, f, -1)    
    
    accuracy = clf.score(X_test, y_test)
    
    return accuracy

#Must now run cross validation to determine the best machine learning algorithm to fit to our data
'''
First of all we have to decide a metrics to evaluate the results of our prediction. Generally a well accepted measurement is the Area Under the Curve (AUC), which consists in the percentage of misclassified events counted at several probability thresholds. The output of a prediction algorithm can always be interpreted as a probability for a certain case to belong to a specific class or not. For a binary classification the default behavior of the algorithm is to classify as “0” a case whose probability of belonging to “1” is less than 50%, and viceversa. This threshold can be varied as needed, depending on the field of investigation. There may be situations in which this kind of tricks is absolutely fundamental. This is the case for example in which the relative proportion of the two classes is extremely skewed towards one of them. In this case the 50% threshold does not really make sense. If I’m building a model to detect whether a person is affected by a particular disease or not and the disease rate in the whole population is let’s say 3% then I do want to be very careful, as this 3% is very likely to fall in my misclassification error.

The AUC takes care of this kind of issues measuring the robustness of a classifier at several probability thresholds. In my case, being stock markets notoriously randomic I decided to stick to the more classic accuracy of a classifier fixing my threshold at 50%.

Having said that let’s come back to Cross Validation. In order to show the technique I’ll walk through a practical example.

We want to assess the performance of a Random Forest Classifier in the following conditions:

100 trees (n_estimators in Scikit-Learn)
n = 2 / delta = 2 . Thus we are lagging the returns at maximun for 2 days and we are computing at maximum 2-day-returns and 2-day-moving average returns.
What we do next is what follows:

Split data in train and test set given a Date (in my case after 1 April 2014 included).
Split train set (before 1 April 2014 not included) in 10 consecutive time folds.
Then, in order not lo lose the time information, perform the following steps:
Train on fold 1 –>  Test on fold 2
Train on fold 1+2 –>  Test on fold 3
Train on fold 1+2+3 –>  Test on fold 4
Train on fold 1+2+3+4 –>  Test on fold 5
Train on fold 1+2+3+4+5 –>  Test on fold 6
Train on fold 1+2+3+4+5+6 –>  Test on fold 7
Train on fold 1+2+3+4+5+6+7 –>  Test on fold 8
Train on fold 1+2+3+4+5+6+7+8 –>  Test on fold 9
Train on fold 1+2+3+4+5+6+7+8+9 –>  Test on fold 10
Compute the average of the accuracies of the 9 test folds (number of folds  – 1)
Repeat steps 1-12 in the following conditions:

n = 2 / delta = 3
n = 2 / delta = 4
n = 2 / delta = 5
n = 3 / delta = 2
n = 3 / delta = 3
n = 3 / delta = 4
n = 3 / delta = 5
n = 4 / delta = 2
n = 4 / delta = 3
n = 4 / delta = 4
n = 4 / delta = 5
n = 5 / delta = 2
n = 5 / delta = 3
n = 5 / delta = 4
n = 5 / delta = 5
and get average of the accuracies of the 9 test folds  in each one of the previous cases. Obviously there is an infinite number of possibilities to generate and assess. What I did is to stop at a maximum of 10 days. Thus I basically performed a double for loop up to n = 10 / delta = 10.

Each time the script gets into one iteration of the for loop it generates a brand new dataframe with a different set of features. Then, on top of the newborn dataframe, 10-fold Cross Validation is performed in order to assess the performance of the selected algorithm  with that particular set of predictors.

I repeated this set of operations for all the algorithms introducued in the previous post (Random Forest, KNN, SVM Adaptive Boosting, Gradient Tree Boosting, QDA) and after all the computations the best result is the following:

Random Forest | n = 9 / delta = 9 | Average Accuracy = 0.530748
The output for this specific conditions is provided below together with the python function in charge of looping over n and delta and performing CV on each iteration
'''
def performFeatureSelection(maxdeltas, maxlags, fout, cut, start_test, path_datasets, savemodel, method, folds, parameters):
    """
    Performs Feature selection for a specific algorithm
    """
    
    for maxlag in range(3, maxlags + 2):
        lags = range(2, maxlag) 
        print ''
        print '============================================================='
        print 'Maximum time lag applied', max(lags)
        print ''
        for maxdelta in range(3, maxdeltas + 2):
            datasets = loadDatasets(path_datasets, fout)
            delta = range(2, maxdelta) 
            print 'Delta days accounted: ', max(delta)
            datasets = applyRollMeanDelayedReturns(datasets, delta)
            finance = mergeDataframes(datasets, 6, cut)
            print 'Size of data frame: ', finance.shape
            print 'Number of NaN after merging: ', count_missing(finance)
            finance = finance.interpolate(method='linear')
            print 'Number of NaN after time interpolation: ', count_missing(finance)
            finance = finance.fillna(finance.mean())
            print 'Number of NaN after mean interpolation: ', count_missing(finance)    
            finance = applyTimeLag(finance, lags, delta)
            print 'Number of NaN after temporal shifting: ', count_missing(finance)
            print 'Size of data frame after feature creation: ', finance.shape
            X_train, y_train, X_test, y_test  = prepareDataForClassification(finance, start_test)
            
            print performCV(X_train, y_train, folds, method, parameters, fout, savemodel)
            print ''
'''
It is very important to notice that nothing has been done at an algorithmic parameter level. What I mean is that with the previous approach we have been able to achieve two very important goals:

Assess the best classification algorithm, comparing all of them on the same set of features.
Assess for each algorithm the best set of features.
Pick the couple (Model/Features) maximizing the CV Accuracy
'''
# last trading day accounted
end_period = datetime(2014,8,28)

# symbol of the stock required for future plotting
symbol = 'S&P-500'

# name of the file of the output of prediction (S&P 500 in this case)
name = 'C:\Users\Drew\Desktop\SCikit' + '/sp500.csv'

# calls the best model previously saved in pickle file and runs it on the test set retutning an array of 0,1 (Down, Up) according to predicted returns
prediction = 100#pystocks.getPredictionFromBestModel(9, 9, 'sp500', cut, start_test, path_datasets, 'sp500.pickle')[0]

# dataframe of S&P 500 historical prices (saved locally from Yahho Finance)
bars = pd.read_csv(name, index_col=0, parse_dates=True)    
start_test = '2005-01-01'
end_period = '2007-01-01'
# subset of the data corresponding to test set
bars = bars[start_test:end_period]

# initialize empty dataframe indexed as the bars. There's going to be perfect match between dates in bars and signals 
signals = pd.DataFrame(index=bars.index)

# initialize signals.signal column to zero
signals['signal'] = 0.0

# copying into signals.signal column results of prediction
signals['signal'] = prediction

# replace the zeros with -1 (new encoding for Down day)
signals.signal[signals.signal == 0] = -1

# compute the difference between consecutive entries in signals.signal. As
# signals.signal was an array of 1 and -1 return signals.positions will 
# be an array of 0s and 2s.
signals['positions'] = signals['signal'].diff()     

# calling portfolio evaluation on signals (predicted returns) and bars 
# (actual returns)
portfolio = pystocks.PortfolioManager(symbol, bars, signals)

# backtesting the portfolio and generating returns on top of that 
returns = portfolio.backtest_portfolio()

# last trading day accounted
end_period = datetime.datetime(2014,8,28)

# symbol of the stock required for future plotting
symbol = 'S&P-500'

# name of the file of the output of prediction (S&P 500 in this case)
name = path_datasets + '/sp500.csv'

# calls the best model previously saved in pickle file and runs it on the test set retutning an array of 0,1 (Down, Up) according to predicted returns
prediction = backtest.getPredictionFromBestModel(9, 9, 'sp500.csv', cut, start_test, path_datasets, 'sp500_57.pickle')[0]

# dataframe of S&P 500 historical prices (saved locally from Yahho Finance)
bars = pd.read_csv(name, index_col=0, parse_dates=True)    

# subset of the data corresponding to test set
bars = bars[start_test:end_period]

# initialize empty dataframe indexed as the bars. There's going to be perfect match between dates in bars and signals 
signals = pd.DataFrame(index=bars.index)

# initialize signals.signal column to zero
signals['signal'] = 0.0

# copying into signals.signal column results of prediction
signals['signal'] = prediction

# replace the zeros with -1 (new encoding for Down day)
signals.signal[signals.signal == 0] = -1

# compute the difference between consecutive entries in signals.signal. As
# signals.signal was an array of 1 and -1 return signals.positions will 
# be an array of 0s and 2s.
signals['positions'] = signals['signal'].diff()     

# calling portfolio evaluation on signals (predicted returns) and bars 
# (actual returns)
portfolio = portfolio.PortfolioManager(symbol, bars, signals)

# backtesting the portfolio and generating returns on top of that 
returns = portfolio.backtest_portfolio()

def getPredictionFromBestModel(bestdelta, bestlags, fout, cut, start_test, path_datasets, best_model):
    """
    returns array of prediction and score from best model.
    """
    lags = range(2, bestlags + 1) 
    datasets = loadDatasets(path_datasets, fout)
    delta = range(2, bestdelta + 1) 
    datasets = applyRollMeanDelayedReturns(datasets, delta)
    finance = mergeDataframes(datasets, 6, cut)
    finance = finance.interpolate(method='linear')
    finance = finance.fillna(finance.mean())    
    finance = applyTimeLag(finance, lags, delta)
    X_train, y_train, X_test, y_test  = prepareDataForClassification(finance, start_test)    
    with open(best_model, 'rb') as fin:
        model = cPickle.load(fin)        
        
    return model.predict(X_test), model.score(X_test, y_test)

portfolio = portfolio.PortfolioManager(symbol, bars, signals)

returns = portfolio.backtest_portfolio()

