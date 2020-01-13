import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt


# Max drawdown function
def max_drawdown_numbers(X):
    mdd = 0
    peak = X[0]
    for x in X:
        if x > peak:
            peak = x
        dd = (peak - x) / peak
        if dd > mdd:
            mdd = dd

    i = np.argmax(np.maximum.accumulate(X) - X)  # end of the period
    j = np.argmax(X[:i])  # start of period

    plt.plot(X)
    plt.plot([i, j], [X[i], X[j]], 'o', color='Red', markersize=10)

    return mdd

def max_drawdown_timesseries(X):
    '''

    :param X: dataframe containing prices
    :return:  max drawdown and the graph

    '''
    X_orig = X.copy()
    X = X.values
    mdd = 0
    peak = X[0]
    for x in X:
        if x > peak:
            peak = x
        dd = (peak - x) / peak
        if dd > mdd:
            mdd = dd

    i = np.argmax(np.maximum.accumulate(X) - X)  # end of the period
    j = np.argmax(X[:i])  # start of period

    plt.plot(X_orig)
    plt.plot([X_orig.index[i], X_orig.index[j]], [X_orig.iloc[i], X_orig.iloc[j]], 'o', color='Red', markersize=10)

    return mdd

def sharpe_and_dd_calcs(df,freq):
    '''
    :param : send df containing pct_change
    :return: sharpe and drawdown values
    '''
    #sharpe ratio
    if freq == 'M': #monthly
        fq = 12
    elif freq == 'Q': #quarterly
        fq = 6
    else:
        fq = 252 #daily

    vol = np.std(df) * np.sqrt(fq) * 100
    cumret = (np.cumprod(1 + df) - 1).iloc[-1] * 100 #in percentage
    ann_returns = (((1+cumret/100)**(1/(df.shape[0]/fq)))-1)*100 #in percentage
    sharpe = ann_returns / vol

    #drawdown
    cum_returns = (1 + df).cumprod()
    maxdrawdown = np.max(1 - cum_returns.div(cum_returns.cummax())) * 100

    #create a Dataframe
    srdf = pd.DataFrame(sharpe, columns=['Sharpe Ratio'])
    mdddf = pd.DataFrame(maxdrawdown, columns=['Max DD'])
    ann_returndf = pd.DataFrame(ann_returns)
    ann_returndf.columns = ['Ann Returns']
    voldf = pd.DataFrame(vol, columns=['Ann Vols'])
    table = pd.concat([srdf,mdddf,ann_returndf,voldf],axis=1)
    return sharpe,maxdrawdown,ann_returndf,table

if __name__ == '__main__':
    n = 1000
    xs = np.random.randn(n).cumsum()

    df = pd.read_csv('data_new_Q.csv', index_col='date', parse_dates=True)
    stock = 'MSFT'
    xsts = df[[stock]].dropna()

    #xs = xsts.values
    #drawSeries = max_drawdown_numbers(xs)
    #MaxDD = abs(drawSeries.min()*100)
    #print('The max drawdown is {:.3f}%'.format(MaxDD))

    drawSeries = max_drawdown_timesseries(xsts)
    MaxDD = abs(drawSeries.min() * 100)
    print('The max drawdown is {:.3f}%'.format(MaxDD))


    plt.show()