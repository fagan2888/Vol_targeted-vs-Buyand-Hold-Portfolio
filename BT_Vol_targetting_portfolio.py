'''

backtests a portfolio of buy and hold vs voltargetted. i.e. if the portfolio volatility drops below X% (defined by you)
it will run and optimisation to reassign weights.

'''
import timeit
start = timeit.default_timer()
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from backtest import Strategy,Portfolio
#from Risk_Parity_v1 import _get_risk_parity_weights
#from OPT_cvxpy_optimisation import markowitz_portfolio

from OPT_vol_targetting import run_unconstrained_SR_optimiser
from OPT_vol_targetting import run_vol_target_optimiser
from dateutil.relativedelta import relativedelta

from max_drawdown_updated import sharpe_and_dd_calcs

class VolTargetting(Strategy):

    def __init__(self,df_m,date_list,names):
        self.df = df_m
        self.datelist = date_list
        self.names = names

    def generate_signals(self):
        sig = pd.DataFrame(index=self.datelist)

        for x,name in enumerate(self.names):
            sig[name] = 1
        return sig




class MarketOnClosePortfolio(Portfolio):

    def __init__(self,sig,df_m,date_list,bounds,vol_target):
        self.sig = sig
        self.df = df_m
        self.datelist = date_list
        self.bounds = bounds
        self.voltarget = vol_target
        self.positions,self.vols = self.generate_positions()

    def generate_all_positions(self):
        assetcount = len(list(self.df.columns.to_list()))
        weights_all = []
        vols_all = []
        sr_all = []
        for i,dts in enumerate(self.datelist):
            if i==0:
                #dts = dts + relativedelta(months=-1)
                lb_startdate = dts + relativedelta(months=-6)
                covariances = 252 * self.df[lb_startdate:dts].pct_change(1).dropna().cov().values
                rtns = 252 * self.df[lb_startdate:dts].pct_change(1).dropna().mean()
                optweights = run_unconstrained_SR_optimiser(assetcount,rtns,covariances,self.bounds)[3]
                optvols = np.sqrt(np.dot(optweights, np.dot(covariances, optweights.T)))

            else:
                # dts = dts + relativedelta(months=-1)
                lb_startdate = dts + relativedelta(months=-6)
                covariances = 252 * self.df[lb_startdate:dts].pct_change(1).dropna().cov().values
                optvols = np.sqrt(np.dot(optweights,np.dot(covariances,optweights.T)))

                if optvols > 0.1:
                    rtns = 12 * self.df[lb_startdate:dts].pct_change(1).dropna().mean()
                    optweights = run_vol_target_optimiser(assetcount,rtns,covariances,self.voltarget,self.bounds)[3]
                    optvols = np.sqrt(np.dot(optweights, np.dot(covariances, optweights.T)))

            #cfweigths = optweights

            weights_all.append(optweights)
            vols_all.append(optvols)


        df_weights = pd.DataFrame(weights_all, index=date_list)
        df_vols = pd.DataFrame(vols_all, index=date_list)
        return df_weights,df_vols

    def generate_positions(self):
        positions,vols = self.generate_all_positions()
        return positions,vols



    def backtest_portfolio(self,df_monthly_changes):
        port_rtns_vol_targetted = np.sum(self.positions.mul(df_monthly_changes.values), axis=1)

        #calculate buy and hold returns
        pos_bh = self.positions.copy()
        pos_bh.iloc[1:, :] = pos_bh.iloc[1:, :] * np.nan
        pos_bh.ffill(inplace=True)
        port_rtns_buyandhold = np.sum(pos_bh.mul(df_monthly_changes.values), axis=1)
        return port_rtns_vol_targetted,port_rtns_buyandhold



if __name__ == '__main__':

    from dateutil.relativedelta import relativedelta

    df = pd.read_csv('all_asset_class.csv', index_col='Date', parse_dates=True)
    df.drop(['dollar', 'yc', 'senti', '7yTR', '10yTR', '30yTR'], axis=1, inplace=True)
    df.ffill(inplace=True)
    df.columns = ['Crude', 'Gold', 'DM Equity', 'EM Corp', 'EM Equity', 'TSY', '$Corp', '$HY', '$BBB']


    stocks = ['Crude', 'Gold', 'DM Equity', 'EM Corp', 'EM Equity', 'TSY', '$Corp', '$HY', '$BBB']

    df = df[stocks]
    all_list = []
    for i, x in enumerate(stocks):
        al = str(stocks[i])
        all_list.append(al)

    df_m = df.resample('BM').last()


    # Get the start date
    lookbackperiod = 6  # months
    start_date = pd.date_range(df_m.index[0], periods=lookbackperiod + 2, freq='BM')[lookbackperiod + 1]
    date_list = df_m[start_date:].index.tolist()

    # get separate startdate to calculate returns
    start_date = pd.date_range(df_m.index[0], periods=lookbackperiod + 1, freq='BM')[lookbackperiod]
    date_list_returns = df_m[start_date:].index.tolist()


    df_monthly_changes = df_m.reindex(date_list_returns).pct_change().dropna()
    bounds = ((0.05, .15), (0.05, .15), (0.1, .4), (0.05, .2), (0.05, .2), (0.1, .2),(0.05, .2),(0.05, .2),(0.05, .2))

    #define volatility target
    vol_target = 0.07
    #initiate classes
    vt_prt = VolTargetting(df,date_list,stocks)
    sig = vt_prt.generate_signals()

    port = MarketOnClosePortfolio(sig,df,date_list,bounds,vol_target)
    vol_targetted_rtns, buyhold_rtns  = port.backtest_portfolio(df_monthly_changes)

    cum_vol_targetted_rtns = np.cumproduct(1+vol_targetted_rtns)-1
    cum_buyhold_rtns = np.cumproduct(1 + buyhold_rtns) - 1

    #get sharpe and maxdrawdowns
    comb = pd.concat([vol_targetted_rtns, buyhold_rtns], axis=1)
    comb.columns = ['Vol targetted', 'Buy and hold']
    table = sharpe_and_dd_calcs(comb,'M')[3]
    print(table.round(2))
    #plot graphs
    cum_vol_targetted_rtns.plot(label='vol targetted',legend=True)
    cum_buyhold_rtns.plot(label='Buy and hold',legend=True)
    plt.xlabel('Years')
    plt.ylabel('Returns')
    plt.grid('True')
    plt.show()