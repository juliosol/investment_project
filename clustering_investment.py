# Indicators to compute

#
#
#

import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_ta as ta
import pandas_datareader.data as web
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

import warnings
warnings.filterwarnings("ignore")

def get_ticker_data(end_date='2025-05-30', load=False):

    if not load:
        #import pdb
        #pdb.set_trace()
        
        # Set User-Agent header to avoid 403 Forbidden error
        import requests
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url, headers=headers)
        sp500 = pd.read_html(response.content)[0]

        sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')

        symbols_list = sp500['Symbol'].unique().tolist()

        no_symbols = ['SOLV', 'GEV', 'SW', 'VLTO']
        
        start_date = pd.to_datetime(end_date) - pd.DateOffset(365*8)#10)

        df = yf.download(tickers=symbols_list, start=start_date, end=end_date, auto_adjust=False).stack()
        #, group_by='ticker', auto_adjust=True)

        df.index.names = ['date', 'ticker']
        df.columns = df.columns.str.lower()

        df.to_pickle('sp500_data.pkl')

    else:
        df = pd.read_pickle('sp500_data.pkl')

    return df

def compute_indicators(df):

    # Indicators to compute
    # Garman-Klass volatility
    # RSI
    # Bollinger Bands
    # ATR
    # MACD
    # Dollar Volume

    df['garman_klass'] = ((np.log(df['high']) - np.log(df['low']))**2)/2 - (2*np.log(2)-1) * ((np.log(df['adj close']) - np.log(df['open']))**2)

    df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: ta.rsi(close=x, length=20))

    df['bb_low'] = df.groupby(level=1)['adj close'].transform(lambda x: ta.bbands(close=np.log1p(x), length=20).iloc[:,0])
    df['bb_mid'] = df.groupby(level=1)['adj close'].transform(lambda x: ta.bbands(close=np.log1p(x), length=20).iloc[:,1])
    df['bb_high'] = df.groupby(level=1)['adj close'].transform(lambda x: ta.bbands(close=np.log1p(x), length=20).iloc[:,2])

    def compute_atr(stock_data):
        high = stock_data['high']
        low = stock_data['low']
        close = stock_data['close']
        atr = ta.atr(high=high, low=low, close=close, length=14)
        return atr.sub(atr.mean()).div(atr.std())
    
    df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)

    def compute_macd(close):
        macd = ta.macd(close=close, length=20).iloc[:,0]
        return macd.sub(macd.mean()).div(macd.std())
        
    df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)

    df['dollar_volume'] = (df['adj close'] * df['volume'])/1e6

    return df

def aggregate_filter(df, save=False, load=False):
    """
    Method to aggregate the data and filter out stocks with low dollar volume (keep top 150 stocks by dollar volume).
    """

    def rolling_average_dv(df, window=5):
        """
        Compute rolling averages for the features in the DataFrame.
        """
        df['dollar_volume'] = df.loc[:, 'dollar_volume'].unstack('ticker').rolling(window*12, min_periods=12).mean().stack()

        df['dollar_vol_rank'] = df.groupby(level='date')['dollar_volume'].rank(ascending=False)

        df = df[df['dollar_vol_rank'] < 150].drop(['dollar_vol_rank', 'dollar_volume'], axis=1)

        return df

    if not load:
        features_cols = [c for c in df.columns.unique(0) if c not in ['dollar_volume', 'volume', 'open', 'high', 'low', 'close']]

        df_vols = df.unstack('ticker')['volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume')

        df_features = df.unstack()[features_cols].resample('M').last().stack('ticker')

        df = pd.concat([df_features, df_vols], axis=1).dropna()

        df.to_pickle('sp500_feature_data.pkl')
    else:
        df = pd.read_pickle('sp500_feature_data.pkl')

    df = rolling_average_dv(df)

    return df

def monthly_returns_diff_time_horizons(df):
    def calculate_returns(df):
        lags = [1, 2, 3, 6, 9, 12]
        outlier_cutoff = 0.0005

        for lag in lags:
            df[f'returns_{lag}m'] = (df['adj close']
                                    .pct_change(lag)
                                    .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                            upper=x.quantile(1-outlier_cutoff)))
                                    .add(1)
                                    .pow(1/lag)
                                    .sub(1))
        return df
    
    df = df.groupby(level=1, group_keys=False).apply(calculate_returns).dropna()

    return df

def get_fama_french_factors(df, load=False):
    """
    Load Fama-French factors from a CSV file.
    """
    if not load:
        ff_factors = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='2010')[0].drop('RF', axis=1)
        ff_factors.index = ff_factors.index.to_timestamp()
        ff_factors = ff_factors.resample('M').last().div(100)
        ff_factors.index.name = 'date'
        ff_factors = ff_factors.join(df['returns_1m']).sort_index()
        
        # Filter stocks with less than 10 months of data
        observations = ff_factors.groupby(level=1).size()
        valid_stocks = observations[observations >= 10]
        ff_factors = ff_factors[ff_factors.index.get_level_values('ticker').isin(valid_stocks.index)]

        # Calculate rolling factor betas
        betas = (ff_factors.groupby(level=1,
                                    group_keys=False)
                            .apply(lambda x: RollingOLS(endog=x['returns_1m'],
                                                        exog=sm.add_constant(x.drop('returns_1m', axis=1)),
                                                        window=min(24, x.shape[0]),
                                                        min_nobs=len(x.columns)+1)  
                            .fit(params_only=True)
                            .params
                            .drop('const', axis=1)))
        
        factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

        df = (df.join(betas.groupby('ticker').shift()))

        df.loc[:, factors] = df.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))

        df = df.drop('adj close', axis=1)

        df = df.dropna()

        df.to_pickle('augmented_sp500_feature_data.pkl')

        return df
    else:
        df = pd.read_pickle('augmented_sp500_feature_data.pkl')

    return df

def kmeans_clustering(df):
    """
    Perform KMeans clustering on the DataFrame.
    """
    #df = df.dropna()
    #features = df.select_dtypes(include=[np.number]).columns.tolist()


    target_rsi_values = [30, 45, 55, 85]

    initial_centroids = np.zeros((len(target_rsi_values), 18))

    initial_centroids[:, 1] = target_rsi_values
    
    kmeans = KMeans(n_clusters=4, random_state=0, init=initial_centroids)
    df['cluster'] = kmeans.fit(df).labels_
    
    return df

def plot_clusters(data):

    cluster_0 = data[data['cluster']==0]
    cluster_1 = data[data['cluster']==1]
    cluster_2 = data[data['cluster']==2]
    cluster_3 = data[data['cluster']==3]

    plt.scatter(cluster_0.iloc[:,5] , cluster_0.iloc[:,1] , color = 'red', label='cluster 0')
    plt.scatter(cluster_1.iloc[:,5] , cluster_1.iloc[:,1] , color = 'green', label='cluster 1')
    plt.scatter(cluster_2.iloc[:,5] , cluster_2.iloc[:,1] , color = 'blue', label='cluster 2')
    plt.scatter(cluster_3.iloc[:,5] , cluster_3.iloc[:,1] , color = 'black', label='cluster 3')
    
    plt.legend()
    plt.show()
    return

def select_stocks(df, last_nums=10):
    filtered_df = df[df['cluster']==3].copy()
    filtered_df = filtered_df.reset_index(level=1)
    filtered_df.index = filtered_df.index + pd.DateOffset(1)
    filtered_df = filtered_df.reset_index().set_index(['date', 'ticker'])
    dates = filtered_df.index.get_level_values('date').unique().tolist()

    fixed_dates = {}

    for d in dates:#[-last_nums:]:
        top_rsi_stocks = filtered_df.xs(d, level=0).sort_values('rsi').iloc[-10:]
        fixed_dates[d.strftime('%Y-%m-%d')] = top_rsi_stocks.index.tolist()

    return fixed_dates

def optimize_weights(prices, lower_bound=0):
    
    returns = expected_returns.mean_historical_return(prices=prices,
                                                      frequency=252)
    
    cov = risk_models.sample_cov(prices=prices,
                                 frequency=252)
    
    ef = EfficientFrontier(expected_returns=returns,
                           cov_matrix=cov,
                           weight_bounds=(lower_bound, .1),
                           solver='SCS')
    
    weights = ef.max_sharpe()
    
    return ef.clean_weights()

def portfolio_optimization(new_df, fixed_dates):
    returns_dataframe = np.log(new_df['Adj Close']).diff()

    portfolio_df = pd.DataFrame()

    date_weights = {}

    for start_date in fixed_dates.keys():
        
        try:

            end_date = (pd.to_datetime(start_date)+pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')

            cols = fixed_dates[start_date]

            optimization_start_date = (pd.to_datetime(start_date)-pd.DateOffset(months=12)).strftime('%Y-%m-%d')

            optimization_end_date = (pd.to_datetime(start_date)-pd.DateOffset(days=1)).strftime('%Y-%m-%d')
            
            optimization_df = new_df[optimization_start_date:optimization_end_date]['Adj Close'][cols]
            
            success = False

            #import pdb
            #pdb.set_trace()

            try:
                lower_bound = round(1/(len(optimization_df.columns)*2),3)
                weights = optimize_weights(prices=optimization_df,
                                    lower_bound=lower_bound)

                weights = pd.DataFrame(weights, index=pd.Series(0))
                
                success = True
            except:
                print(f'Max Sharpe Optimization failed for {start_date}, Continuing with Equal-Weights')
            
            if success==False:
                weights = pd.DataFrame([1/len(optimization_df.columns) for i in range(len(optimization_df.columns))],
                                        index=optimization_df.columns.tolist(),
                                        columns=pd.Series(0)).T
            
            temp_df = returns_dataframe[start_date:end_date]

            date_weights[start_date] = weights


            temp_df = temp_df.stack().to_frame('return').reset_index(level=0)\
                    .merge(weights.stack().to_frame('weight').reset_index(level=0, drop=True),
                            left_index=True,
                            right_index=True)\
                    .reset_index().set_index(['Date', 'Ticker']).unstack().stack()

            temp_df['weighted_return'] = temp_df['return']*temp_df['weight']

            temp_df = temp_df.groupby(level=0)['weighted_return'].sum().to_frame('Strategy Return')

            portfolio_df = pd.concat([portfolio_df, temp_df], axis=0)
        
        except Exception as e:
            print(e)

    portfolio_df = portfolio_df.drop_duplicates()

    return portfolio_df, date_weights

def compare_sp_500(port_df):
    spy = yf.download(tickers='SPY',
                    start='2015-01-01',
                    end=dt.date.today(),
                    auto_adjust=False)

    
    spy_ret = np.log(spy[['Adj Close']]).diff().dropna().rename({'Adj Close':'SPY Buy&Hold'}, axis=1)
    spy_ret.columns = spy_ret.columns.droplevel(1)

    port_df = port_df.merge(spy_ret,
                            left_index=True,
                            right_index=True)

    return port_df

def plot_comparison(port_df):
    import matplotlib.ticker as mtick

    plt.style.use('ggplot')

    portfolio_cumulative_return = np.exp(np.log1p(port_df).cumsum())-1

    portfolio_cumulative_return.plot(figsize=(16,6))

    plt.title('Unsupervised Learning Trading Strategy Returns Over Time')

    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))

    plt.ylabel('Return')

    plt.show()

if __name__ == "__main__":

    df = get_ticker_data(end_date=pd.to_datetime('today'), load=False)
    import pdb
    pdb.set_trace()
    df = compute_indicators(df)
    feature_data = aggregate_filter(df, load=False)
    feature_data = monthly_returns_diff_time_horizons(feature_data)
    feature_data = get_fama_french_factors(feature_data, load=False)

    clustered_data = feature_data.dropna().groupby(level='date', group_keys=False).apply(kmeans_clustering)
    print(clustered_data)

    #plt.style.use('ggplot')

    #for i in clustered_data.index.get_level_values('date').unique().tolist()[-5:]:
    #    
    #    g = clustered_data.xs(i, level=0)
    #    
    #    plt.title(f'Date {i}')
    #    
    #    plot_clusters(g)

    fixed_dates = select_stocks(clustered_data)

    #import pdb
    #pdb.set_trace()

    stocks = clustered_data.index.get_level_values('ticker').unique().tolist()

    new_df = yf.download(tickers=stocks,
                        start=clustered_data.index.get_level_values('date').unique()[0]-pd.DateOffset(months=12),
                        end=clustered_data.index.get_level_values('date').unique()[-1],
                        auto_adjust=False)


    optimal_port, port_weights = portfolio_optimization(new_df, fixed_dates)

    combined_optimal_port = compare_sp_500(optimal_port).iloc[-365*2:]

    last_date_keys = list(port_weights.keys())[-3:]

    print("####### Allocations for last 3 months #######")
    for d in last_date_keys:
        print(f"Date: {d}")
        print(port_weights[d].T)

    plot_comparison(combined_optimal_port)

    import pdb
    pdb.set_trace()


