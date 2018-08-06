import pandas as pd
import talib
import os
import datetime
import numpy as np

def load_chart_data(filename):
    df=pd.read_csv('{0}_1h.csv'.format(filename))
    df=df.set_index('o_t')
    df=df[['o_p','h_p','l_p','c_p','v']]
    df.columns=['open', 'high', 'low', 'close', 'volume']
    df=df.reindex(index=df.index[::-1])
    df.head()
    check = 'BTC'
    path_dir = 'binance'
    file_list = os.listdir(path_dir)
    cut=len(df)
    for files in file_list:
        if files.find(check) is not -1 and files[len(check)] != check and files.find(filename) is -1:
            df1=pd.read_csv('binance/{0}'.format(files))
            df1.set_index('o_t')
            df1 = df1.reindex(index=df1.index[::-1])
            df1.head()
            name=files[:]
            if len(df)<len(df1):
                df1=df1[-cut:]
            df['{0}_c_p'.format(name.replace('BTC_1h.csv', ''))] = np.nan
            df['{0}_c_p'.format(name.replace('BTC_1h.csv',''))] = df1[['o_t','c_p']].set_index(('o_t'))
            #df['{0}_c_p'.format(name.replace('BTC_1h.csv', ''))] =df['{0}_c_p'.format(name.replace('BTC_1h.csv',''))].mul(df['close'])
    df=df.reindex(index=df.index[::-1])
    df.head()
    df['o_t']=df.index
    df['o_t']=pd.to_datetime(df['o_t'],unit='ms')
    return df

def preprocess(chart_data):
    prep_data = chart_data
    windows = [5, 10, 20, 60, 120]
    for window in windows:
        prep_data['close_ma{}'.format(window)] = prep_data['close'].rolling(window).mean()
        prep_data['volume_ma{}'.format(window)] = (
            prep_data['volume'].rolling(window).mean())
    return prep_data


def build_training_data(prep_data):
    training_data = prep_data

    training_data['high_close_ratio'] = \
        (training_data['high'].values - training_data['close'].values) / \
        training_data['close'].values
    training_data['low_close_ratio'] = \
        (training_data['low'].values - training_data['close'].values) / \
        training_data['close'].values
    training_data['close_lastclose_ratio'] = np.zeros(len(training_data))
    training_data.loc[1502946000000:, 'close_lastclose_ratio'] = \
        (training_data['close'][1:].values - training_data['close'][:-1].values) / \
        training_data['close'][:-1].values
    training_data['volume_lastvolume_ratio'] = np.zeros(len(training_data))
    training_data.loc[1502946000000:, 'volume_lastvolume_ratio'] = \
        (training_data['volume'][1:].values - training_data['volume'][:-1].values) / \
        training_data['volume'][:-1]\
            .replace(to_replace=0, method='ffill') \
            .replace(to_replace=0, method='bfill').values
    training_data['ema12'] = talib.EMA(training_data['close'].values, 12)
    training_data['ema26'] = talib.EMA(training_data['close'].values, 26)
    upper, middle, lower = talib.BBANDS(training_data['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    training_data['dn'] = lower
    training_data['mavg'] = middle
    training_data['up'] = upper
    training_data['pctB'] = (training_data.close - training_data.dn) / (training_data.up - training_data.dn)
    
    #rsi14 = talib.RSI(training_data['close'].reshape((len(training_data['close'],1))), 14)
    #training_data['rsi14'] = rsi14
    
    macd, macdsignal, macdhist = talib.MACD(training_data['close'], 12, 26, 9)
    training_data['macd'] = macd
    training_data['signal'] = macdsignal
    training_data['cci'] = talib.CCI(training_data['high'].values, training_data['low'], training_data['close'], 7)
    
    windows = [5, 10, 20, 60, 120]
    for window in windows:
        training_data['close_ma%d_ratio' % window] = \
            (training_data['close'] - training_data['close_ma%d' % window]) / \
            training_data['close_ma%d' % window]
        training_data['volume_ma%d_ratio' % window] = \
            (training_data['volume'] - training_data['volume_ma%d' % window]) / \
            training_data['volume_ma%d' % window]
    training_data.index=training_data['o_t']
    return training_data


# chart_data = pd.read_csv(fpath, encoding='CP949', thousands=',', engine='python')
