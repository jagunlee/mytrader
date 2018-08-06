from __future__ import (absolute_import, division, print_function,unicode_literals)
import data_manager
import pandas as pd
import os
import settings
import backtrader as bt
import datetime
import numpy as np
import os.path
import sys
import backtrader.indicators as btind


class TestStrategy(bt.Strategy):
    params = (('fast', 10), ('slow', 30),('maperiod', 15))
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        print('%s, %s' % (dt, txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        sma_fast = btind.SMA(period=self.p.fast)
        sma_slow = btind.SMA(period=self.p.slow)
        '''
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.maperiod)

        # Indicators for the plotting show
        bt.indicators.ExponentialMovingAverage(self.datas[0], period=25)
        bt.indicators.WeightedMovingAverage(self.datas[0], period=25,
                                            subplot=True)
        bt.indicators.StochasticSlow(self.datas[0])
        bt.indicators.MACDHisto(self.datas[0])
        rsi = bt.indicators.RSI(self.datas[0])
        bt.indicators.SmoothedMovingAverage(rsi, period=10)
        bt.indicators.ATR(self.datas[0], plot=False)
        '''
        self.buysig = btind.CrossOver(sma_fast, sma_slow)
        print('-----------------------------------------------')
        print("           This is test strategy               ")
        print('-----------------------------------------------')

    def next(self):
        self.log('Close, %.2f' % self.dataclose[0])
        if self.position.size:
            if self.buysig < 0:
                self.log('SELL CREATE, %.2f' % self.dataclose[0])
                self.sell()

        elif self.buysig > 0:
            self.log('BUY CREATE, %.2f' % self.dataclose[0])
            self.buy()
if __name__ == '__main__':

    cerebro = bt.Cerebro(stdstats=True)
    cerebro.addstrategy(TestStrategy)
    modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    data = bt.feeds.GenericCSVData(
        dataname='out.csv',

        fromdate=datetime.datetime(2017,11,24,23,0,0),
        todate=datetime.datetime(2017,12,16,10,0,0),
        timeframe=bt.TimeFrame.Minutes, compression=240,
        nullvalue=0.0,
        dtformat=('%Y-%m-%d %H:%M:%S'),
        datetime=0,
        high=1,
        low=2,
        open=3,
        close=4,
        volume=5,
        openinterest=-1
    )

    cerebro.adddata(data)
    cerebro.broker.setcash(100000.0)
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    cerebro.run()

    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    cerebro.plot(iplot=True)[0][0].savefig('plot.png')


