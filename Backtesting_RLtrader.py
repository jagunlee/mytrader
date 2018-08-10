'''
Made by Jagun, POSTECH
강화학습으로 만들어진 h5모델을 strategy로 돌릴 수 있게 만든 backtesting 코드
'''
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
import os
import locale
import logging
import numpy as np
import settings
from environment import Environment
from agent_custom import Agent
from policy_network import PolicyNetwork
import math

class TestStrategy(bt.Strategy):
    #strategy 만들기
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        print('%s, %s' % (dt, txt))

    def __init__(self):
        #필요 데이터 이름 저장 파트
        stock_code='BTCUSDT'
        model_ver='20180806055008'  #write the number in the model name
        self.dataclose = self.datas[0].close
        log_dir = os.path.join(settings.BASE_DIR, 'logs/%s' % stock_code)
        timestr = settings.get_time_str()
        if not os.path.exists('logs/%s' % stock_code):
            os.makedirs('logs/%s' % stock_code)
        file_handler = logging.FileHandler(filename=os.path.join(
            log_dir, "%s_%s.log" % (stock_code, timestr)), encoding='utf-8')
        stream_handler = logging.StreamHandler()
        file_handler.setLevel(logging.DEBUG)
        stream_handler.setLevel(logging.INFO)
        logging.basicConfig(format="%(message)s",
                        handlers=[file_handler, stream_handler], level=logging.DEBUG)

    
        chart_data = data_manager.load_chart_data(
            os.path.join(settings.BASE_DIR,
                     '{}'.format(stock_code)))
        prep_data = data_manager.preprocess(chart_data)
        training_data = data_manager.build_training_data(prep_data)

   
        training_data = training_data.loc['2018-07-01 01:00:00':]
    


        features_chart_data = ['o_t', 'open', 'high', 'low', 'close', 'volume']
        chart_data = training_data[features_chart_data]

        #you can change the features training data
        features_training_data = [
             
         'high_close_ratio', 'low_close_ratio',
        'close_lastclose_ratio', 'volume_lastvolume_ratio',
        'close_ma5_ratio', 'volume_ma5_ratio',
        'close_ma10_ratio', 'volume_ma10_ratio',
        'close_ma20_ratio', 'volume_ma20_ratio',
        'close_ma60_ratio', 'volume_ma60_ratio',
        'close_ma120_ratio', 'volume_ma120_ratio',
        'ema12','ema26','dn','mavg','up','pctB','macd','signal','cci'
    ]

        training_data = training_data[features_training_data]  
        training_data = training_data.dropna(axis=1)
        chart_data=chart_data.dropna(axis=1)
        lr=0.1
        self.TRADING_TAX =0
        self.TRADING_CHARGE=0
        self.stock_code = stock_code  
        self.chart_data = chart_data
        self.environment = Environment(chart_data)  
        self.agent = Agent(self.environment)
        self.training_data = training_data  
        self.sample = None
        self.training_data_idx = -1
        self.num_features = self.training_data.shape[1]
        self.policy_network = PolicyNetwork(
            input_dim=self.num_features, output_dim=self.agent.NUM_ACTIONS, lr=lr)
        model_path=os.path.join(settings.BASE_DIR,'models/{}/model_{}.h5'.format(stock_code,model_ver))
        self.policy_network.load_model(model_path=model_path)
        self.agent.set_balance(self.broker.getcash())
        self.num_stocks=0
        df=pd.read_csv('out.csv')
        self.stopit=df.loc[7692]['c_p'] #2903 4135 7692
        #원하는 범위의 데이터 중 가장 마지막 데이터의 번호-1을 작성
        #예를 들어 가장 마지막까지 돌리려면 마지막 숫자인 7693 -1인 7692를 넣어서 돌리기
        
        
    def next(self):
        #가장 최근 라인을 자동으로 넘어가주며 행동을 결정해주는 파트
        self.log('Close, %.2f' % self.dataclose[0])
        self.environment.observe()
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            self.sample = self.training_data.iloc[self.training_data_idx].tolist()
        next_sample= self.sample
        if next_sample is None:
            return
        action, confidence, exploration = self.agent.decide_action(
                    self.policy_network, self.sample, self.epsilon)
        if self.dataclose[0]==self.stopit:
            next_price=self.dataclose[0]
        else:
            next_price=self.dataclose[1]
        if action==Agent.ACTION_SELL:
            trading_unit = self.num_stocks*math.exp(float(confidence)-1)
            invest_amount = next_price * (1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit
            self.num_stocks -= trading_unit  
            self.sell(size=trading_unit)
            self.log('sell')

        elif action==Agent.ACTION_BUY:
            
            trading_unit = self.broker.getcash()*math.exp(confidence-1)/(float(next_price)*(1+self.TRADING_CHARGE))  
            self.num_stocks += trading_unit
            self.buy(size=trading_unit)
            self.log('buy')
        self.log('value : %.2f , cash : %.2f , action : %.2f , confidence : %.2f , stock : %.2f' % (self.broker.getvalue(),self.broker.getcash(),action, confidence,self.num_stocks))
        #어떤 행동을 어떤 confidence로 선택했는지 매번 로그로 나오게 할 수 있음
        
        
        
if __name__ == '__main__':

    cerebro = bt.Cerebro(stdstats=True)
    cerebro.addstrategy(TestStrategy)
    modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    data = bt.feeds.GenericCSVData(
        dataname='out.csv',

        fromdate=datetime.datetime(2018,6,30,23,0,0),
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
    '''
    아래의 데이터는 각각 가장 하락세인 500시간, 가장 상승세인 500시간 구간
    위의 fromdate를 지우고 아래 두개씩 넣어준 뒤 위의 stopit구간도 업데이트 필요
    fromdate=datetime.datetime(2018,1,15,2,0,0),
    todate=datetime.datetime(2018,2,5,19,0,0),
    
    fromdate=datetime.datetime(2017,11,24,23,0,0),
    todate=datetime.datetime(2017,12,16,10,0,0),
    '''
    cerebro.adddata(data)
    cerebro.broker.setcash(100000.0) #초기자금 설정 가능
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    cerebro.run()

    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())            
    cerebro.plot(iplot=True)[0][0].savefig('plot3.png') #plot3.png 파일로 저장