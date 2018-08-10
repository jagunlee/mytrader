import numpy as np
import math

class Agent:
    STATE_DIM = 2  

    TRADING_CHARGE = 0.00015 
    TRADING_TAX = 0.003 

    # 행동
    ACTION_BUY = 0 
    ACTION_SELL = 1  
    ACTION_HOLD = 2  

    ACTIONS = [ACTION_BUY, ACTION_SELL]  # ACTION_HOLD까지 고려를 해보고 싶은 경우 리스트 안에 추가해주기만 하면 됨
    NUM_ACTIONS = len(ACTIONS) 

    def __init__(
        self, environment, min_trading_unit=0, 
        delayed_reward_threshold=.05):
        self.environment = environment 


        self.min_trading_unit = min_trading_unit  
        self.delayed_reward_threshold = delayed_reward_threshold  


        self.initial_balance = 0  # 초기 자본금
        self.balance = 0  # 현재 현금 잔고
        self.num_stocks = 0  # 보유 주식 수
        self.portfolio_value = 0  # balance + num_stocks * {현재 주식 가격}
        self.base_portfolio_value = 0  # 직전 학습 시점의 PV
        self.num_buy = 0  # 매수 횟수
        self.num_sell = 0  # 매도 횟수
        self.num_hold = 0  # 홀딩 횟수
        self.immediate_reward = 0  # 즉시 보상

        # Agent 클래스의 상태
        self.ratio_hold = 0  # 주식 보유 비율
        self.ratio_portfolio_value = 0  # 포트폴리오 가치 비율

    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    def set_balance(self, balance):
        self.initial_balance = balance

    def get_states(self):
        self.ratio_hold = self.num_stocks / int(
            self.portfolio_value / self.environment.get_next_price())
        self.ratio_portfolio_value = self.portfolio_value / self.initial_balance
        return (
            self.ratio_hold,
            self.ratio_portfolio_value
        )

    def decide_action(self, policy_network, sample, epsilon):
        #min/max trading unit이 없기 때문에 랜덤하게 행동할 경우 confidence를 0과 0.5사이로 랜덤하게 선택
        if np.random.rand() < epsilon:
            exploration = True
            action = np.random.randint(self.NUM_ACTIONS)
            confidence = np.random.rand()*0.5
        else:
            exploration = False
            probs = policy_network.predict(sample)  
            action = np.argmax(probs) 
            confidence = probs[action]
        return action, confidence, exploration

    def validate_action(self, action):
        validity = True
        if action == Agent.ACTION_SELL:
            # 주식 잔고가 있는지 확인 
            if self.num_stocks <= 0:
                validity = False
        return validity


    def act(self, action, confidence):
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_next_price()

        # 즉시 보상 초기화
        self.immediate_reward = 0

        # 매수
        if action == Agent.ACTION_BUY:
            # 매수할 단위는 confidence에 지수적으로 비례하게 설정, 0.001단위로 내림
            trading_unit = int((self.balance*math.exp(confidence-1)/(float(curr_price)*(1+self.TRADING_CHARGE)))*1000)/1000
            balance = self.balance - curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            self.balance -= invest_amount  # 보유 현금을 갱신
            self.num_stocks += trading_unit  # 보유 주식 수를 갱신
            self.num_buy += 1  # 매수 횟수 증가

            self.immediate_reward = 1

        # 매도
        elif action == Agent.ACTION_SELL:  # sell
            # 매도할 단위를 판단, 매수 방법과 같음
            trading_unit = int((self.num_stocks*math.exp(float(confidence)-1))*1000)/1000
            trading_unit = min(trading_unit, self.num_stocks)
            # 매도
            invest_amount = curr_price * (
                1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit
            self.num_stocks -= trading_unit  # 보유 주식 수를 갱신
            self.balance += invest_amount  # 보유 현금을 갱신
            self.num_sell += 1  # 매도 횟수 증가

            self.immediate_reward = 1

        # 홀딩
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  # 홀딩 횟수 증가
            self.immediate_reward = -1

        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + curr_price * self.num_stocks
        profitloss = (self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value

        # 지연 보상 판단
        if profitloss > self.delayed_reward_threshold:
            delayed_reward = 1
            # 목표 수익률 달성하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
        elif profitloss < -self.delayed_reward_threshold:
            delayed_reward = 0
            # 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
        else:
            delayed_reward = -1
        return self.immediate_reward, delayed_reward
