import numpy as np
import math


class Agent:
 
    STATE_DIM = 2  

   
    TRADING_CHARGE = 0.001
    TRADING_TAX = 0


    ACTION_BUY = 0
    ACTION_SELL = 1 
    ACTION_HOLD = 2 
    ACTIONS = [ACTION_BUY, ACTION_SELL]
    NUM_ACTIONS = len(ACTIONS) 

    def __init__(
        self, environment, delayed_reward_threshold=.05):
       
        self.environment = environment  

        self.delayed_reward_threshold = delayed_reward_threshold  

        
        self.initial_balance = 0  
        self.balance = 0 
        self.num_stocks = 0 
        self.portfolio_value = 0 
        self.base_portfolio_value = 0 
        self.num_buy = 0  
        self.num_sell = 0 
        self.num_hold = 0  
        self.immediate_reward = 0

   
        self.ratio_hold = 0 
        self.ratio_portfolio_value = 0 

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
        #print (self.portfolio_value)
        self.ratio_hold = self.num_stocks / int(
            self.portfolio_value / self.environment.get_price())
        self.ratio_portfolio_value = self.portfolio_value / self.base_portfolio_value
        return (
            self.ratio_hold,
            self.ratio_portfolio_value
        )

    def decide_action(self, policy_network, sample, epsilon):
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

    def validate_action(self, action,confidence):
        validity = True
        
        if confidence<0.2:
            validity=False
        
        if action == Agent.ACTION_SELL:
            if self.num_stocks <= 0:
                validity = False
        return validity

    def act(self, action, confidence):
        if not self.validate_action(action,confidence):
            action = Agent.ACTION_HOLD

     
        curr_price = self.environment.get_price()
        next_price=self.environment.get_next_price()
        #print (next_price)
        self.immediate_reward = 0

       
        if action == Agent.ACTION_BUY:
        
            trading_unit = self.balance*math.exp(confidence-1)/(float(next_price)*(1+self.TRADING_CHARGE))
            balance = self.balance - next_price * (1 + self.TRADING_CHARGE) * trading_unit
            if balance < 0:
                invest_amount = self.balance
                trading_unit = self.balance/(next_price*(1+self.TRADING_CHARGE))
            else:
                invest_amount = next_price * (1 + self.TRADING_CHARGE) * trading_unit
            self.balance -= invest_amount  
            self.num_stocks += trading_unit
            self.num_buy += 1 

       
        elif action == Agent.ACTION_SELL:
            
            trading_unit = self.num_stocks*math.exp(float(confidence)-1)
            invest_amount = next_price * (1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit
            self.num_stocks -= trading_unit  
            self.balance += invest_amount  
            self.num_sell += 1  

     
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  

      
        self.portfolio_value = self.balance + next_price * self.num_stocks
        profitloss = (
            (self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value)

     
        self.immediate_reward = 1 if profitloss >= 0 else -1

      
        if profitloss > self.delayed_reward_threshold:
            delayed_reward = 1
       
            self.base_portfolio_value = self.portfolio_value
        elif profitloss < -self.delayed_reward_threshold:
            delayed_reward = -1
          
            self.base_portfolio_value = self.portfolio_value
        else:
            delayed_reward = 0
        return self.immediate_reward, delayed_reward
