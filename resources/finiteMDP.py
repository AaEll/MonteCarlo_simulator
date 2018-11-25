import numpy as np
import math
import random

class MDP():
    termial_absorbing_state = "inf"
    def __init__(self, states, actions, transition_function, reward_function, initial_state_func, damping_constant, policy):
        self.states = states
        self.actions = actions
        self.transition_function = transition_function
        self.reward_function = reward_function
        self.initial_state_func = initial_state_func
        self.damping_constant = damping_constant
        self.policy = policy
        
        self.current_state = self.initial_state_func()
        self.current_reward = 0
        self.timestep = 0

        
    def run_step(self):
        action = self.policy(self.current_state,self.actions)
        new_state = self.transition_function(self.current_state,action)
        self.step_reward = self.reward_function(self.current_state,action,new_state)
        self.current_reward += self.step_reward*self.damping_constant**(self.timestep)
        self.timestep +=1
        self.current_state = new_state
    
    def run_mdp(self):
        self.current_state = self.initial_state_func()
        self.current_reward = 0
        self.timestep = 0
        
        while self.current_state != self.termial_absorbing_state:
            self.run_step()
        return self.current_reward
    
    def sample_repeated(self,n):
        i = n
        while (i>0):
            i-=1
            yield self.run_mdp()

            
    def check_boolean(self,boolean_func):
        self.current_state = self.initial_state_func()
        self.timestep = 0
        
        while self.current_state != self.termial_absorbing_state:
            self.run_step()            
            boolean = boolean_func(self)
            if boolean is None:
                pass
            else:
                return boolean
        return False
    
    def Yield_run_mdp(self):
        self.current_state = self.initial_state_func()
        yield(self.current_state)
        self.current_reward = 0
        self.timestep = 0
        while self.current_state != self.termial_absorbing_state:
            for item in self.Yield_run_step():
                yield item
    def Yield_run_step(self):
        action = self.policy(self.current_state,self.actions)
        yield action
        new_state = self.transition_function(self.current_state,action)
        yield new_state
        self.step_reward = self.reward_function(self.current_state,action,new_state)
        yield self.step_reward
        self.current_reward += self.step_reward*self.damping_constant**(self.timestep)
        self.timestep +=1
        self.current_state = new_state



def TemporalDifference(no_episodes,alpha,MDP):
    V_pi = np.zeros(len(MDP.states))
    i = 0
    while i < no_episodes:
        i+=1
        MDP.current_state = MDP.initial_state_func()
        MDP.current_reward = 0
        MDP.timestep = 0        
        prev_state = MDP.current_state
        while MDP.current_state != MDP.termial_absorbing_state:
            MDP.run_step()
            if MDP.current_state == MDP.termial_absorbing_state:
                V_pi[prev_state] += alpha*(MDP.step_reward\
                        -V_pi[prev_state])
            else:
                V_pi[prev_state] += alpha*(MDP.step_reward\
                        +V_pi[MDP.current_state]*MDP.damping_constant\
                        -V_pi[prev_state])
            prev_state = MDP.current_state
    return V_pi
def TestTD(no_episodes,alpha,MDP,V_pi):
    i = 0
    meansq_TD = 0
    n = 0
    while i < no_episodes:
        i+=1
        MDP.current_state = MDP.initial_state_func()
        MDP.current_reward = 0
        MDP.timestep = 0        
        prev_state = MDP.current_state
        while MDP.current_state != MDP.termial_absorbing_state:
            MDP.run_step()
            if MDP.current_state == MDP.termial_absorbing_state:
                meansq_TD += (MDP.step_reward\
                        -V_pi[prev_state])**2
            else:            
                meansq_TD += (MDP.step_reward\
                        +V_pi[MDP.current_state]*MDP.damping_constant\
                        -V_pi[prev_state])**2
                
            n+=1
            prev_state = MDP.current_state
                         
    return meansq_TD/n

_q_ = None
def epsilon_greedy_policy(state,actions,_epsilon_ =.04):
    p = np.random.random()
    if p<_epsilon_:
        return random.choice(actions)
    else:
        return actions[np.argmax(_q_[state,:])]
def Sarsa_tabular(no_episodes,no_states,no_actions,action_id,alpha,MDP):
    global _q_
    _q_ = np.zeros((no_states,no_actions))
    n = 0
    while n<no_episodes:
        MDP.policy = lambda state,actions: epsilon_greedy_policy(state,actions,1/(n+1))
        n = n+1
        Episode = MDP.Yield_run_mdp() # Test this line, I am assuming lazy evaluation
        s = Episode.__next__()
        a = action_id[Episode.__next__()]
        current_reward = 0
        t = 0

        while True:
            try:
                s_ = Episode.__next__()
                r_ = Episode.__next__()
                a_ = action_id[Episode.__next__()]  # StopIteration Here 
                _q_[s,a] = _q_[s,a]+ alpha*(r_ + MDP.damping_constant*_q_[s_,a_]-_q_[s,a])
                a = a_
                s = s_
                current_reward += r_*(MDP.damping_constant**t)
                t+=1
            except StopIteration:
                _q_[s,a] = _q_[s,a]+ alpha*(r_ -_q_[s,a])
                current_reward += r_*(MDP.damping_constant**t)
                break
        yield current_reward

def Q_tabular(no_episodes,no_states,no_actions,action_id,alpha,MDP):
    global _q_
    _q_ = np.zeros((no_states,no_actions))
    n = 0
    while n<no_episodes:
        MDP.policy = lambda state,actions: epsilon_greedy_policy(state,actions,1/(n+1))
        n = n+1
        Episode = MDP.Yield_run_mdp() # Test this line, I am assuming lazy evaluation
        s = Episode.__next__()
        a = action_id[Episode.__next__()]
        current_reward = 0
        t = 0
        while True:
            try:
                s_ = Episode.__next__()
                r_ = Episode.__next__()
                a_ = action_id[Episode.__next__()]
                _q_[s,a] = _q_[s,a]+ alpha*(r_ + MDP.damping_constant*np.max(_q_[s_,:])-_q_[s,a])
                current_reward += r_*(MDP.damping_constant**t)                
                a = a_
                s = s_
                t+=1
            except StopIteration:
                _q_[s,a] = _q_[s,a]+ alpha*(r_-_q_[s,a])  
                current_reward += r_*(MDP.damping_constant**t)
                break
        yield current_reward


    
    
    
    
    
    
    

