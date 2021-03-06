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
_v_ = None
_H_ = None
_T_ = None
def epsilon_greedy_policy(state,actions,_epsilon_ =.04):
    p = np.random.random()
    if p<_epsilon_:
        return random.choice(actions)
    else:
        actionset = np.argwhere(_q_[state,:] == np.amax(_q_[state,:])).flatten()
        return actions[np.random.choice(actionset)]
def softmax_greedy_policy(state,actions,sigma):
    P = np.ones(len(actions))*np.e
    P = np.power(P,_q_[state,:]*sigma)
    P = P/np.sum(P)
    return np.random.choice(actions,p = P)
def softmax_AC_policy(state,actions):
    P = np.ones(len(actions))*np.e
    P = np.power(P,_T_[state,:])
    P = P/np.sum(P)
    return np.random.choice(actions,p = P)
def delta_softmax_AC(state,action,no_actions):
    global _X_
    _X_.fill(0)
    P = np.ones(no_actions)*np.e
    P = np.power(P,_T_[state,:])
    P = -1*P/np.sum(P)
    P[action] = P[action]+1
    _X_[state,:] = P
    return _X_
                 
def Sarsa_tabular(no_episodes,no_states,no_actions,action_id,alpha,MDP,softmax = False, sigma = .2):
    global _q_
    _q_ = np.zeros((no_states,no_actions))
    n = 0
    while n<no_episodes:
        if softmax :
            MDP.policy = lambda state,actions: softmax_greedy_policy(state,actions,n*sigma)            
        else:
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

def Q_tabular(no_episodes,no_states,no_actions,action_id,alpha,MDP,softmax = False, sigma = .2):
    global _q_
    _q_ = np.zeros((no_states,no_actions))
    n = 0
    while n<no_episodes:
        if softmax :
            MDP.policy = lambda state,actions: softmax_greedy_policy(state,actions,n*sigma)            
        else:
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

def Sarsa_Lambda_tabular(lam,no_episodes,no_states,no_actions,action_id,alpha,MDP,softmax = False, sigma = .2):
    global _q_
    _q_ = np.zeros((no_states,no_actions))
    n = 0
    while n<no_episodes:
        if softmax :
            MDP.policy = lambda state,actions: softmax_greedy_policy(state,actions,n*sigma)            
        else:
            MDP.policy = lambda state,actions: epsilon_greedy_policy(state,actions,1/(n+1))
        n = n+1
        Episode = MDP.Yield_run_mdp() # Test this line, I am assuming lazy evaluation
        s = Episode.__next__()
        a = action_id[Episode.__next__()]
        current_reward = 0
        t = 0
        e = 0

        while True:
            try:
                s_ = Episode.__next__()
                r_ = Episode.__next__()
                a_ = action_id[Episode.__next__()]  # StopIteration Here 
                e = MDP.damping_constant*lam*e+1
                d = r_ + MDP.damping_constant*_q_[s_,a_]-_q_[s,a]
                _q_[s,a] = _q_[s,a]+ alpha*d*e
                a = a_
                s = s_
                current_reward += r_*(MDP.damping_constant**t)
                t+=1
            except StopIteration:
                e = MDP.damping_constant*lam*e+1
                d = r_ -_q_[s,a]
                _q_[s,a] = _q_[s,a]+ alpha*d*e
                current_reward += r_*(MDP.damping_constant**t)
                break
        yield current_reward

def Q_Lambda_tabular(lam,no_episodes,no_states,no_actions,action_id,alpha,MDP,softmax = False, sigma = .2):
    global _q_
    #global _H_
    #_H_ = []
    _q_ = np.zeros((no_states,no_actions))
    n = 0
    while n<no_episodes:
        if softmax :
            MDP.policy = lambda state,actions: softmax_greedy_policy(state,actions,n*sigma)            
        else:
            MDP.policy = lambda state,actions: epsilon_greedy_policy(state,actions,1/(n+1))
        n = n+1
        Episode = MDP.Yield_run_mdp() # Test this line, I am assuming lazy evaluation
        s = Episode.__next__()
        a = action_id[Episode.__next__()]
        current_reward = 0
        t = 0
        e = 0
        while True:
            try:
                s_ = Episode.__next__()
                r_ = Episode.__next__()
                #_H_.append((s,a,r_,e))
                a_ = action_id[Episode.__next__()]                
                e = MDP.damping_constant*lam*e+1
                d = r_ + MDP.damping_constant*np.max(_q_[s_,:])-_q_[s,a]
                _q_[s,a] = _q_[s,a]+ alpha*d*e
                current_reward += r_*(MDP.damping_constant**t)     
                a = a_
                s = s_
                t+=1
            except (StopIteration):
                e = MDP.damping_constant*lam*e+1
                d = r_ - _q_[s,a]
                _q_[s,a] = _q_[s,a]+ alpha*d*e
                current_reward += r_*(MDP.damping_constant**t)
                break
        yield current_reward

def actor_critic(alpha,beta,lam,no_episodes,no_states,no_actions,action_id,MDP):
    global _v_
    global _T_
    global _X_
    global _H_
    _H_ = []
    _v_ = np.zeros((no_states))
    _T_ = np.zeros((no_states,no_actions))
    _X_ = np.zeros((no_states,no_actions))
    n = 0
    while n<no_episodes:
        MDP.policy = lambda state,actions: softmax_AC_policy(state,actions)            
        n = n+1
        Episode = MDP.Yield_run_mdp()
        s = Episode.__next__()
        a = action_id[Episode.__next__()]
        current_reward = 0
        t = 0
        ev = np.zeros(no_states)
        et = np.zeros((no_states,no_actions))
        while True:
            try:
                s_ = Episode.__next__()
                r_ = Episode.__next__()
                a_ = action_id[Episode.__next__()]                
                ev = MDP.damping_constant*lam*ev
                ev[s] += 1
                d = r_ + MDP.damping_constant*_v_[s_]-_v_[s]
                _v_ = _v_+ alpha*d*ev
                et = MDP.damping_constant*lam*et + delta_softmax_AC(s,a,no_actions)
                _T_ = _T_ + beta*d*et
                current_reward += r_*(MDP.damping_constant**t) 
                a = a_
                s = s_
                t+=1
            except (StopIteration):
                ev = MDP.damping_constant*lam*ev
                ev[s] += 1
                d = r_ -_v_[s]
                _v_ = _v_+ alpha*d*ev
                et = MDP.damping_constant*lam*et + delta_softmax_AC(s,a,no_actions)
                _T_ = _T_ + beta*d*et
                current_reward += r_*(MDP.damping_constant**t)     
                break
        yield current_reward        

    