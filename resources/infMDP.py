import numpy as np
import math
import random

class infinite_state_MDP():
    def __init__(self, actions, transition_function, reward_function, termination_function, initial_state_func, damping_constant, policy):
        self.actions = actions
        self.termination_function = termination_function
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
        self.step_reward = self.reward_function(new_state,action)
        self.current_reward += self.step_reward*self.damping_constant**(self.timestep)
        self.timestep +=1
        self.current_state = new_state
    
    def run_mdp(self):
        self.current_state = self.initial_state_func()
        self.current_reward = 0
        self.timestep = 0
        
        while not self.termination_function(self.current_state,self.timestep):
            self.run_step()
        return self.current_reward
    
    def sample_repeated(self,n):
        i = n
        while (i>0):
            i-=1
            X = self.run_mdp()
            global standard_dev
            standard_dev += X**2
            yield X
            
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
    def sample_boolean_repeated(self,boolean_func,n):
        i = n
        while (i>0):
            i-=1
            yield self.check_boolean(boolean_func)
    def Yield_run_step(self):
        action = self.policy(self.current_state,self.actions)
        yield action
        new_state = self.transition_function(self.current_state,action)
        yield new_state
        self.step_reward = self.reward_function(new_state,action)
        yield self.step_reward
        self.current_reward += self.step_reward*self.damping_constant**(self.timestep)
        self.timestep +=1
        self.current_state = new_state
    
    def Yield_run_mdp(self):
        self.current_state = self.initial_state_func()
        self.current_reward = 0
        self.timestep = 0
        yield self.current_state
        while not self.termination_function(self.current_state,self.timestep):
            for item in self.Yield_run_step():
                yield item
                    


def TemporalDifferenceParameters(no_dim,basis,delta_policy,func_policy,no_episodes,alpha,MDP):
    w = np.zeros(no_dim)
    i = 0
    while i < no_episodes:
        i+=1
        MDP.current_state = MDP.initial_state_func()
        MDP.current_reward = 0
        MDP.timestep = 0
        prev_state = MDP.current_state
        while not MDP.termination_function(MDP.current_state,MDP.timestep):
            MDP.run_step()
            w += alpha*(MDP.step_reward\
                 +func_policy(w,MDP.current_state,basis)*MDP.damping_constant\
                 -func_policy(w,prev_state,basis))*delta_policy(w,prev_state,basis)
            print(w.shape)
            print(delta_policy(w,prev_state,basis).shape)
            assert(0)
            prev_state = MDP.current_state
    return w

def test_TDParameters(no_dim,basis,delta_policy,func_policy,no_episodes,alpha,MDP,w):
    mean_TD = 0
    n = 0
    i = 0 
    while i < no_episodes:
        i+=1
        MDP.current_state = MDP.initial_state_func()
        MDP.current_reward = 0
        MDP.timestep = 0
        prev_state = MDP.current_state
        while not MDP.termination_function(MDP.current_state,MDP.timestep):
            MDP.run_step()
            mean_TD += (MDP.step_reward+MDP.damping_constant*func_policy(w,MDP.current_state, basis)\
                           -func_policy(w,prev_state,basis))**2
            prev_state = MDP.current_state
            n+=1
    return float(mean_TD/n)

def linear_policy(weights, states, basis):
    return np.dot(weights, basis(states))




_w_ = None
_H_ = None
_X_ = None

def epsilon_greedy_policy(state,actions, _basis_ ,_epsilon_ =.04):
    p = np.random.random()
    if p<_epsilon_:
        return random.choice(actions)
    else:
        return actions[np.argmax(np.matmul(_basis_(state),_w_))]
def delta_linear(state_norm,action):
    global _X_
    _X_.fill(0)
    _X_[:,action] = state_norm
    return _X_

def softmax_AC_policy(state,actions,basis):
    P = np.ones(len(actions))*np.e
    P = np.power(P,np.dot(_T_.T,basis(state)))
    P = P/np.sum(P)
    return np.random.choice(actions,p = P)
def delta_softmax_AC(state,action,basis,no_actions):
    P = np.ones((1,no_actions))*np.e
    P = np.power(P,np.dot(_T_.T,basis(state)))
    P = -1*P/np.sum(P)
    P[:,action] = P[:,action]+1
    Phi = basis(state)
    return np.matmul(Phi.reshape((len(Phi), 1)),P)

  
def Sarsa(no_episodes,no_dim,no_actions,action_id,alpha,basis,delta_q,MDP):
    global _w_
    global _X_
    _w_ = np.zeros((no_dim,no_actions))
    _X_ = np.zeros((no_dim,no_actions))
    n = 0
    while n<no_episodes:
        MDP.policy = lambda state,actions: epsilon_greedy_policy(state,actions,basis,.2/(n+1))
        n = n+1
        Episode = MDP.Yield_run_mdp() 
        s  = Episode.__next__()
        a = action_id[Episode.__next__()]
        current_reward = 0
        t = 0

        while True:
            try:
                s_ = Episode.__next__()
                r_ = Episode.__next__()
                a_ = action_id[Episode.__next__()]  # StopIteration Here 
                _w_ = _w_+ alpha*(r_ + MDP.damping_constant*np.dot(basis(s_),_w_[:,a_])-np.dot(basis(s),_w_[:,a]))*delta_q(basis(s),a)
                a = a_
                s = s_
                current_reward += r_*(MDP.damping_constant**t)
                t+=1
            except StopIteration:
                _w_ = _w_+ alpha*(r_ + MDP.damping_constant*np.dot(basis(s_),_w_[:,a_])-np.dot(basis(s),_w_[:,a]))*delta_q(basis(s),a)
                current_reward += r_*(MDP.damping_constant**t)  
                break
        yield current_reward

def Sarsa_Lambda(lam,no_episodes,no_dim,no_actions,action_id,alpha,basis,delta_q,MDP):
    global _w_
    global _X_
    _w_ = np.zeros((no_dim,no_actions))
    _X_ = np.zeros((no_dim,no_actions))
    n = 0
    while n<no_episodes:
        MDP.policy = lambda state,actions: epsilon_greedy_policy(state,actions,basis,.2/(n+1))
        n = n+1
        Episode = MDP.Yield_run_mdp() 
        s  = Episode.__next__()
        a = action_id[Episode.__next__()]
        current_reward = 0
        t = 0
        e = 0
        while True:
            try:
                s_ = Episode.__next__()
                r_ = Episode.__next__()
                a_ = action_id[Episode.__next__()]  # StopIteration Here
                e = MDP.damping_constant*lam*e+delta_q(basis(s),a)
                d = r_ + MDP.damping_constant*np.dot(basis(s_),_w_[:,a_])-np.dot(basis(s),_w_[:,a])
                _w_ = _w_+ alpha*d*e
                a = a_
                s = s_
                current_reward += r_*(MDP.damping_constant**t)
                t+=1
            except StopIteration:
                e = MDP.damping_constant*lam*e+delta_q(basis(s),a)
                d = r_ -np.dot(basis(s),_w_[:,a])
                _w_ = _w_+ alpha*d*e
                current_reward += r_*(MDP.damping_constant**t)  
                break
        yield current_reward
        
def Q(no_episodes,no_dim,no_actions,action_id,alpha,basis,delta_q,MDP):
    global _w_
    global _H_
    global _X_
    _H_ = []
    _w_ = np.zeros((no_dim,no_actions))
    _X_ = np.zeros((no_dim,no_actions))

    n = 0
    while n<no_episodes:
        MDP.policy = lambda state,actions: epsilon_greedy_policy(state,actions,basis,.2/(n+1))
        n = n+1
        Episode = MDP.Yield_run_mdp() # Test this line, I am assuming lazy evaluation
        s  = Episode.__next__()
        a = action_id[Episode.__next__()]
        current_reward = 0
        t = 0
        while True:
            try:
                s_ = Episode.__next__()
                r_ = Episode.__next__()
                a_ = action_id[Episode.__next__()]
                _w_ = _w_+ alpha*(r_ + MDP.damping_constant*np.max(np.matmul(basis(s_),_w_))-np.dot(basis(s),_w_[:,a]))*delta_q(basis(s),a)
                current_reward += r_*(MDP.damping_constant**t) 
                a = a_
                s = s_
                t+=1
            except StopIteration:
                _w_ = _w_+ alpha*(r_ - np.dot(basis(s),_w_[:,a]))*delta_q(basis(s),a)
                current_reward += r_*(MDP.damping_constant**t)
                break
            #yield(s,a,r_)
        yield current_reward
def Q_Lambda(lam,no_episodes,no_dim,no_actions,action_id,alpha,basis,delta_q,MDP):
    global _w_
    global _X_
    #global _H_
    _w_ = np.zeros((no_dim,no_actions))
    _X_ = np.zeros((no_dim,no_actions))    
    #_H_ = []
    n = 0
    while n<no_episodes:
        MDP.policy = lambda state,actions: epsilon_greedy_policy(state,actions,basis,.2/(n+1))
        n = n+1
        Episode = MDP.Yield_run_mdp() # Test this line, I am assuming lazy evaluation
        s  = Episode.__next__()
        a = action_id[Episode.__next__()]
        current_reward = 0
        t = 0
        e = 0
        while True:
            try:
                s_ = Episode.__next__()
                r_ = Episode.__next__()
                a_ = action_id[Episode.__next__()]
                e = MDP.damping_constant*lam*e+delta_q(basis(s),a)
                d = r_ + MDP.damping_constant*np.max(np.matmul(basis(s_),_w_))-np.dot(basis(s),_w_[:,a])
                _w_ = _w_+ alpha*d*e
                current_reward += r_*(MDP.damping_constant**t) 
                a = a_
                s = s_
                t+=1
            except StopIteration:
                e = MDP.damping_constant*lam*e+delta_q(basis(s),a)
                d = r_ - np.dot(basis(s),_w_[:,a])
                _w_ = _w_+ alpha*d*e
                current_reward += r_*(MDP.damping_constant**t) 
                break
        yield current_reward
        
        
def actor_critic(alpha,beta,lam,no_episodes,no_dim,no_actions,action_id,basis,MDP):
    global _w_
    global _T_
    #global _X_
    #_X_ = np.zeros((no_dim,no_actions))
    _T_ = np.zeros((no_dim,no_actions))
    _w_ = np.zeros((no_dim))
    n = 0
    while n<no_episodes:
        MDP.policy = lambda state,actions: softmax_AC_policy(state,actions,basis)
        n = n+1
        Episode = MDP.Yield_run_mdp() 
        s  = Episode.__next__()
        a = action_id[Episode.__next__()]
        current_reward = 0
        t = 0
        ev = np.zeros(no_dim)
        et = np.zeros((no_dim,no_actions))
        while True:
            try:
                s_ = Episode.__next__()
                r_ = Episode.__next__()
                a_ = action_id[Episode.__next__()]  # StopIteration Here
                ev = MDP.damping_constant*lam*ev+basis(s)
                d = r_ + MDP.damping_constant*np.dot(basis(s_),_w_)-np.dot(basis(s),_w_)
                _w_ = _w_+ alpha*d*ev
                et = MDP.damping_constant*lam*et+delta_softmax_AC(s,a,basis,no_actions)
                _T_=_T_ +beta*d*et
                a = a_
                s = s_
                current_reward += r_*(MDP.damping_constant**t)
                t+=1
            except StopIteration:
                ev = MDP.damping_constant*lam*ev+basis(s)
                d = r_ + MDP.damping_constant*np.dot(basis(s_),_w_)-np.dot(basis(s),_w_)
                _w_ = _w_+ alpha*d*ev
                et = MDP.damping_constant*lam*et+delta_softmax_AC(s,a,basis,no_actions)
                _T_=_T_ +beta*d*et
                current_reward += r_*(MDP.damping_constant**t)  
                break
        yield current_reward
