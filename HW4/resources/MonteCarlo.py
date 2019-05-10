from random import random
from scipy.stats import norm


def foo(X):
    print (X)
    print (X+1)
    return (X+1)



class RandomVariable():
    
    def __init__(self, transform_func= lambda x:x, description = 'no description',\
                 discrete = False, verbose = True):
        self.description = description
        self.transform = transform_func 
        self.discrete = discrete
        if verbose:
            print("testing random variable with distribution {},\n {}\n".format(self.description,int(self.sample())))
                  
    def sample(self):
        if self.discrete:
            return self.transform()
        else:
            x = random()
            return self.transform(x)
        
    def sample_repeated(self,n):
        while n>0:
            yield self.sample()
            n -= 1
        

class Simulator():
    def __init__(self, RV = RandomVariable('default random variable',verbose = False), cost_f = lambda x:x, desc = 'no description', verbose = True):
        self.cost_function = cost_f
        self.description = desc
        self.random_variable = RV
        if verbose:
            print("New simulator for {},\n with {} input".format(self.description, self.random_variable.description))
    def sample(self):
        return self.cost_function(self.random_variable.sample())
    def sample_repeated(self,n):
        for x in self.random_variable.sample_repeated(n):
            yield self.cost_function(x)
            
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
        action = self.policy(self.current_state,actions)
        new_state = self.transition_function(self.current_state,action)
        self.current_reward += self.reward_function(new_state,action)*self.damping_constant**(self.timestep)
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
    def sample_boolean_repeated(self,boolean_func,n):
        i = n
        while (i>0):
            i-=1
            yield self.check_boolean(boolean_func)
    

            
# inverse transforms for various common distributions
def inverse_exponential():
    assert(0)
    #TODO
def inverse_continuous_power_law(x,xmin,alpha):
    assert (alpha < -1 and xmin > 0)
    return xmin / ((1-x))**(1/(-1-alpha))

def inverse_discrete_power_law(x,xmin,alpha):
    assert(0)
    #TODO
    
# Useful statistics functions
def Calculate_ConfIntv(sum1,sumsq,n,confidence = .95):
    variance = (sumsq - sum1**2)/n
    dev = norm.ppf(confidence)*(variance**(.5))
    print ("We obtained a {conf}% confidence interval of\n{mean} +- {deviation}".format(mean = sum1/n,conf=confidence,deviation=dev))
    