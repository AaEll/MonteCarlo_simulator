from random import random
from scipy.stats import norm


def foo(X):
    print (X)
    print (X+1)
    return (X+1)



class RandomVariable():
    
    def __init__(self, inv_t= lambda x:x, desc = 'no description',\
                 verbose = True):
        self.description = desc
        self.inverse_transform = inv_t
        if verbose:
            print("testing random variable with distribution {},\n {}\n".format(self.description,self.sample()))
                  
    def sample(self):
        x = random()
        return self.inverse_transform(x)
        
    def sample_repeated(self,n):
        while n>0:
            yield self.sample()
            n -= 1

class Simulator():
    def __init__(self, RV = RandomVariable(), cost_f = lambda x:x, desc = 'no description', verbose = True):
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

            
# inverse transforms for various common distributions
def inverse_exponential():
    assert(0)
    #TODO
def inverse_continuous_power_law(x,xmin,alpha):
    assert (alpha != -1 and xmin > 0)
    return (xmin*((1-x)**(1/(alpha+1))))

def inverse_discrete_power_law(x,xmin,alpha):
    assert(0)
    #TODO
    
# Useful statistics functions
def Calculate_ConfIntv(sum1,sumsq,n,confidence = .95):
    variance = (sumsq - sum1**2)/n
    dev = norm.ppf(confidence)*(variance**(.5))
    print ("We obtained a {conf}% confidence interval of\n{mean} +- {deviation}".format(mean = sum1/n,conf=confidence,deviation=dev))