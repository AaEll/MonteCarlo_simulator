import random

def foo(X):
    print (X)
    print (X+1)
    return (X+1)

class RandomVariable():
    
    def __init__(self, dist = 'uniform on (0,1)',\
                 inv_t= lambda x:x,\
                 verbose = True):
        self.distribution = dist
        self.inverse_transform = inv_t
        if verbose:
            print("testing random variable with distribution {},\n {}\n"\
                  .format(self.distribution,self.sample()))
                  
                  
    def sample(self):
        x = random.random()
        return self.inverse_transform(x)
        
    def sample_repeated(self,n):
        while n>0:
            yield self.sample()
            n -= 1
            
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
