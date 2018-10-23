import numpy as np
import math
from functools import reduce
import matplotlib.pyplot as plt
import random
from numpy.random import multivariate_normal
from pandas import cut
%matplotlib inline

def Question1and3(sim, n):
    sum1,sq_sum1,mini,maxi = reduce(lambda x,y : (x[0]+y,x[1]+y**2,min(x[2],y),max(x[3],y)),\
                                        sim.sample_repeated(n),(0,0,999999,-999999))
    print("The mean is {}".format(sum1/n))
    print("The std_dev is {}".format(math.sqrt(sq_sum1/n - (sum1/n)**2)))
    print("The min is {}".format(mini))
    print("The max is {}".format(maxi))
def Question4(sim,boolean_func, n):
    sum1 = reduce(lambda x,y :x + int(y), sim.sample_boolean_repeated(boolean_func,n),0)
    print("The probability is {}".format(sum1/n))

nextpos = 0
reward_array = None
deviation_array = None
episodes_array = None
standard_dev = 0

vectors = {}
for i in range(10):
    v = np.zeros(10)
    v[i] = 1
    vectors[i] = v



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
        self.current_reward += self.reward_function(new_state,action)*self.damping_constant**(self.delta_t)
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
        action = self.policy(self.current_state,actions)
        new_state = self.transition_function(self.current_state,action)
        self.current_reward += self.reward_function(new_state,action)*self.damping_constant**(self.timestep)
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

def policy_paramaterized(param,sigma,state,actions):
    vel,pos,ang_vel,theta = state
    class_theta = cut([theta],np.linspace(-math.pi/2,math.pi/2,11),labels = list(range(10)))[0]
    theta_vec = vectors[class_theta]
    class_pos = cut([theta],np.linspace(-3,3,11),labels = list(range(10)))[0]
    pos_vec = vectors[class_pos]    
    I = np.dot(param[:10],theta_vec)+np.dot(param[30:40],theta_vec)*ang_vel+np.dot(param[10:20],pos_vec)+np.dot(param[20:30],pos_vec)*vel
    pred = 1/(1+math.e**(-I)) # logistic regression
    return np.random.choice(actions, p=[1-pred,pred])


#.05,no_itterations,1,run_mdp
def CrossEntropyPolicySearch(no_dim,mean_policy,covariance,no_policies,top_e,\
                             no_episodes,no_itterations,epsilon,sigma,MDP):
    assert ( no_policies >= top_e)

    global reward_array
    reward_array = np.zeros(no_itterations)
    global deviation_array
    deviation_array = np.zeros(no_itterations)
    global episodes_array
    episodes_array = np.zeros(no_itterations)

    #reward_array = np.zeros((no_episodes*no_policies*no_itterations),dtype=np.float64)
    global nextpos
    nextpos = 0
    avgRew = 0
    i = 0
    while i < no_itterations:
        i+=1
        J = []
        k = 0
        params = multivariate_normal(mean_policy,covariance,[no_policies])        
        global standard_dev
        standard_dev = 0
        for p in params:
            MDP.policy = lambda st,acts: policy_paramaterized(p,sigma,st,acts)
            J.append((p,sum(MDP.sample_repeated(no_episodes))/no_episodes))
        J.sort(key = lambda x : x[1], reverse = True)
        top_policy,rewards = zip(*J)
        avgRew = sum(rewards)/len(rewards)
        standard_dev = standard_dev/(len(rewards)*no_episodes) - avgRew
        top_policy = top_policy[0:top_e]
        mean_policy = np.average(top_policy,axis = 0)
        centered_policy = map(lambda x : x - mean_policy,top_policy)
        covar_policy = map(lambda x : np.outer(x,x),centered_policy)
        covariance = (epsilon*np.eye(no_dim)+sum(covar_policy))/(epsilon+top_e)
        #print("itteration#{} with average reward : {}".format(i,avgRew))
        #print('')
        reward_array[nextpos] = avgRew
        deviation_array[nextpos] = math.sqrt(standard_dev)
            
        episodes_array[nextpos] = no_episodes*i*no_policies 
        nextpos += 1
        
def HillClimbingPolicySearch(no_dim,mean_policy,covariance,\
                             no_episodes,no_itterations,sigma,MDP):
    global reward_array
    reward_array = []
    global deviation_array
    deviation_array = []
    global episodes_array
    episodes_array = []

    global standard_dev
    standard_dev = 0    
    avgRew = 0
    mean_J = 0
    prev_iter = 0
    MDP.policy = lambda st,acts: policy_paramaterized(mean_policy,sigma,st,acts)    
    J = sum(MDP.sample_repeated(no_episodes))/no_episodes
    i = 0
    while i < no_itterations:
        i+=1
        param = multivariate_normal(mean_policy,covariance)
        MDP.policy = lambda st,acts: policy_paramaterized(param,sigma,st,acts)
        new_J = sum(MDP.sample_repeated(no_episodes))/no_episodes
        mean_J +=new_J
        if new_J > J:
            print(i)
            J = new_J
            mean_policy = param
            standard_dev = (standard_dev/no_episodes - mean_J)/(i-prev_iter)
            mean_J  = 0
            prev_iter = i
            reward_array.append(new_J)
            deviation_array.append(math.sqrt(standard_dev))
            episodes_array.append(no_episodes*i)
            standard_dev = 0

        #print("itteration#{} with average reward : {}".format(i,avgRew))
        #print('')
    
        #print("itteration#{} with average reward : {}".format(i,avgRew))
        #print('')
actions = ("L","R")
def policy_random(state,actions):
    return random.choice(actions)

def termination_func(state,time):
    vel,pos,ang_vel,theta = state    
    max_angle = math.pi/2
    min_angle = -math.pi/2
    max_t = 20.2/.02
    max_x = 3
    min_x = -3
    return (pos > max_x or pos<min_x or theta<min_angle or theta>max_angle or time >= max_t) 
    
def transition_func(state,action):
    vel,pos,ang_vel,theta = state
    delta_t = .02
    m_p = .1
    m_c = 1
    l = .5
    motor_force = 10 #in newtons
    g = 9.8

    
    pos += vel*delta_t/2
    theta += ang_vel*delta_t/2
    if action == 'L':
        F = -motor_force
    else:
        F = motor_force
    ang_acc = (g*math.sin(theta)+math.cos(theta)*\
                (-F-m_p*l*math.sin(sigma)*ang_vel**2))/\
                (l*(4/3-(m_p*math.cos(theta)**2)/(m_c+m_p)))
    acc = (F+m_p*l*(math.sin(theta)*ang_vel**2-ang_acc*math.cos(theta)))/(m_c+m_p)
    vel += acc*delta_t
    ang_vel += ang_acc*delta_t
    pos += vel*delta_t/2
    theta += ang_vel*delta_t/2
    return (vel,pos,ang_vel,theta)
# Answer to Question 1

myMDP = infinite_state_MDP(actions,transition_func,lambda st,act: 1,termination_func,lambda : (0,0,0,0), 1, None)
no_samples= 30 #N
no_dims = 40
no_policies = 20 #K
top_examples = 2
no_itterations = 100 #no_itterations
sigma = .25
epsilon = .05
omega = .5
CrossEntropyPolicySearch(no_dims,np.random.random_sample((no_dims,)),np.eye(no_dims),\
                         no_policies,top_examples,no_samples,no_itterations,epsilon,sigma,myMDP)
#plt.plot(range(no_itterations),reward_array)
plt.errorbar(episodes_array,reward_array,deviation_array,label = 'Cross Entropy',errorevery =25)
                         
HillClimbingPolicySearch(no_dims,np.random.random_sample((no_dims,)),np.eye(no_dims)*omega,\
                             no_samples,no_policies*no_itterations,sigma,myMDP)
plt.errorbar(episodes_array,reward_array,deviation_array,label = "Hill Climbing",errorevery =25)
plt.legend()
plt.title("CartPole", fontsize=16, fontweight='bold')
plt.ylabel("expected return")
plt.xlabel("number of episodes")
plt.show()
plt.savefig('results/CartPole_fig.png')
