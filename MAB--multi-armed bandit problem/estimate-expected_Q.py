import numpy as np
import matplotlib.pyplot as plt


# to calculate each bandit's expected_Q from taking large amount of samples
# define general bernouliBandit
class BernouliBandit():
    '''K means how many bandits here'''
    def __init__(self,k):
        rng = np.random.default_rng(seed=1) # set seed for repeating feasibility
        self.probs = rng.uniform(0, 1, size=k) # uniformly draw samples between [0,1) 
        self.best_idx = np.argmax(self.probs) # return the largest prob's idx
        self.best_prob = self.probs[self.best_idx] # the largest prob
        self.K = k 
    
    # the k-th arm running
    def step(self,k):
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0



K = 10
bandit_10_arms = BernouliBandit(K)
print("Randomly generate a %d armed bandit" %K)
print('the max expected_Q is: %.4f, its number is: %d' %(bandit_10_arms.best_prob, bandit_10_arms.best_idx))
