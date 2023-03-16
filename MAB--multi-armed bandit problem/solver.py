# MAB's basic framework

import numpy as np
from estimate_expected_Q import BernouliBandit

class Solver:
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K) # record each arm's operating times
        self.regret = 0. # current step's regret
        self.actions = [] # record every step's action
        self.regrets = [] # record every step's accumulated_regret


    # calculate the accumulated regrets
    def update_regret(self, k):
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)
    

    # game of exploration & exploitation
    # return k-th arm
    def run_one_step(self):
        raise NotImplementedError


    # run num_steps times
    def run(self,num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.update_regret(k)
            self.actions.append(k)

            

# init Bandit
K = 10
bandit_10_arms = BernouliBandit(K)
print("Randomly generate a %d armed bandit" %K)
print('the max expected_Q is: %.4f, its number is: %d' %(bandit_10_arms.best_prob, bandit_10_arms.best_idx))


# init Solver
solver = Solver(bandit_10_arms)