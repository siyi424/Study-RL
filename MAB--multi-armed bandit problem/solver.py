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


# MAB's basic framework
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
    # exploration has much more uncertainties, while exploitation always exploit the max_prob arm
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


# strategies of exploration & exploitation

# epsilon-Greedy algorithm
# set: epsolon = 0.01, T = 5000
class EpsilonGreedy(Solver):   
    # why needing to be inherited from Solver? 
    # solver is a parent class, defining the basis of strategies. It values!
    def __init__(self, bandit, epsilon = 0.01, init_prob = 1.0):
        super().__init__(bandit) 
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.K)


    def run_one_step(self):
        if np.random.random() < self.epsilon: # exploration
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k) # the reward of this step
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k]) # update the expected_Q
        return k
    



# plot the relation curves that the accumulated regrets varies with time 
# solvers is a list that each items is a strategy
def plot_results(solvers, solver_name):
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label = solver_name[idx])
    plt.xlabel('time steps')
    plt.ylabel('cumulative regrets')
    plt.title('%d-armed bndit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()

        

# init Bandit
K = 10
bandit_10_arms = BernouliBandit(K)
print("Randomly generate a %d armed bandit" %K)
print('the max expected_Q is: %.4f, its number is: %d' %(bandit_10_arms.best_prob, bandit_10_arms.best_idx))


epsilon_greedy_solver = EpsilonGreedy(bandit_10_arms, epsilon=0.01)
epsilon_greedy_solver.run(5000)
print('the cumulative regrets is: ', epsilon_greedy_solver.regret)
plot_results([epsilon_greedy_solver], ['epsilon-greedy'])
