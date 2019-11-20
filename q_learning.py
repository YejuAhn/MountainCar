#Setting: a car that starts at the bottom of a valley in the mountain
#Goal: reach the flag at the top right by controllng a car
#ends with terminal state or when the max episode length is reached

from environment import MountainCar
import sys
import numpy as np
import matplotlib.pyplot as plt


class Linear_q_network(object):
    #initialize deep q-network
    def __init__(self, state_space, gamma, learn_rate):
        #initialize weight and bias
        self.w = np.zeros((state_space, 3))
        self.bias = 0.0
        self.gamma = gamma
        self.learn_rate = learn_rate

    #return q(s, a; w) for a given state s and action a
    def evaluate(self, s, a, mode): 
        product = 0.0
        for i,v in s.items():
            product += v * self.w[i, a]
        return product + self.bias

    #update parameters
    def update(self, s, a, r, s_next, a_next, mode):
        q_s_a = self.evaluate(s, a, mode) #actions given state s
        q_s_a_next = self.evaluate(s_next, a_next, mode)
        TD_error = self.learn_rate * (q_s_a - (r + self.gamma * q_s_a_next))
        if TD_error != 0.0:
            for i,v in s.items(): #i = row, a = column, v = value of the state
                self.w[i, a] = self.w[i, a] - TD_error * v 
            self.bias = self.bias - TD_error


class Agent(object):
    def __init__(self, car):
        self.car = car

    #epsilon-greedy policy
    def policy(self, s, q_network, exploit, mode):
        if exploit == True:
            actions = [q_network.evaluate(s, a, mode) for a in range(3)] #find action with maximum reward
            # print('list', actions)
            # print('action',actions.index(max(actions)))
            return actions.index(max(actions))#exploit
        else:
            return np.random.choice(3,1)[0] #explore: pick random action! 


    #train agent and the linear Q network and return from all training episodes
    def train(self, q_network, episodes, max_iterations, epsilon, returns_out, mode):
        return_out_file = open(returns_out, 'w')
        total_rewards = []
        total_episodes = []
        total_rolling_means = []
        count = 0
        for episode in range(episodes):
            rewards = 0
            s = self.car.reset() #receive state from the environmnet
            total_episodes.append(episode)
            for t in range(max_iterations):
                if (epsilon == 0):
                    a = self.policy(s, q_network, True, mode)
                    print(a)
                else:
                    if (np.random.uniform(0,1) < epsilon):
                        exploit = False #explore
                    else:
                        exploit = True
                    a = self.policy(s, q_network, exploit, mode) 
                s_next, r, done = self.car.step(a)
                #next best action
                a_next = self.policy(s_next, q_network, True, mode)
                #update
                q_network.update(s, a, r, s_next, a_next, mode)
                rewards += r
                if done == True:
                    print('here')
                    break
                s = s_next
            total_rewards.append(rewards)
            print(total_rewards)
            if episode < 24:
                # rolling_mean = sum(total_rewards) / 25
                total_rolling_means.append(total_rewards[-1])
            else:
                rolling_mean = sum(total_rewards[episode - 24:])/25
                total_rolling_means.append(rolling_mean)
            return_out_file.write(str(rewards) + "\n")
        return (q_network, total_rewards, total_episodes, total_rolling_means)


def plot_analysis(total_rewards, total_episodes, total_rolling_means):
    fig = plt.figure()
    fig.suptitle('Tile Mode', fontsize=16)
    #train
    plt.errorbar(total_episodes, total_rewards, label='Rewards')
    #validation
    plt.errorbar(total_episodes, total_rolling_means, label='Rolling Means')
    plt.legend(loc='upper left')
    plt.show()




def main(args):
    #mode to run environment in: raw / tile
    mode = args[1]
    #path to output the weights of the linear model
    weight_out = args[2]
    #path to output the returns of the agent
    returns_out = args[3]
    #number of episodes to train the agent for 
    episodes = int(args[4])
    #metrics outputs
    max_iterations = int(args[5])
    #epsilon-greedy variant
    epsilon = float(args[6])
    #discount factor
    gamma = float(args[7])
    #learning rate
    learn_rate = float(args[8])

    #instantiate environment 
    car =  MountainCar(mode)
    state_space = car.state_space

    #instantiate qnetwork 
    q_network = Linear_q_network(state_space, gamma, learn_rate)

    #instantiate agent
    agent = Agent(car)

    #train
    return_val = agent.train(q_network, episodes, max_iterations, epsilon, returns_out, mode)
    q_network = return_val[0]

    total_rewards, total_episodes, total_rolling_means = return_val[1], return_val[2], return_val[3]

    # plot_analysis(total_rewards,total_episodes, total_rolling_means)



    #output weight
    weight_out_file = open(weight_out, 'w')
    weight_out_file.write(str(q_network.bias) + "\n")
    # print(q_network.bias)
    # for row in range(len(q_network.w)):
    #     for col in range(len(q_network.w[row,])):
    #         print(q_network.w[row,col])
    #         weight_out_file.write(str(q_network.w[row,col]) + "\n")









if __name__ == "__main__":
    main(sys.argv)
