import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse

class MountainCar:
    def __init__(self, env, algo, alpha, gamma, epsilon):
        self.env = env
        self.algo = algo
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.n_states = np.round((self.env.observation_space.high - self.env.observation_space.low)*np.array([20, 200]),0).astype(int) + 1
        self.Q = np.zeros((self.n_states[0], self.n_states[1], self.env.action_space.n))
        self.ep_reward = []
        self.reward_list = []
        self.episodes = 0
    
    def _discretize_state(self, state):
        return np.round((state - self.env.observation_space.low)*np.array([20, 200]),0).astype(int)
    
    def _epsilon_greed(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.env.action_space.n-1)
        else:
            return np.argmax(self.Q[state[0], state[1]])
    
    def _get_expected_reward(self, state):
        actions = [i for i in range(self.env.action_space.n)]
        # Find greedy action, (1-ep)*greedy reward
        greedy_action = np.argmax(self.Q[state[0], state[1]])
        greedy_reward = (1-self.epsilon) * self.Q[state[0], state[1], greedy_action]
        # other actions, ep/n_actions * reward
        actions.remove(greedy_action)
        other_rewards = 0
        for r in actions:
            other_rewards += (self.epsilon/len(actions)) * self.Q[state[0], state[1], r]
        return greedy_reward + other_rewards
    
    def calc_avg_reward(self):
        avg_ep_reward = np.mean(self.ep_reward)
        self.reward_list.append(avg_ep_reward)
        self.ep_reward = []
        return(avg_ep_reward)
    
    def run_episode(self, decay, render = False):
        if self.episodes % 500 == 0 or render:
            print(f'Running episode {self.episodes} using {self.algo}...')
        end = False
        total_reward, reward = 0,0
        # Initial State
        S = self.env.reset()
        # Discreize State
        S_dis = self._discretize_state(S)
        while not end:
            if render:
                env.render()
                
            action = self._epsilon_greed(S_dis)
            S_next, reward, end, _ = self.env.step(action)
            S_dis_next = self._discretize_state(S_next)
            
            # If end of episode
            if end and S_next[0] >= 0.5:
                self.Q[S_dis[0], S_dis[1], action] = reward
                
            # otherwise update according to chosen algorithm
            else:
                if self.algo == 'q':
                    next_reward = np.max(self.Q[S_dis_next[0], S_dis_next[1]])
                elif self.algo == 'expected_sarsa':
                    next_reward = self._get_expected_reward(S_dis_next)
                elif self.algo == 'sarsa':
                    next_reward = self.Q[S_dis_next[0], S_dis_next[1], self._epsilon_greed(S_dis_next)]
                    
                delta = self.alpha * (reward + (self.gamma * next_reward) - self.Q[S_dis[0], S_dis[1], action])
                self.Q[S_dis[0], S_dis[1], action] += delta
                
            S_dis = S_dis_next
            total_reward += reward
            
        if render:
            if (S_next[0] >= 0.5):
                print('Success :)')
            else: 
                print('Failure :(')
            
        self.ep_reward.append(total_reward)
        self.episodes += 1
        self.epsilon -= decay
        if self.episodes % 100 == 0:
            avg_reward = self.calc_avg_reward()
            if self.episodes % 500 == 0:
                print(f'Avg Reward Over Last 100 Episodes = {avg_reward}...')
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", dest='algo', choices=['sarsa', 'expected_sarsa', 'q'])
    args = parser.parse_args()
    
    env = gym.make('MountainCar-v0')
    env.reset()
    episodes = 5000
    alpha = .1
    gamma = .9
    epsilon = .75
    decay = epsilon/episodes
    car = MountainCar(env, args.algo, alpha, gamma, epsilon)
                      
    for i in range(episodes):
        if i >= episodes - 10:
            car.run_episode(decay, render = True)
        else:
            car.run_episode(decay)
            
    env.close()
    
    plt.plot(100*(np.arange(len(car.reward_list)) + 1), car.reward_list)
    plt.xlabel('Episodes')
    plt.ylabel('Smoothed Reward')
    plt.title('Smoothed Reward vs Episodes')
    plt.show()