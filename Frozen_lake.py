# These packages/libraries need to be imported if not already installed.
import numpy as np
import gym
from gym import wrappers

# Various Parameters like alpha, gamma, count and so on.
alpha = 0.6 # Harcoded alpha to 0.6 i.e the learning rate
Gamma = 0.3 # Harcoded gamma to 0.3 i.e the discount
count = 80000 # count of episodes

# This is the value of epsilon
epsilon = 0.4 # Epsilon value is hardcoded to 0.4.

#this is the max limit
N = 110

class LakeAgent:
    # Initliaziation part.
    # Initialize env, episode and q_val.
    def __init__(self, env):
        self.env = env
        self.episode_reward = 0.0
        self.q_val = np.zeros(16 * 4).reshape(16, 4).astype(np.float32)

    # Function where learning happens.
    def learning(self):
        # one episode learning
        state = self.env.reset()
        
        # When in the range, do exploration.
        for t in range(N):
            # make sure value is less than epsilon
            if np.random.rand() < epsilon: 
                act = self.env.action_space.sample() # Do the sampling to explore.
            else: 
                act = np.argmax(self.q_val[state])
            # Calculate the next_state, reward and info
            next_state, reward, done, info = self.env.step(act)
            q_next_max = np.max(self.q_val[next_state])
            self.q_val[state][act] = (1 - alpha) * self.q_val[state][act] + alpha * (reward + Gamma * q_next_max)
            
            # If done return reward else move to next_state
            if not done:
                state = next_state
            else:
                return reward
        return 0.0 # Invalid and above the limit, so return 0 value.

# Main function definition
def main():
    # Environment imported from openAIgym package
    env = gym.make("FrozenLake-v0")
    agent = LakeAgent(env)

    print("In learning state. This might take a few seconds ....")

    reward_total = 0.0 # Initilize a reward variable
    for i in range(count):
        reward_total += agent.learning() # Calculate the total reward from learning.
        
    print("Number of episodes = ",count) # Count of number of episodes taken.
    print("Sum of reward gained = ",reward_total) # Total reward that was attained.
    
    # Verify by printing out the Q-values.
    #print("The corresponding q-values=",agent.q_val)

# Invoke the main function.
if __name__ == "__main__":
    main()
