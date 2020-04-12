import pandas as pd


arms = rewards = contexts = []

with open('dataset.txt', 'r') as f:
    for line in f:
        line_list = line.strip().split()
        print(line_list)
        arms.append(line_list[0])
        rewards.append(line_list[1])

print(arms)
print(rewards)


data = pd.read_csv('dataset.txt', header=None, sep=' ')
print(data.shape)


dict = {'a': (1, 5), "b": (2, 3)}
max(dict, key=lambda x: dict.get(x)[1])
rewards = [1, 2, 9, 3]
rewards.index(max(rewards))


# Implementing UCB
import math
N = 100
d = 10
ads_selected = []
numbers_of_selections = [0] * d
sums_of_reward = [0] * d
total_reward = 0

for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_reward[i] / numbers_of_selections[i]
            delta_i = math.sqrt(2 * math.log(n+1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            print(i)
            max_upper_bound = upper_bound
            ad = i

