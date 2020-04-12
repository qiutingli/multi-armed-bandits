import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

np.random.seed(12)


def random_tie_breaking(values):
    """
    Tie-breaking of the given values

    :param values: list
        values that needs tie_breaking
    :return: int
        random-chosen index of the element with the maximum value
    """
    max_value = np.amax(values)
    candidate_indices = np.where(values == max_value)[0]
    random_max_index = np.random.choice(candidate_indices)
    return random_max_index


class MAB(ABC):
    """
    Abstract class that represents a multi-armed bandit (MAB)
    """

    @abstractmethod
    def play(self, tround, context):
        """
        Play a round

        Arguments
        =========
        tround : int
            positive integer identifying the round

        context : 1D float array, shape (self.ndims * self.narms), optional
            context given to the arms

        Returns
        =======
        arm : int
            the positive integer arm id for this round
        """

    @abstractmethod
    def update(self, arm, reward, context):
        """
        Updates the internal state of the MAB after a play

        Arguments
        =========
        arm : int
            a positive integer arm id in {1, ..., self.narms}

        reward : float
            reward received from arm

        context : 1D float array, shape (self.ndims * self.narms), optional
            context given to arms
        """


class EpsGreedy(MAB):
    """
    Epsilon-Greedy multi-armed bandit

    Arguments
    =========
    narms : int
        number of arms

    epsilon : float
        explore probability

    Q0 : float, optional
        initial value for the arms
    """

    def __init__(self, narms, epsilon, Q0=np.inf):
        self.narms = narms
        self.epsilon = epsilon
        self.Q0 = Q0
        self.history_counts = [0] * narms  # history choosing counts for each arm
        self.avg_rewards = [0.0] * narms  # average rewards for each arm
        self.arm_list = [str(i) for i in range(1,narms+1)] # arm name list with elements are string
        # here is ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

    def play(self, tround, context=None):
        exploit_arm = self.arm_list[random_tie_breaking(self.avg_rewards)]
        random_number = np.random.rand()
        if random_number < self.epsilon:
            explore_arms = [arm for arm in self.arm_list if arm != exploit_arm]
            return np.random.choice(explore_arms)
        else:
            return exploit_arm

    def update(self, arm, reward, context=None):
        arm_index = self.arm_list.index(arm)
        current_reward = self.avg_rewards[arm_index]
        current_count = self.history_counts[arm_index]
        self.history_counts[arm_index] += 1
        self.avg_rewards[arm_index] = (current_reward * current_count + reward) / (current_count + 1)


class UCB(MAB):
    """
    Upper Confidence Bound (UCB) multi-armed bandit

    Arguments
    =========
    narms : int
        number of arms

    rho : float
        positive real explore-exploit parameter

    Q0 : float, optional
        initial value for the arms
    """

    def __init__(self, narms, rho, Q0=np.inf):
        self.narms = narms
        self.rho = rho
        self.Q0 = Q0
        self.history_counts = [0] * narms
        self.avg_rewards = [0.0] * narms
        self.arm_list = [str(i) for i in range(1, narms + 1)]  # arm name list with elements are string

    def play(self, tround, context=None):
        upper_bounds = [self.Q0] * self.narms
        for i in range(self.narms):
            if self.history_counts[i] > 0:
                delta_i = np.sqrt(self.rho * np.log(tround) / self.history_counts[i])
                upper_bound_i = self.avg_rewards[i] + delta_i
                upper_bounds[i] = upper_bound_i
        arm = self.arm_list[random_tie_breaking(upper_bounds)]
        return arm

    def update(self, arm, reward, context=None):
        arm_index = self.arm_list.index(arm)
        current_reward = self.avg_rewards[arm_index]
        current_count = self.history_counts[arm_index]
        self.history_counts[arm_index] += 1
        self.avg_rewards[arm_index] = (current_reward * current_count + reward) / (current_count + 1)


class BetaThompson(MAB):
    """
    Beta-Bernoulli Thompson sampling multi-armed bandit

    Arguments
    =========
    narms : int
        number of arms

    alpha0 : float, optional
        positive real prior hyperparameter

    beta0 : float, optional
        positive real prior hyperparameter
    """

    def __init__(self, narms=10, alpha0=1.0, beta0=1.0):
        self.narms = narms
        self.arm_list = [str(i) for i in range(1, narms + 1)]  # arm name list with elements are string
        self.alpha_beta_dict = self.init_alphas_betas(alpha0,beta0)  # dictionary with keys are arms and values are alphas and betas
        self.history_counts = [0] * narms  # history choosing counts for each arm
        self.avg_rewards = [0.0] * narms  # average rewards for each arm

    def init_alphas_betas(self, alpha0, beta0):
        alpha_beta_dict = {}
        for arm in self.arm_list:
            alpha_beta_dict[arm] = [alpha0, beta0]
        return alpha_beta_dict

    def play(self, tround, context=None):
        theta_dict = {}
        for arm in self.arm_list:
            alpha = self.alpha_beta_dict[arm][0]
            beta = self.alpha_beta_dict[arm][1]
            theta = np.random.beta(alpha, beta)
            theta_dict[arm] = theta
        max_theta = max(theta_dict.values())
        arm = np.random.choice([arm for (arm, theta) in theta_dict.items() if theta == max_theta])
        return arm

    def update(self, arm, reward, context=None):
        arm_index = self.arm_list.index(arm)
        current_reward = self.avg_rewards[arm_index]
        current_count = self.history_counts[arm_index]
        self.history_counts[arm_index] += 1
        self.avg_rewards[arm_index] = (current_reward * current_count + reward) / (current_count + 1)
        if reward == 1:
            self.alpha_beta_dict[arm][0] += 1
        else:
            self.alpha_beta_dict[arm][1] += 1


class LinUCB(MAB):
    """
    Contextual multi-armed bandit (LinUCB)

    Arguments
    =========
    narms : int
        number of arms

    ndims : int
        number of dimensions for each arm's context

    alpha : float
        positive real explore-exploit parameter
    """

    def __init__(self, narms, ndims, alpha):
        self.narms = narms
        self.ndims = ndims
        self.alpha = alpha
        self.arm_list = [str(i) for i in range(1, narms + 1)]  # arm name list with elements are string
        self.init_parameters()
        self.history_counts = [0] * narms

    def init_parameters(self):
        """
        initialize A's and b's for LinUCB

        """
        self.A_dict = {}  # dictionary of A's, where A_dict[arm] is a n by n matrix for some arm .
        self.b_dict = {}  # dictionary of b's, where b_dict[arm] is a n by 1 vector for some arm .
        for arm in self.arm_list:
            self.A_dict[arm] = np.identity(self.ndims)
            self.b_dict[arm] = np.zeros((self.ndims,1))

    def get_scores(self, context):
        """
        Get scores for arms using LinUCB algorithm

       :param context: 1D float array, shape (self.ndims * self.narms), optional
           context given to arms
       :return: list
           scores for arms
       """
        scores = []
        for arm in self.arm_list:
            arm_context = context[(int(arm)-1)*10:int(arm)*10]
            context_array = np.array([[float(feature)] for feature in arm_context])
            A_inv = inv(self.A_dict[arm])
            theta = A_inv.dot(self.b_dict[arm])
            estimated_reward = theta.T.dot(context_array)
            uncertainty = self.alpha * np.sqrt(context_array.T
                                               .dot(A_inv)
                                               .dot(context_array))
            score = estimated_reward + uncertainty
            scores.append(score)
        return scores

    def play(self, tround, context):
        scores = self.get_scores(context)
        arm = self.arm_list[random_tie_breaking(scores)]
        return arm

    def update(self, arm, reward, context):
        arm_context = context[(int(arm) - 1) * 10:int(arm) * 10]
        context_array = np.array([[float(feature)] for feature in arm_context])
        self.A_dict[arm] += context_array.dot(context_array.T)
        self.b_dict[arm] += reward*context_array
        arm_index = self.arm_list.index(arm)
        self.history_counts[arm_index] += 1


class LinThompson(MAB):
    """
    Contextual Thompson sampled multi-armed bandit (LinThompson)

    Arguments
    =========
    narms : int
        number of arms

    ndims : int
        number of dimensions for each arm's context

    v : float
        positive real explore-exploit parameter
    """

    def __init__(self, narms, ndims, v):
        self.narms = narms
        self.ndims = ndims
        self.v = v
        self.arm_list = [str(i) for i in range(1, narms + 1)]  # arm name list with elements are string
        self.init_parameters()
        self.history_counts = [0] * narms

    def init_parameters(self):
        """
        initialize B's, mu hats and f's for LinThompson

        """
        self.B_dict = {}  # dictionary of B's, where A_dict[arm] is a n by n matrix for some arm .
        self.mu_hat_dict = {}  # dictionary of mu hats, where mu_hat_dict[arm] is a n by 1 vector for some arm .
        self.f_dict = {}   # dictionary of f's, where f_dict[arm] is a n by 1 vector for some arm .
        for arm in self.arm_list:
            self.B_dict[arm] = np.identity(self.ndims)
            self.mu_hat_dict[arm] = np.zeros(self.ndims)
            self.f_dict[arm] = np.zeros(self.ndims)

    def get_scores(self, context):
        """
        Get scores for arms using LinThompson algorithm

        :param context: 1D float array, shape (self.ndims * self.narms), optional
            context given to arms
        :return: list
            scores for arms
        """
        scores = []
        for arm in self.arm_list:
            arm_context = context[(int(arm) - 1) * 10:int(arm) * 10]
            context_array = np.array([float(feature) for feature in arm_context])
            mu_hat = self.mu_hat_dict[arm]
            mu_tilde = np.random.multivariate_normal(mu_hat.flat,
                                                     self.v ** 2 * inv(self.B_dict[arm]))[..., np.newaxis]
            score = context_array.T.dot(mu_tilde)[0]
            scores.append(score)
        return scores

    def play(self, tround, context):
        scores = self.get_scores(context)
        arm = self.arm_list[random_tie_breaking(scores)]
        return arm

    def update(self, arm, reward, context):
        arm_context = context[(int(arm) - 1) * 10:int(arm) * 10]
        context_array = np.array([float(feature) for feature in arm_context])
        self.B_dict[arm] += context_array.dot(context_array.T)
        self.f_dict[arm] += context_array.dot(reward)
        self.mu_hat_dict[arm] += inv(self.B_dict[arm]).dot(self.f_dict[arm])
        arm_index = self.arm_list.index(arm)
        self.history_counts[arm_index] += 1


def offlineEvaluate(mab, arms, rewards, contexts, nrounds=None):
    """
    Offline evaluation of a multi-armed bandit

    Arguments
    =========
    mab : instance of MAB

    arms : 1D int array, shape (nevents,)
        integer arm id for each event

    rewards : 1D float array, shape (nevents,)
        reward received for each event

    contexts : 2D float array, shape (nevents, mab.narms*nfeatures)
        contexts presented to the arms (stacked horizontally)
        for each event.

    nrounds : int, optional
        number of matching events to evaluate `mab` on.

    Returns
    =======
    out : 1D float array
        rewards for the matching events
    """
    # history = []
    matching_event_rewards = []
    num_events = len(arms)
    tround = 0
    # mean_rewards = []
    for t in range(num_events):
        if tround < nrounds:
            arm = arms[t]
            r = rewards[t]
            context = contexts[t]
            arm_played = mab.play(tround, context)
            if arm_played == arm:
                mab.update(arm_played, r, context)
                matching_event_rewards.append(r)
                # history.append((t, arm_played, r))
                # mean_rewards.append(np.mean(matching_event_rewards))
                tround += 1
        else:
            break
    # plt.plot(range(1, len(np.array(mean_rewards))+1), np.array(mean_rewards), label=mab.__class__.__name__)
    return matching_event_rewards


def plot_mab_results(results_dict):
    """
    Plot the results of mab's

    :param results_dict: dictionary.  key: string  value: list
        dictionary with keys are mab names and values are reward histories respectively
    """
    plt.figure(figsize=(12, 9))
    for (mab_name, matching_event_rewards) in results_dict.items():
        stepwise_mean_rewards = []
        for i in range(len(matching_event_rewards)):
            stepwise_mean_reward = np.mean(matching_event_rewards[0:i+1])
            stepwise_mean_rewards.append(stepwise_mean_reward)
        plt.plot(range(1, len(np.array(stepwise_mean_rewards)) + 1),
                 np.array(stepwise_mean_rewards),
                 label=mab_name)
    plt.legend(loc='upper left')
    plt.title("MAB Average Reward vs. Round Played")
    plt.xlabel("Round Played")
    plt.ylabel("Average Reward")
    plt.show()


def optimize_linucb(arms, rewards, contexts):
    """
    Find the optimal alpha for LinUCB and plot the result

    """
    results = []
    alphas = np.arange(0, 1, 0.1)
    for alpha in alphas:
        mab = LinUCB(10, 10, alpha)
        result = np.mean(offlineEvaluate(mab, arms, rewards, contexts, 800))
        results.append(result)
    print('Mean reward for LinUCB achieves maximum value {} at alpha equals to {}'
          .format(max(results), alphas[np.argmax(results)]))
    plt.plot(alphas, results)
    plt.title("LinUCB Mean Reward vs. Alpha")
    plt.xlabel("Alpha")
    plt.ylabel("Mean Reward")
    plt.show()


def optimize_linthompson():
    """
    Find the optimal v for LinThompson and plot the result

    """
    results = []
    vs = np.arange(0, 1, 0.1)
    for v in vs:
        mab = LinThompson(10, 10, v)
        result = np.mean(offlineEvaluate(mab, arms, rewards, contexts, 800))
        results.append(result)
    print('Mean reward for LinThompson achieves maximum value {} at alpha equals to {}'
          .format(max(results), np.argmax(results)))
    plt.plot(vs, results)
    plt.title("LinThompson Mean Reward vs. v")
    plt.xlabel("v")
    plt.ylabel("Mean Reward")
    plt.show()


if __name__ == '__main__':

    arms = []
    rewards = []
    contexts = []

    with open('dataset.txt', 'r') as f:
        for line in f:
            line_list = line.strip().split()
            arms.append(line_list[0])
            rewards.append(int(line_list[1]))
            contexts.append(line_list[2:103])

    mab = EpsGreedy(10, 0.05)
    results_EpsGreedy = offlineEvaluate(mab, arms, rewards, contexts, 800)
    print('EpsGreedy average reward', np.mean(results_EpsGreedy))

    mab = UCB(10, 1.0)
    results_UCB = offlineEvaluate(mab, arms, rewards, contexts, 800)
    print('UCB average reward', np.mean(results_UCB))

    mab = BetaThompson(10, 1.0, 1.0)
    results_BetaThompson = offlineEvaluate(mab, arms, rewards, contexts, 800)
    print('BetaThompson average reward', np.mean(results_BetaThompson))

    mab = LinUCB(10, 10, 1.0)
    results_LinUCB = offlineEvaluate(mab, arms, rewards, contexts, 800)
    print('LinUCB average reward', np.mean(results_LinUCB))

    mab = LinThompson(10, 10, 1.0)
    results_LinThompson = offlineEvaluate(mab, arms, rewards, contexts, 800)
    print('LinThompson average reward ', np.mean(results_LinThompson))

    results_dict = {
        'EpsGreedy': results_EpsGreedy,
        'UCB': results_UCB,
        'BetaThompson': results_BetaThompson,
        'LinUCB': results_LinUCB,
        'LinThompson': results_LinThompson
    }

    plot_mab_results(results_dict)

    # optimize_linucb(arms, rewards, contexts)
    # optimize_linthompson()

