import copy
import numpy as np


class Bandit:
    def __init__(self, p_r, p_tr):
        self.p_r = p_r
        self.num_arms = p_r.shape[0]
        self.p_tr = p_tr

    def generate_data(self, alpha, beta, num_steps):
        rewards = []
        actions = []
        values = []
        v = np.ones(self.num_arms)
        for _ in range(num_steps):
            pi = np.exp(v) / np.sum(np.exp(v), keepdims=True)
            a = np.random.choice(self.num_arms, p=pi)
            r = np.zeros(self.num_arms)
            if np.random.uniform() < self.p_r[a]:
                r[a] += 1

            rewards.append(copy.deepcopy(r))
            actions.append(a)
            values.append(v)

            r *= beta
            v = (1 - alpha) * v + alpha * r

            if np.random.uniform() < self.p_tr:
                np.random.shuffle(self.p_r)
        return np.array(rewards), np.array(actions), np.array(values)


class RewActBandit:
    def __init__(self, p_r, p_tr):
        self.p_r = p_r
        self.num_arms = p_r.shape[0]
        self.p_tr = p_tr

    def generate_data(self, alpha_r, beta_r, alpha_a, beta_a, num_steps):
        rewards = []
        actions = []
        value_rs = []
        value_as = []
        v_r = np.ones(self.num_arms)
        v_a = np.ones(self.num_arms)
        for _ in range(num_steps):
            pi = np.exp(v_r + v_a) / np.sum(np.exp(v_r + v_a), keepdims=True)
            a = np.random.choice(self.num_arms, p=pi)

            u_r = np.zeros(self.num_arms)
            if np.random.uniform() < self.p_r[a]:
                u_r[a] += 1
            u_a = np.zeros(self.num_arms)
            u_a[a] = 1

            rewards.append(copy.deepcopy(u_r))
            actions.append(a)
            value_rs.append(v_r)
            value_as.append(v_a)

            u_r *= beta_r
            u_a *= beta_a
            v_r = (1 - alpha_r) * v_r + alpha_r * u_r
            v_a = (1 - alpha_a) * v_a + alpha_a * u_a

            if np.random.uniform() < self.p_tr:
                np.random.shuffle(self.p_r)
        return np.array(rewards), np.array(actions), np.array(value_rs), np.array(value_as)
