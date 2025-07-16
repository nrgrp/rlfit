import numpy as np
import cvxpy as cp
import scipy as sp
import numbers
from concurrent.futures import ProcessPoolExecutor


class RLFit:
    def __init__(self, horizon_len=-1, share_param=False):
        if (horizon_len != -1) and (horizon_len <= 0):
            raise ValueError("'horizon_len' must be positive or -1")
        if not isinstance(horizon_len, int):
            raise ValueError("'horizon_len' must be an integer")
        self.horizon_len = horizon_len

        self.share_param = share_param

        # for recovering parameters
        self._num_repeats = None
        self._solver = None

        # results
        self.G_ = None
        self.alpha_ = None
        self.beta_ = None

    def __repr__(self):
        class_name = self.__class__.__name__
        params = [
            f"horizon_len={self.horizon_len}",
        ]
        return f"{class_name}({', '.join(params)})"

    def fit(self, rewards, actions, w=1, reduced_tol_gap_abs=5e-5, reduced_tol_gap_rel=5e-5):
        if isinstance(rewards, np.ndarray):
            n, m = rewards.shape
            k = 1
            rewards = [rewards]
        elif isinstance(rewards, list):
            if not all(isinstance(r, np.ndarray) for r in rewards):
                raise TypeError("all elements in 'rewards' list must be ndarrays")
            shapes = [r.shape for r in rewards]
            if len(set(shapes)) != 1:
                raise ValueError("all arrays in 'rewards' list must have the same shape")
            n, m = shapes[0]
            k = len(shapes)
        else:
            raise TypeError("'rewards' must be a ndarray or a list of ndarrays")

        if not isinstance(actions, np.ndarray):
            raise TypeError("'actions' must be a ndarray")
        if actions.shape != (n, m):
            raise ValueError(f"'actions' shape {actions.shape} does not match 'rewards' shape {(n, m)}")
        if not (np.all(np.logical_or(actions == 0, actions == 1)) and np.all(actions.sum(axis=1) == 1)):
            raise ValueError("each row of 'actions' must be one-hot encoded")

        w = np.ones(k) if w == 1 else w
        if k != w.shape[0]:
            raise ValueError("mismatch between the dimension of 'w' and the number of subrewards")

        if self.horizon_len == -1 or self.horizon_len >= n:
            p = n
        else:
            p = self.horizon_len

        if not self.share_param:
            Gs = [cp.Variable((m, p)) for _ in range(k)]
        else:
            Gs = []
            for _ in range(k):
                g = cp.Variable(p)
                Gs.append(cp.vstack([g for _ in range(m)]))
        X = []
        Y = actions
        for t in range(n):
            Us = []
            for rew in rewards:
                if t < p:
                    Us.append(np.vstack((rew[:t][::-1], np.zeros((p - t, m)))))
                else:
                    Us.append(rew[t - p: t][::-1])
            z = []
            for G, U in zip(Gs, Us):
                z.append(cp.sum(cp.multiply(G, U.T), axis=1))
            X.append(cp.vstack(z).T @ w)
        X = cp.vstack(X)
        obj = cp.sum(cp.sum(cp.multiply(X, Y), axis=1) - cp.log_sum_exp(X, axis=1))
        constraints = []
        for G in Gs:
            constraints.append(G[:, -1] >= 0)
            constraints.append(cp.diff(G, axis=1) <= 0)

        prob = cp.Problem(cp.Maximize(obj), constraints)
        try:
            prob.solve(reduced_tol_gap_abs=reduced_tol_gap_abs, reduced_tol_gap_rel=reduced_tol_gap_rel)
        except cp.SolverError:
            raise RuntimeError(f"solver failed with reduced_tol_gap_abs={reduced_tol_gap_abs}, "
                               f"and reduced_tol_gap_rel={reduced_tol_gap_rel}, "
                               f"consider increasing the precision tolerance")

        # write results
        self.G_ = [G.value for G in Gs]
        self.alpha_ = None
        self.beta_ = None

    def fit_param(self, min_beta=0, max_beta=1e3, num_repeats=5,
                  solver='L-BFGS-B', concurrent=True, num_workers=4):
        if self.G_ is None:
            raise RuntimeError("model not fit, run 'fit()' method on the data first")

        k = len(self.G_)
        m, p = self.G_[0].shape
        Gs = np.vstack(self.G_)

        if isinstance(min_beta, numbers.Number):
            min_beta = [min_beta for _ in range(k)]
        elif isinstance(min_beta, list):
            if len(min_beta) != k:
                raise ValueError("'min_beta' must have the same length as the number of subrewards")
        else:
            raise TypeError("'min_beta' must be a number or a list of numbers")

        if isinstance(max_beta, numbers.Number):
            max_beta = [max_beta for _ in range(k)]
        elif isinstance(max_beta, list):
            if len(max_beta) != k:
                raise ValueError("'max_beta' must have the same length as the number of subrewards")
        else:
            raise TypeError("'max_beta' must be a number or a list of numbers")

        self._num_repeats = num_repeats
        self._solver = solver

        min_betas = np.hstack([np.repeat(minb, m) for minb in min_beta])
        max_betas = np.hstack([np.repeat(maxb, m) for maxb in max_beta])
        if concurrent:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(self._recover_param, Gs, min_betas, max_betas))
        else:
            results = []
            for (g, minb, maxb) in zip(Gs, min_betas, max_betas):
                best_alpha, best_beta = self._recover_param(g, minb, maxb)
                results.append((best_alpha, best_beta))
        htalpha, htbeta = np.array(results).T
        if k == 1:
            self.alpha_ = [htalpha]
            self.beta_ = [htbeta]
        else:
            self.alpha_ = [htalpha[i * m: (i + 1) * m] for i in range(k)]
            self.beta_ = [htbeta[i * m: (i + 1) * m] for i in range(k)]

    def score(self, rewards, actions, w=1):
        if isinstance(rewards, np.ndarray):
            n, m = rewards.shape
            k = 1
            rewards = [rewards]
        elif isinstance(rewards, list):
            if not all(isinstance(r, np.ndarray) for r in rewards):
                raise TypeError("all elements in 'rewards' list must be ndarrays")
            shapes = [r.shape for r in rewards]
            if len(set(shapes)) != 1:
                raise ValueError("all arrays in 'rewards' list must have the same shape")
            n, m = shapes[0]
            k = len(shapes)
        else:
            raise TypeError("'rewards' must be a ndarray or a list of ndarrays")

        if not isinstance(actions, np.ndarray):
            raise TypeError("'actions' must be a ndarray")
        if actions.shape != (n, m):
            raise ValueError(f"'actions' shape {actions.shape} does not match 'rewards' shape {(n, m)}")
        if not (np.all(np.logical_or(actions == 0, actions == 1)) and np.all(actions.sum(axis=1) == 1)):
            raise ValueError("each row of 'actions' must be one-hot encoded")

        w = np.ones(k) if w == 1 else w
        if k != w.shape[0]:
            raise ValueError("mismatch between the dimension of 'w' and the number of subrewards")

        if self.horizon_len == -1 or self.horizon_len >= n:
            p = n
        else:
            p = self.horizon_len

        if self.alpha_ is None:
            Gs = self.G_
        else:
            Gs = []
            for alpha, beta in zip(self.alpha_, self.beta_):
                Gs.append(np.array([alpha * (1 - alpha) ** i * beta for i in range(p)]).T)

        X = []
        Y = actions
        for t in range(n):
            Us = []
            for rew in rewards:
                if t < p:
                    Us.append(np.vstack((rew[:t][::-1], np.zeros((p - t, m)))))
                else:
                    Us.append(rew[t - p: t][::-1])
            z = []
            for G, U in zip(Gs, Us):
                z.append(np.sum(np.multiply(G, U.T), axis=1))
            X.append(np.vstack(z).T @ w)
        X = np.vstack(X)
        return np.sum(np.sum(np.multiply(X, Y), axis=1) - sp.special.logsumexp(X, axis=1))

    def predict(self, rewards, w=1, return_value=False, return_subvalue=False):
        if isinstance(rewards, np.ndarray):
            n, m = rewards.shape
            k = 1
            rewards = [rewards]
        elif isinstance(rewards, list):
            if not all(isinstance(r, np.ndarray) for r in rewards):
                raise TypeError("all elements in 'rewards' list must be ndarrays")
            shapes = [r.shape for r in rewards]
            if len(set(shapes)) != 1:
                raise ValueError("all arrays in 'rewards' list must have the same shape")
            n, m = shapes[0]
            k = len(shapes)
        else:
            raise TypeError("'rewards' must be a ndarray or a list of ndarrays")

        w = np.ones(k) if w == 1 else w
        if k != w.shape[0]:
            raise ValueError("mismatch between the dimension of 'w' and the number of subrewards")

        if self.horizon_len == -1 or self.horizon_len >= n:
            p = n
        else:
            p = self.horizon_len

        if self.alpha_ is None:
            Gs = self.G_
        else:
            Gs = []
            for alpha, beta in zip(self.alpha_, self.beta_):
                Gs.append(np.array([alpha * (1 - alpha) ** i * beta for i in range(p)]).T)

        X = []
        Z = []
        for t in range(n):
            Us = []
            for rew in rewards:
                if t < p:
                    Us.append(np.vstack((rew[:t][::-1], np.zeros((p - t, m)))))
                else:
                    Us.append(rew[t - p: t][::-1])
            z = []
            for G, U in zip(Gs, Us):
                z.append(np.sum(np.multiply(G, U.T), axis=1))
            X.append(np.vstack(z).T @ w)
            Z.append(np.array(z))
        X = np.vstack(X)
        Z = np.array(Z)
        pi = np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)

        if return_subvalue or return_subvalue:
            results = [pi]
            if return_value:
                results.append(X)
            if return_subvalue:
                results.append([z for z in Z.transpose(1, 0, 2)])
            return results
        else:
            return pi

    def _recover_param(self, g, minb, maxb):
        def fn(params):
            alpha, beta = params
            return np.sum([(alpha * (1 - alpha) ** i * beta - g[i]) ** 2 for i in range(g.shape[0])])

        lls = []
        params = []
        for _ in range(self._num_repeats):
            alpha_init = np.random.uniform()
            beta_init = np.random.uniform(minb, maxb)
            bounds = [(0, 1), (minb, maxb)]

            s2_prob = sp.optimize.minimize(fn, (alpha_init, beta_init), bounds=bounds, method=self._solver)

            loss = s2_prob.fun
            lls.append(loss)
            params.append(s2_prob.x)

        best_alpha, best_beta = params[np.argmin(lls)]
        return best_alpha, best_beta
