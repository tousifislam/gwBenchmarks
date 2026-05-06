import numpy as np
from sklearn.linear_model import LinearRegression
from gplearn.genetic import SymbolicRegressor

class SVDGPLearn:
    def __init__(self, n_coeffs=5):
        self.n_coeffs = n_coeffs
        self.regressors = [SymbolicRegressor(
            population_size=1000,
            generations=10,
            tournament_size=20,
            function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'neg', 'inv'],
            metric='mse',
            parsimony_coefficient=0.001,
            max_samples=1.0,
            verbose=0,
            random_state=42+i,
        ) for i in range(n_coeffs)]
        self.backup = LinearRegression()
        
    def fit(self, X, y):
        for i in range(self.n_coeffs):
            self.regressors[i].fit(X, y[:, i])
        self.backup.fit(X, y[:, self.n_coeffs:])
        
    def predict(self, X):
        X = np.atleast_2d(X)
        res = []
        for i in range(self.n_coeffs):
            res.append(self.regressors[i].predict(X))
        res_backup = self.backup.predict(X)
        # If res_backup is 1D, make it 2D
        if res_backup.ndim == 1:
            res_backup = res_backup.reshape(X.shape[0], -1)
        # res are also 1D if single sample
        res_2d = [r.reshape(-1, 1) for r in res]
        return np.column_stack(res_2d + [res_backup])
