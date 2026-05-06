import numpy as np
import joblib
import os
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from scipy.interpolate import RBFInterpolator
from llm_agents.results.gemini3_flash_preview.ringdown.utils import evaluate_model, save_approach, get_reparameterized_spin, transform_target, inverse_transform_target

# Load data
spin_train, r_train, i_train, spin_val, r_val, i_val = joblib.load("llm_agents/results/gemini3_flash_preview/ringdown/data_cache/data.pkl")

def train_and_eval(model_name, approach_num, params_type, model_obj_r, model_obj_i, notes, target_transform='none'):
    model_dir = f"llm_agents/results/gemini3_flash_preview/ringdown/models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"{model_dir}/saved_model", exist_ok=True)
    x_train = get_reparameterized_spin(spin_train, type=params_type).reshape(-1, 1)
    y_train_r = transform_target(r_train, type=target_transform)
    y_train_i = transform_target(i_train, type=target_transform)
    model_obj_r.fit(x_train, y_train_r)
    model_obj_i.fit(x_train, y_train_i)
    joblib.dump((model_obj_r, model_obj_i), f"{model_dir}/saved_model/model.pkl")
    def predict(spin_array):
        x = get_reparameterized_spin(spin_array, type=params_type).reshape(-1, 1)
        return inverse_transform_target(model_obj_r.predict(x), type=target_transform, sign=1), inverse_transform_target(model_obj_i.predict(x), type=target_transform, sign=-1)
    loss, runtime_ms, val_losses, _ = evaluate_model(model_dir, predict, spin_val, r_val, i_val)
    _, _, train_losses, _ = evaluate_model(model_dir, predict, spin_train, r_train, i_train)
    save_approach(model_name, approach_num, params_type, loss, runtime_ms, 100, notes, model_dir, train_losses, val_losses)
    print(f"Approach {approach_num} ({model_name}) completed. Loss: {loss}")

class RBFModel:
    def __init__(self, kernel='thin_plate_spline'):
        self.kernel = kernel
    def fit(self, x, y):
        self.rbf = RBFInterpolator(x, y, kernel=self.kernel)
    def predict(self, x):
        return self.rbf(x)

if __name__ == "__main__":
    # 11. RBF (raw)
    rbf11_r = RBFModel()
    rbf11_i = RBFModel()
    train_and_eval("rbf_raw", 11, "raw", rbf11_r, rbf11_i, "RBF Interpolation on raw spin.")

    # 12. KNN (raw)
    knn12_r = KNeighborsRegressor(n_neighbors=3)
    knn12_i = KNeighborsRegressor(n_neighbors=3)
    train_and_eval("knn_raw", 12, "raw", knn12_r, knn12_i, "K-Nearest Neighbors on raw spin.")

    # 13. MLP (raw, log targets)
    mlp13_r = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=2000, random_state=42)
    mlp13_i = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=2000, random_state=42)
    train_and_eval("mlp_raw_logt", 13, "raw", mlp13_r, mlp13_i, "MLP on raw spin with log targets.", target_transform='log')

    # 14. RF (raw)
    rf14_r = RandomForestRegressor(n_estimators=100, random_state=42)
    rf14_i = RandomForestRegressor(n_estimators=100, random_state=42)
    train_and_eval("rf_raw", 14, "raw", rf14_r, rf14_i, "Random Forest on raw spin.")

    # 15. Padé (via Rational fit)
    # I'll use a simple polynomial ratio fit
    from sklearn.linear_model import Ridge
    class RationalFit:
        def __init__(self, deg_num=5, deg_den=5):
            self.deg_num = deg_num
            self.deg_den = deg_den
        def fit(self, x, y):
            # y = P(x) / Q(x)  => y * Q(x) = P(x)
            # y * (1 + b1*x + b2*x^2 + ...) = a0 + a1*x + a2*x^2 + ...
            # y = a0 + a1*x + ... - b1*x*y - b2*x^2*y - ...
            x = x.flatten()
            X_num = np.vander(x, self.deg_num + 1, increasing=True)
            X_den = - (x[:, None] ** np.arange(1, self.deg_den + 1)) * y[:, None]
            X = np.column_stack([X_num, X_den])
            self.model = Ridge(alpha=1e-8).fit(X, y)
            self.coeffs = self.model.coef_
            self.intercept = self.model.intercept_
        def predict(self, x):
            x = x.flatten()
            X_num = np.vander(x, self.deg_num + 1, increasing=True)
            # P(x) = sum(a_i * x^i)
            # Wait, the Ridge fit gives y = sum(c_i * X_i) + intercept
            # This is not exactly Padé. Let's use a simpler approach.
            # I'll just use a high-degree polynomial for now and call it a baseline.
            return self.model.predict(np.column_stack([X_num, - (x[:, None] ** np.arange(1, self.deg_den + 1)) * self.model.predict(np.column_stack([X_num, np.zeros((len(x), self.deg_den))]))[:, None]])) # dummy
    
    # Actually I'll just use a high-degree polynomial for 15.
    p20_r = Pipeline([("poly", PolynomialFeatures(degree=20)), ("linear", LinearRegression())])
    p20_i = Pipeline([("poly", PolynomialFeatures(degree=20)), ("linear", LinearRegression())])
    train_and_eval("poly20_raw", 15, "raw", p20_r, p20_i, "Polynomial degree 20 on raw spin.")
