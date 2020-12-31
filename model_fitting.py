import numpy as np
from sklearn.linear_model import LogisticRegression

def compute_l1(vec, lamb):
    return np.abs(vec).sum() * lamb

def compute_logistic_loss(X, y, beta, lamb):
    l1_loss = compute_l1(vec=beta, lamb=lamb)
    x_beta = X.dot(beta)
    nll = - (y * x_beta - np.log(1.0 + np.exp(x_beta))).mean()
    return nll + l1_loss

def compute_quadratic_loss(X, z, w, beta, lamb):
    l1_loss = compute_l1(vec=beta, lamb=lamb)
    x_beta = X.dot(beta)
    wss = 0.5 * (w * ( (x_beta - z) ** 2) ).mean()
    return wss + l1_loss

def get_w_z(X, y, beta):
    x_beta = X.dot(beta)
    preds = 1 / (1 + np.exp(-x_beta))
    w = preds * (1.0 - preds)
    w = np.clip(w, a_min=1e-5, a_max=1.0)
    z = x_beta + (y - preds) / w 
    return w, z

def get_beta(X, y, lamb, warm_beta=None, verbose=0):
    scaled_c = 1 / (y.shape[0] * lamb)
    if warm_beta is None:
        model = LogisticRegression(warm_start=False, penalty='l1', C=scaled_c, fit_intercept=False, solver='liblinear', verbose=verbose)
    else:
        if verbose:
            print("Operating with warm_start = True")
        model = LogisticRegression(warm_start=True, penalty='l1', C=scaled_c, fit_intercept=False, solver='liblinear', verbose=verbose)
        model.coef_ = warm_beta
        print(model.coef_)
    model.fit(X=X, y=y)
    return model.coef_.flatten()