import numpy as np
from keras import Model


def permutation_feature_importance(model: Model, X, y, n_perm=100, return_diff=False):
    def get_loss(x):
        y_pred = model.predict(x, verbose=0)
        return model.compute_loss(None, y, y_pred)
    loss_orig = get_loss(X).numpy()
    loss_perm = np.empty((X.shape[1], n_perm))
    for i in range(X.shape[1]):
        X_i = X.copy()
        for j in range(n_perm):
            np.random.shuffle(X_i[:, i])
            loss_perm[i,j] = get_loss(X_i)
    if return_diff:
        loss_perm - loss_orig
    return loss_perm / loss_orig
