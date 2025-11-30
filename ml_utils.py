import numpy as np
from tqdm import tqdm
from keras import Model
from keras.models import clone_model
from keras.layers import Input
from keras.callbacks import EarlyStopping


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

def leave_one_feature_out_importance(
        model: Model,
        X_train, X_test, y_train, y_test,
        return_diff=False,
):
    def get_loss(model, x):
        y_pred = model.predict(x, verbose=0)
        return model.compute_loss(None, y_test, y_pred)
    loss_orig = get_loss(model, X_test).numpy()
    loss_perm = np.empty(X_train.shape[1])
    for i in tqdm(range(X_train.shape[1])):
        X_i = np.delete(X_train, i, axis=1)
        model_i = clone_model(model, Input(shape=X_i.shape[1:]))
        model_i.compile()
        history_i = model_i.fit(
            X_i, y_train,
            epochs=200, validation_split=0.2, verbose=0,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
        )
        loss_perm[i] = get_loss(model_i, np.delete(X_test, i, axis=1))
    if return_diff:
        loss_perm - loss_orig
    return loss_perm / loss_orig
