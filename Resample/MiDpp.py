import numpy as np
from sklearn.metrics import pairwise_distances


def support_points_undersample(
        X_train, Y_train,
        target_class=1,
        target_num_samples=1852,
        sp_iters=10,
        sp_batch_size=1000,
        sp_lr=0.1,
        random_state=44
):

    X = np.asarray(X_train)
    Y = np.asarray(Y_train)
    rng = np.random.RandomState(random_state)

    maj_idx = np.where(Y == target_class)[0]
    min_idx = np.where(Y != target_class)[0]

    if len(maj_idx) <= target_num_samples:
        combined_idx = np.concatenate([min_idx, maj_idx])
        return X[combined_idx], Y[combined_idx], combined_idx

    X_maj = X[maj_idx]
    m = target_num_samples

    init = rng.choice(len(X_maj), size=m, replace=False)
    S = X_maj[init].astype(float)

    for _ in range(sp_iters):
        batch_idx = rng.choice(len(X_maj), size=sp_batch_size, replace=False)
        B = X_maj[batch_idx]

        diff1 = S[:, None, :] - B[None, :, :]
        dist1 = np.linalg.norm(diff1, axis=2, keepdims=True)
        grad1 = (2.0 / B.shape[0]) * np.sum(diff1 / (dist1 + 1e-12), axis=1)

        diff2 = S[:, None, :] - S[None, :, :]
        dist2 = np.linalg.norm(diff2, axis=2, keepdims=True)
        mask = ~np.eye(m, dtype=bool)[:, :, None]
        grad2 = -(2.0 / m) * np.sum((diff2 / (dist2 + 1e-12)) * mask, axis=1)

        grad = grad1 + grad2
        S -= sp_lr * grad


    D = pairwise_distances(S, X_maj)
    nn = np.argmin(D, axis=1)
    sel_rel = np.unique(nn)
    sel_maj_idx = maj_idx[sel_rel]

    if len(sel_maj_idx) < m:
        remain = np.setdiff1d(maj_idx, sel_maj_idx)
        extra = rng.choice(remain, size=m - len(sel_maj_idx), replace=False)
        sel_maj_idx = np.concatenate([sel_maj_idx, extra])

    resampled_indices = np.concatenate([min_idx, sel_maj_idx])
    X_resampled = X[resampled_indices]
    Y_resampled = Y[resampled_indices]

    return X_resampled, Y_resampled, resampled_indices
