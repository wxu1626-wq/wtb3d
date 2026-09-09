"""Model wrappers (CPU, scikit-learn based). All expose fit / predict_proba."""
from sklearn.ensemble import (RandomForestClassifier,
                              HistGradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier


def make_model(kind="rf", n_estimators=300):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1,
                                      random_state=0)
    if kind == "gbm":
        return HistGradientBoostingClassifier(
            max_iter=max(50, n_estimators // 3), learning_rate=0.06,
            max_depth=None, l2_regularization=1.0, random_state=0)
    if kind == "mlp":
        return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400,
                             random_state=0)
    raise ValueError("unknown model kind: %s" % kind)