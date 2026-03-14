import numpy as np


def demographic_parity_difference(y_true, y_pred, sensitive_attribute):
    """
    Regression adaptation of demographic parity.
    Compares mean predicted values across groups.
    A difference of 0 means perfect parity.
    """
    groups = np.unique(sensitive_attribute)
    means = {}
    for group in groups:
        mask = sensitive_attribute == group
        means[str(group)] = float(np.mean(y_pred[mask]))

    mean_values = list(means.values())
    difference = max(mean_values) - min(mean_values)

    return {
        'group_means': means,
        'difference': difference,
    }


def disparate_impact_ratio(y_true, y_pred, sensitive_attribute,
                           unprivileged_value, privileged_value):
    """
    Regression adaptation of disparate impact.
    Ratio of mean predicted charges between groups.
    A ratio of 1 means perfect fairness.
    """
    mask_unpriv = sensitive_attribute == unprivileged_value
    mask_priv = sensitive_attribute == privileged_value

    mean_unpriv = float(np.mean(y_pred[mask_unpriv]))
    mean_priv = float(np.mean(y_pred[mask_priv]))

    ratio = mean_unpriv / mean_priv if mean_priv > 0 else 0.0

    return {
        'unprivileged_mean': mean_unpriv,
        'privileged_mean': mean_priv,
        'ratio': ratio,
    }


def group_regression_metrics(y_true, y_pred, sensitive_attribute):
    """
    Compute R², RMSE, MAE per group for regression fairness analysis.
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    groups = np.unique(sensitive_attribute)
    results = {}
    for group in groups:
        mask = sensitive_attribute == group
        yt, yp = y_true[mask], y_pred[mask]
        results[str(group)] = {
            'n': int(mask.sum()),
            'R²': float(r2_score(yt, yp)),
            'RMSE': float(mean_squared_error(yt, yp) ** 0.5),
            'MAE': float(mean_absolute_error(yt, yp)),
            'Mean Actual': float(np.mean(yt)),
            'Mean Predicted': float(np.mean(yp)),
        }
    return results
