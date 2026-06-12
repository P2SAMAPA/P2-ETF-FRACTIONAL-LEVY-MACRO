import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def fractional_levy_score(returns, macro_df, H=0.45, jump_threshold=0.02):
    """
    Compute expected jump return = λ × μⱼ, where λ depends on macro via logistic regression.
    - First, detect jumps in returns (|return| > threshold).
    - Estimate jump size mu_j as mean of positive jumps minus mean of negative jumps.
    - Use macro variables to predict jump probability via logistic regression.
    - Score = predicted jump probability × mu_j.
    """
    if len(returns) < 10 or macro_df is None or len(macro_df) < 10:
        return 0.0
    # Align lengths
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Detect jumps
    positive_jumps = returns[returns > jump_threshold]
    negative_jumps = returns[returns < -jump_threshold]
    jump_indicators = (np.abs(returns) > jump_threshold).astype(int)
    # Jump intensity (unconditional)
    lam = np.mean(jump_indicators)
    # Expected jump size (positive minus negative)
    mu_j = np.mean(positive_jumps) - np.mean(negative_jumps) if len(positive_jumps) > 0 and len(negative_jumps) > 0 else 0.0
    if lam == 0 or mu_j == 0:
        return 0.0
    # Use macro to predict jump probability via logistic regression
    # Scale macro
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    # Fit logistic regression
    try:
        log_reg = LogisticRegression(max_iter=200)
        log_reg.fit(macro_scaled, jump_indicators)
        # Predict probability of jump for the last macro observation
        last_macro = macro_df.iloc[-1].values.reshape(1, -1)
        last_macro_scaled = scaler.transform(last_macro)
        prob_jump = log_reg.predict_proba(last_macro_scaled)[0, 1]
    except:
        prob_jump = lam  # fallback to unconditional
    # Expected jump return
    expected_jump_return = prob_jump * mu_j
    return float(expected_jump_return)
