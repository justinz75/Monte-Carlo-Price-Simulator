import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from constants import DEFAULT_CONFIDENCE_LEVEL, DEFAULT_SEED, DEFAULT_N_PATHS, DEFAULT_N_PATHS_LARGE, DEFAULT_N_STEPS

rng = np.random.default_rng(seed = DEFAULT_SEED)

z = rng.standard_normal(10)
print(z)

mu, sigma = 0.05, 0.2
returns = rng.normal(mu, sigma, DEFAULT_N_PATHS)
print(f"Mean: {returns.mean():.4f}, Std: {returns.std():.4f}")

u = rng.uniform(size=DEFAULT_N_PATHS)

def simulate_gbm(S0, mu, sigma, T, dt, n_paths=DEFAULT_N_PATHS, seed = DEFAULT_SEED):
    """Simulate stock price paths using Geometric Brownian Motion paths"""
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    
    Z = rng.standard_normal((n_steps, n_paths))
    
    #drift and diffusion components of GBM model and log returns for each step
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_returns = drift + diffusion
    
    #cumulative log returns and exponentiate to get stock price paths
    log_paths = np.vstack([np.zeros(n_paths), np.cumsum(log_returns, axis=0)])
    paths = S0 * np.exp(log_paths)
    
    return paths

def european_call(S0, K, r, sigma, T, n_paths=DEFAULT_N_PATHS, seed=DEFAULT_SEED):
    """Price a European call option using Monte Carlo simulation"""
    rng = np.random.default_rng(seed)
    
    #simulate terminal stock prices
    Z = rng.standard_normal(n_paths)
    price_at_time = S0*np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(price_at_time - K, 0)
    price = np.exp(-r * T) * payoffs.mean()
    
    standard_error = np.exp(-r * T) * payoffs.std() / np.sqrt(n_paths)
    
    return price, standard_error

def black_scholes_call(S0, K, r, sigma, T):
    """Exact Black-Scholes price for a European call."""
    sqrt_T = np.sqrt(T)
    discount = np.exp(-r * T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * sqrt_T
    call_value = S0 * stats.norm.cdf(d1) - K * discount * stats.norm.cdf(d2)
    return call_value

def asian_call(S0, K, r, sigma, T, n_steps=DEFAULT_N_STEPS, n_paths=DEFAULT_N_PATHS, seed=DEFAULT_SEED):
    """Price an arithmetic average Asian call option."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    
    Z = rng.standard_normal((n_steps, n_paths))
    
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(log_returns, axis=0)
    paths = S0 * np.exp(log_paths)
    
    avg_prices = paths.mean(axis=0)
    
    payoffs = np.maximum(avg_prices - K, 0)
    discount = np.exp(-r * T)
    price = discount * np.mean(payoffs)
    standard_error = discount * np.std(payoffs) / np.sqrt(len(payoffs))
    
    return price, standard_error

def up_and_out_call(S0, K, B, r, sigma, T, n_steps=DEFAULT_N_STEPS, n_paths=DEFAULT_N_PATHS, seed=DEFAULT_SEED):
    """Price an up-and-out barrier call option."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    
    Z = rng.standard_normal((n_steps, n_paths))
    
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(log_returns, axis=0)
    paths = S0 * np.exp(log_paths)
    
    knocked_out = np.any(paths >= B, axis=0)
    
    price_at_time = paths[-1]
    payoffs = np.where(knocked_out, 0, np.maximum(price_at_time - K, 0))
    
    discount = np.exp(-r * T)
    price = discount * payoffs.mean()
    standard_error = discount * payoffs.std() / np.sqrt(n_paths)
    
    return price, standard_error

def monte_carlo_var(S0, mu, sigma, T, confidence = DEFAULT_CONFIDENCE_LEVEL, n_paths = DEFAULT_N_PATHS_LARGE, seed = DEFAULT_SEED):
    """Estimate Value at Risk (VaR) and Conditional Value at Risk (CVaR) using Monte Carlo simulation."""
    rng = np.random.default_rng(seed)
    
    Z = rng.standard_normal(n_paths)
    price_at_time = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    profit_and_losses = price_at_time - S0
    
    var = -np.percentile(profit_and_losses, (1 - confidence) * 100)
    
    tail_losses = profit_and_losses[profit_and_losses <= -var]
    cvar = -tail_losses.mean() if len(tail_losses) > 0 else 0.0
    
    return var, cvar, profit_and_losses

def simulate_correlated_portfolio(S0_vec, mu_vec, sigma_vec, correlation_matrix, T, dt, n_paths=DEFAULT_N_PATHS, seed=DEFAULT_SEED):
    """Simulate correlated GBM paths for multiple assets."""
    rng = np.random.default_rng(seed)
    n_assets = len(S0_vec)
    n_steps = int(T / dt)
    
    #build the covariance matrix from the correlation matrix and standard deviations
    D = np.diag(sigma_vec)
    covariance_matrix = D @ correlation_matrix @ D
    
    #cholesky decomposition to get lower triangular matrix for correlation
    L = np.linalg.cholesky(covariance_matrix)
    
    Z_independent = rng.standard_normal((n_steps, n_paths, n_assets))
    Z_correlation = Z_independent @ L.T 
    
    drift = (mu_vec - 0.5 * sigma_vec**2) * dt
    diffusion = np.sqrt(dt) * Z_correlation
    log_returns = drift + diffusion
    log_paths = np.cumsum(log_returns, axis=0)
    paths = S0_vec * np.exp(log_paths)
    
    return paths

def call_antithetic(S0, K, r, sigma, T, n_paths=DEFAULT_N_PATHS, seed=DEFAULT_SEED):
    """European call with antithetic variance reduction."""
    rng = np.random.default_rng(seed)
    
    Z = rng.standard_normal(n_paths)
    
    #simulate terminal stock prices for both the original and antithetic paths
    common_part = (r - 0.5 * sigma**2) * T
    volatility_part = sigma * np.sqrt(T)
    terminal_prices = S0 * np.exp(
    common_part + volatility_part * np.array([Z, -Z]))
    price_up, price_down = terminal_prices
    
    #calculate payoffs for the original and antithetic paths
    payoff_pos = np.maximum(price_up - K, 0)
    payoff_neg = np.maximum(price_down - K, 0)
    
    #average the payoffs from the original and antithetic paths
    paired_payoffs = 0.5 * (payoff_pos + payoff_neg)
    price = np.exp(-r * T) * paired_payoffs.mean()
    standard_error = np.exp(-r * T) * paired_payoffs.std() / np.sqrt(n_paths)
    
    return price, standard_error

def call_control_variate(S0, K, r, sigma, T, n_paths=DEFAULT_N_PATHS_LARGE, seed=DEFAULT_SEED):
    """European call with control variate variance reduction."""
    rng = np.random.default_rng(seed)
    
    Z = rng.standard_normal(n_paths)
    terminal_prices = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    payoffs = np.maximum(terminal_prices - K, 0)
    discount = np.exp(-r * T)
    
    #control variate: terminal stock price 
    #known expected value of terminal stock price under risk-neutral measure
    expected_price = S0 * np.exp(r * T)
    
    #estimate the covariance between payoffs and terminal prices to compute beta
    covariance_matrix = np.cov(payoffs, terminal_prices)
    beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
    
    #adjust payoffs using the control variate
    adjusted = payoffs - beta * (terminal_prices - expected_price)
    price = discount * adjusted.mean()
    standard_error = discount * adjusted.std() / np.sqrt(n_paths)
    
    return price, standard_error

def convergence_plot(S0, K, r, sigma, T, max_paths=200_000, seed=DEFAULT_SEED):
    """Show the convergence of the Monte Carlo estimate for a European call option price."""
    rng = np.random.default_rng(seed)
    
    Z = rng.standard_normal(max_paths)
    terminal_prices = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(terminal_prices - K, 0) * np.exp(-r * T)
    
    #calculate cumulative means and standard errors for increasing path counts
    checkpoints = np.arange(100, max_paths + 1, 100)
    cumulative_sum = np.cumsum(payoffs)
    cumulative_squared_sum = np.cumsum(payoffs**2)
    means = cumulative_sum[checkpoints - 1] / checkpoints
    variances = cumulative_squared_sum[checkpoints - 1] / checkpoints - means**2
    standard_errors = np.sqrt(np.maximum(variances, 0)) / np.sqrt(checkpoints)
    
    bs_price = black_scholes_call(S0, K, r, sigma, T)
    
    #plot the convergence of the Monte Carlo estimate with 95% confidence intervals
    plt.figure(figsize=(10, 6))
    plt.plot(checkpoints, means, linewidth=0.8, color="steelblue", label="Monte Carlo estimate")
    plt.fill_between(
        checkpoints, means - 1.96 * standard_errors, means + 1.96 * standard_errors,
        alpha=0.2, color="steelblue", label="95% Confidence Interval"
    )
    plt.axhline(bs_price, color="red", linestyle="--", label=f"Black-Scholes: £{bs_price:.4f}")
    plt.xlabel("Number of Paths")
    plt.ylabel("Option Price (£)")
    plt.title("Monte Carlo Convergence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def monte_carlo_pricer(S0, K, r, sigma, T, n_paths=DEFAULT_N_PATHS, seed=DEFAULT_SEED):
    """
    European call option pricer with antithetic sampling and control variates.
    Returns price, standard error, and 95% confidence interval.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    
    #generate the original and antithetic terminal prices separately
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T)
    price_up = S0 * np.exp(drift + diffusion * Z)
    price_down = S0 * np.exp(drift - diffusion * Z)

    payoff_pairs = 0.5 * (
        np.maximum(price_up - K, 0)
        + np.maximum(price_down - K, 0)
    )
    price_pairs = 0.5 * (price_up + price_down)

    expected_price = S0 * np.exp(r * T)
    
    covariance_est = np.cov(payoff_pairs, price_pairs)
    beta = covariance_est[0, 1] / covariance_est[1, 1]
    
    adjusted = payoff_pairs - beta * (price_pairs - expected_price)
    
    discount = np.exp(-r * T)
    price = discount * adjusted.mean()
    standard_error = discount * adjusted.std() / np.sqrt(n_paths)
    confidence_interval = (price - 1.96 * standard_error, price + 1.96 * standard_error)

    return price, standard_error, confidence_interval

#run the full-featured Monte Carlo pricer and compare with Black-Scholes
price, standard_error, confidence_interval = monte_carlo_pricer(S0=100, K=105, r=0.05, sigma=0.25, T=1.0)
bs = black_scholes_call(100, 105, 0.05, 0.25, 1.0)

print(f"MC Price:       £{price:.4f}")
print(f"Std Error:      £{standard_error:.6f}")
print(f"95% CI:         [£{confidence_interval[0]:.4f}, £{confidence_interval[1]:.4f}]")
print(f"Black-Scholes:  £{bs:.4f}")
print(f"Absolute Error: £{abs(price - bs):.6f}")