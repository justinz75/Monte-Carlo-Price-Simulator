import numpy as np
from scipy import optimize, stats
import matplotlib.pyplot as plt
from constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_SEED,
    DEFAULT_N_PATHS,
    DEFAULT_N_PATHS_LARGE,
    DEFAULT_N_STEPS,
)
from multiprocessing import Pool, cpu_count

def validate_inputs(S0=None, K=None, B=None, sigma=None, T=None,
                    n_paths=None, n_steps=None):
    """Check the inputs are valid, leaving r unchecked as rates can be negative."""
    if S0 is not None and np.any(np.asarray(S0) <= 0):
        raise ValueError(f"S0 must be positive, got {S0}")
    if K is not None and np.any(np.asarray(K) < 0):
        raise ValueError(f"K must be non-negative, got {K}")
    if B is not None and np.any(np.asarray(B) <= 0):
        raise ValueError(f"B must be positive, got {B}")
    if sigma is not None and np.any(np.asarray(sigma) <= 0):
        raise ValueError(f"sigma must be positive, got {sigma}")
    if T is not None and np.any(np.asarray(T) <= 0):
        raise ValueError(f"T must be positive, got {T}")
    if n_paths is not None and n_paths < 1:
        raise ValueError(f"n_paths must be at least 1, got {n_paths}")
    if n_steps is not None and n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")

def simulate_gbm(S0, mu, sigma, T, dt, n_paths=DEFAULT_N_PATHS, seed = DEFAULT_SEED):
    """Simulate stock price paths using Geometric Brownian Motion paths"""
    validate_inputs(S0=S0, sigma=sigma, T=T, n_paths=n_paths)
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    
    Z = rng.standard_normal((n_steps, n_paths), dtype=np.float32)
    
    #drift and diffusion components of GBM model and log returns for each step
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_returns = drift + diffusion
    
    #cumulative log returns and exponentiate to get stock price paths
    #float32 zeros so vstack doesn't upcast the paths to float64
    log_paths = np.vstack([np.zeros(n_paths, dtype=np.float32), np.cumsum(log_returns, axis=0)])
    paths = S0 * np.exp(log_paths)
    
    return paths

def european_call(S0, K, r, sigma, T, n_paths=DEFAULT_N_PATHS, seed=DEFAULT_SEED):
    """Price a European call option using Monte Carlo simulation"""
    validate_inputs(S0=S0, K=K, sigma=sigma, T=T, n_paths=n_paths)
    rng = np.random.default_rng(seed)
    
    #simulate terminal stock prices
    Z = rng.standard_normal(n_paths, dtype=np.float32)
    price_at_time = S0*np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(price_at_time - K, 0)
    #accumulate in float64 to keep precision over many paths
    price = np.exp(-r * T) * payoffs.mean(dtype=np.float64)
    
    standard_error = np.exp(-r * T) * payoffs.std(dtype=np.float64) / np.sqrt(n_paths)
    
    return price, standard_error

def european_put(S0, K, r, sigma, T, n_paths=DEFAULT_N_PATHS, seed=DEFAULT_SEED):
    """Price a European put option using Monte Carlo simulation"""
    validate_inputs(S0=S0, K=K, sigma=sigma, T=T, n_paths=n_paths)
    rng = np.random.default_rng(seed)
    
    #simulate terminal stock prices
    Z = rng.standard_normal(n_paths, dtype=np.float32)
    price_at_time = S0*np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(K - price_at_time, 0)
    price = np.exp(-r * T) * payoffs.mean(dtype=np.float64)
    
    standard_error = np.exp(-r * T) * payoffs.std(dtype=np.float64) / np.sqrt(n_paths)
    
    return price, standard_error

def european_call_batched(S0, K, r, sigma, T, n_paths=100_000, batch_size=10_000, seed=DEFAULT_SEED):
    """Memory-efficient batch processing"""
    validate_inputs(S0=S0, K=K, sigma=sigma, T=T, n_paths=n_paths)
    rng = np.random.default_rng(seed)
    total_payoff = 0.0
    total_payoff_sq = 0.0
    
    for i in range(0, n_paths, batch_size):
        current_batch = min(batch_size, n_paths - i)
        Z = rng.standard_normal(current_batch, dtype=np.float32)
        price_at_time = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        payoffs = np.maximum(price_at_time - K, 0)
        
        total_payoff += payoffs.sum(dtype=np.float64)
        total_payoff_sq += np.square(payoffs, dtype=np.float64).sum(dtype=np.float64)
    
    mean_payoff = total_payoff / n_paths
    variance = max((total_payoff_sq / n_paths) - mean_payoff ** 2, 0.0)
    price = np.exp(-r * T) * mean_payoff
    standard_error = np.exp(-r * T) * np.sqrt(variance / n_paths)
    
    return price, standard_error

def parallel_european_chunk(args):
    """Worker function for parallel simulation"""
    S0, K, r, sigma, T, seed, n_paths = args
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths, dtype=np.float32)
    price_at_time = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(price_at_time - K, 0)
    return payoffs

def european_call_parallel(S0, K, r, sigma, T, n_paths=100_000, seed=DEFAULT_SEED):
    """Parallel Monte Carlo using multiprocessing"""
    validate_inputs(S0=S0, K=K, sigma=sigma, T=T, n_paths=n_paths)
    n_processes = min(cpu_count(), n_paths)
    path_counts = [
        n_paths // n_processes + (worker_id < n_paths % n_processes)
        for worker_id in range(n_processes)
    ]
    
    #create a list of tasks for each process, each with a unique seed
    tasks = [
        (S0, K, r, sigma, T, seed + worker_id, path_count)
        for worker_id, path_count in enumerate(path_counts)
    ]
    with Pool(n_processes) as pool: results = pool.map(parallel_european_chunk, tasks)
    all_payoffs = np.concatenate(results)
    price = np.exp(-r * T) * all_payoffs.mean(dtype=np.float64)
    standard_error = np.exp(-r * T) * all_payoffs.std(dtype=np.float64) / np.sqrt(n_paths)
    
    return price, standard_error

def black_scholes_call(S0, K, r, sigma, T, q=0.0):
    """Exact Black-Scholes price for a European call, with an optional dividend yield q."""
    validate_inputs(S0=S0, K=K, sigma=sigma, T=T)
    sqrt_T = np.sqrt(T)
    discount = np.exp(-r * T)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    call_value = S0 * np.exp(-q * T) * stats.norm.cdf(d1) - K * discount * stats.norm.cdf(d2)
    return call_value

def black_scholes_put(S0, K, r, sigma, T, q=0.0):
    """Exact Black-Scholes price for a European put, with an optional dividend yield q."""
    validate_inputs(S0=S0, K=K, sigma=sigma, T=T)
    sqrt_T = np.sqrt(T)
    discount = np.exp(-r * T)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    put_value = K * discount * stats.norm.cdf(-d2) - S0 * np.exp(-q * T) * stats.norm.cdf(-d1)
    return put_value

def implied_volatility(price, S0, K, r, T, q=0.0, option_type="call",
                       sigma_low=1e-4, sigma_high=5.0):
    """Black-Scholes implied volatility using Brent's method, nan if the price can't be matched."""
    validate_inputs(S0=S0, K=K, T=T)
    if option_type == "call":
        pricer = black_scholes_call
    elif option_type == "put":
        pricer = black_scholes_put
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if not price > 0:
        return np.nan

    def pricing_error(sigma):
        return pricer(S0, K, r, sigma, T, q) - price

    #check the price can be matched between the volatility bounds
    if pricing_error(sigma_low) >= 0 or pricing_error(sigma_high) <= 0:
        return np.nan
    return optimize.brentq(pricing_error, sigma_low, sigma_high)

def black_scholes_delta(S0, K, r, sigma, T, q=0.0, option_type="call"):
    """Exact Black-Scholes delta for a European call or put, with an optional dividend yield q."""
    validate_inputs(S0=S0, K=K, sigma=sigma, T=T)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return np.exp(-q * T) * stats.norm.cdf(d1)
    if option_type == "put":
        return -np.exp(-q * T) * stats.norm.cdf(-d1)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

def monte_carlo_delta(S0, K, r, sigma, T, option_type="call", n_paths=DEFAULT_N_PATHS_LARGE, seed=DEFAULT_SEED):
    """European call or put delta using the pathwise method."""
    validate_inputs(S0=S0, K=K, sigma=sigma, T=T, n_paths=n_paths)
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    rng = np.random.default_rng(seed)

    #simulate terminal stock prices
    Z = rng.standard_normal(n_paths, dtype=np.float32)
    price_at_time = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    #pathwise delta: the payoff moves by S_T / S0 per unit of S0 on paths that end in the money
    if option_type == "call":
        path_deltas = np.where(price_at_time > K, price_at_time / S0, 0.0)
    else:
        path_deltas = np.where(price_at_time < K, -price_at_time / S0, 0.0)

    discount = np.exp(-r * T)
    delta = discount * path_deltas.mean(dtype=np.float64)
    standard_error = discount * path_deltas.std(dtype=np.float64) / np.sqrt(n_paths)

    return delta, standard_error

def asian_call(S0, K, r, sigma, T, n_steps=DEFAULT_N_STEPS,
               n_paths=DEFAULT_N_PATHS, batch_size=DEFAULT_BATCH_SIZE,
               seed=DEFAULT_SEED):
    """Price an arithmetic average Asian call option in path batches."""
    validate_inputs(S0=S0, K=K, sigma=sigma, T=T, n_paths=n_paths, n_steps=n_steps)
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    total_payoff = 0.0
    total_payoff_sq = 0.0
    for start in range(0, n_paths, batch_size):
        current_batch = min(batch_size, n_paths - start)
        Z = rng.standard_normal((n_steps, current_batch), dtype=np.float32)
        log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        log_paths = np.cumsum(log_returns, axis=0, dtype=np.float32)
        paths = S0 * np.exp(log_paths)
        payoffs = np.maximum(paths.mean(axis=0, dtype=np.float64) - K, 0)
        total_payoff += payoffs.sum(dtype=np.float64)
        total_payoff_sq += np.square(payoffs, dtype=np.float64).sum(dtype=np.float64)

    discount = np.exp(-r * T)
    mean_payoff = total_payoff / n_paths
    variance = max((total_payoff_sq / n_paths) - mean_payoff**2, 0.0)
    price = discount * mean_payoff
    standard_error = discount * np.sqrt(variance / n_paths)
    
    return price, standard_error

def asian_call_low_memory(S0, K, r, sigma, T, n_steps=DEFAULT_N_STEPS, n_paths=DEFAULT_N_PATHS_LARGE, seed=DEFAULT_SEED):
    """Price an arithmetic average Asian call option with memory optimization but slower execution."""
    validate_inputs(S0=S0, K=K, sigma=sigma, T=T, n_paths=n_paths, n_steps=n_steps)
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    
    price_sum = np.zeros(n_paths, dtype=np.float32)
    current_price = np.full(n_paths, S0, dtype=np.float32)
    
    for step in range(n_steps):
        Z = rng.standard_normal(n_paths, dtype=np.float32)
        log_return = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        current_price = current_price * np.exp(log_return)
        price_sum += current_price
    
    avg_prices = price_sum / n_steps
    payoffs = np.maximum(avg_prices - K, 0)
    discount = np.exp(-r * T)
    price = discount * np.mean(payoffs, dtype=np.float64)
    standard_error = discount * np.std(payoffs, dtype=np.float64) / np.sqrt(n_paths)
    
    return price, standard_error

def up_and_out_call(S0, K, B, r, sigma, T, n_steps=DEFAULT_N_STEPS,
                    n_paths=DEFAULT_N_PATHS, batch_size=DEFAULT_BATCH_SIZE,
                    seed=DEFAULT_SEED):
    """Price an up-and-out barrier call option in path batches, checking the barrier once per step."""
    validate_inputs(S0=S0, K=K, B=B, sigma=sigma, T=T, n_paths=n_paths, n_steps=n_steps)
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    total_payoff = 0.0
    total_payoff_sq = 0.0
    for start in range(0, n_paths, batch_size):
        current_batch = min(batch_size, n_paths - start)
        Z = rng.standard_normal((n_steps, current_batch), dtype=np.float32)
        log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        log_paths = np.cumsum(log_returns, axis=0, dtype=np.float32)
        paths = S0 * np.exp(log_paths)
        knocked_out = np.any(paths >= B, axis=0)
        payoffs = np.where(
            knocked_out,
            0,
            np.maximum(paths[-1] - K, 0),
        )
        total_payoff += payoffs.sum(dtype=np.float64)
        total_payoff_sq += np.square(payoffs, dtype=np.float64).sum(dtype=np.float64)

    discount = np.exp(-r * T)
    mean_payoff = total_payoff / n_paths
    variance = max((total_payoff_sq / n_paths) - mean_payoff**2, 0.0)
    price = discount * mean_payoff
    standard_error = discount * np.sqrt(variance / n_paths)
    
    return price, standard_error

def monte_carlo_var(S0, mu, sigma, T, confidence = DEFAULT_CONFIDENCE_LEVEL, n_paths = DEFAULT_N_PATHS_LARGE, seed = DEFAULT_SEED):
    """Estimate Value at Risk (VaR) and Conditional Value at Risk (CVaR) using Monte Carlo simulation."""
    validate_inputs(S0=S0, sigma=sigma, T=T, n_paths=n_paths)
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must lie strictly between 0 and 1, got {confidence}")
    rng = np.random.default_rng(seed)
    
    Z = rng.standard_normal(n_paths, dtype=np.float32)
    price_at_time = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    profit_and_losses = price_at_time - S0
    
    var = -np.percentile(profit_and_losses, (1 - confidence) * 100)
    
    tail_losses = profit_and_losses[profit_and_losses <= -var]
    cvar = -tail_losses.mean() if len(tail_losses) > 0 else 0.0
    
    return var, cvar, profit_and_losses

def simulate_correlated_portfolio(S0_vec, mu_vec, sigma_vec, correlation_matrix, T, dt, n_paths=DEFAULT_N_PATHS, seed=DEFAULT_SEED):
    """Simulate correlated GBM paths for multiple assets."""
    validate_inputs(S0=S0_vec, sigma=sigma_vec, T=T, n_paths=n_paths)
    rng = np.random.default_rng(seed)
    n_assets = len(S0_vec)
    n_steps = int(T / dt)
    
    #build the covariance matrix from the correlation matrix and standard deviations
    D = np.diag(sigma_vec)
    covariance_matrix = D @ correlation_matrix @ D
    
    #cholesky decomposition to get lower triangular matrix for correlation
    L = np.linalg.cholesky(covariance_matrix)
    
    Z_independent = rng.standard_normal((n_steps, n_paths, n_assets), dtype=np.float32)
    Z_correlation = Z_independent @ L.T 
    
    drift = (mu_vec - 0.5 * sigma_vec**2) * dt
    diffusion = np.sqrt(dt) * Z_correlation
    log_returns = drift + diffusion
    log_paths = np.cumsum(log_returns, axis=0)
    paths = S0_vec * np.exp(log_paths)
    
    return paths

def call_antithetic(S0, K, r, sigma, T, n_paths=DEFAULT_N_PATHS, seed=DEFAULT_SEED):
    """European call with antithetic variance reduction."""
    validate_inputs(S0=S0, K=K, sigma=sigma, T=T, n_paths=n_paths)
    rng = np.random.default_rng(seed)
    
    Z = rng.standard_normal(n_paths, dtype=np.float32)
    
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
    validate_inputs(S0=S0, K=K, sigma=sigma, T=T, n_paths=n_paths)
    rng = np.random.default_rng(seed)
    
    Z = rng.standard_normal(n_paths, dtype=np.float32)
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

def convergence_plot(S0, K, r, sigma, T, max_paths=200_000, seed=DEFAULT_SEED, save_path=None):
    """Show the convergence of the Monte Carlo estimate for a European call option price."""
    validate_inputs(S0=S0, K=K, sigma=sigma, T=T, n_paths=max_paths)
    rng = np.random.default_rng(seed)
    
    Z = rng.standard_normal(max_paths, dtype=np.float32)
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
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()

def monte_carlo_pricer(S0, K, r, sigma, T, n_paths=DEFAULT_N_PATHS, seed=DEFAULT_SEED):
    """
    European call option pricer with antithetic sampling and control variates.
    Returns price, standard error, and 95% confidence interval.
    Beta is estimated from the same paths, so the standard error is slightly optimistic.
    """
    validate_inputs(S0=S0, K=K, sigma=sigma, T=T, n_paths=n_paths)
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths, dtype=np.float32)
    
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

def main():
    price, standard_error, confidence_interval = monte_carlo_pricer(
        S0=100, K=105, r=0.05, sigma=0.25, T=1.0
    )
    bs = black_scholes_call(100, 105, 0.05, 0.25, 1.0)

    print(f"MC Price:       £{price:.4f}")
    print(f"Std Error:      £{standard_error:.6f}")
    print(f"95% CI:         [£{confidence_interval[0]:.4f}, £{confidence_interval[1]:.4f}]")
    print(f"Black-Scholes:  £{bs:.4f}")
    print(f"Absolute Error: £{abs(price - bs):.6f}")


if __name__ == "__main__":
    main()