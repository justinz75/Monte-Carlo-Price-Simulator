import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#set a seed for reproducibility
rng = np.random.default_rng(seed = 50)

#draw 10 samples from a standard normal distribution
z = rng.standard_normal(10)
print(z)

#draw from N(mu, sigma^2)
mu, sigma = 0.05, 0.2
returns = rng.normal(mu, sigma, 10_000)
print(f"Mean: {returns.mean():.4f}, Std: {returns.std():.4f}")

#uniform random numbers on [0, 1)
u = rng.uniform(size=10_000)

def simulate_gbm(S0, mu, sigma, T, dt, n_paths, seed = 50):
    """Simulate stock price paths using Geometric Brownian Motion paths"""
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    
    #generate standard normal random variables for the simulation
    Z = rng.standard_normal((n_steps, n_paths))
    
    #calculate the drift and diffusion components of the GBM model and the log returns for each step
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_returns = drift + diffusion
    
    #calculate the cumulative log returns and exponentiate to get stock price paths
    log_paths = np.vstack([np.zeros(n_paths), np.cumsum(log_returns, axis=0)])
    paths = S0 * np.exp(log_paths)
    
    return paths

# #parameters for the GBM simulation
# S0 = 100  #initial stock price
# mu = 0.05  #5% expected annual return
# sigma = 0.2  #20% annual volatility
# T = 1.0  #time horizon in years
# dt = 1/252  #daily time steps
# n_paths = 1000  #number of simulated paths

# paths = simulate_gbm(S0, mu, sigma, T, dt, n_paths)
# print(f"Simulated stock price paths shape: {paths.shape}")
# print(f"Final stock prices: {paths[-1, :5]}")  #print the final stock prices of the first 5 paths

# time_grid = np.linspace(0, T, paths.shape[0])

# #visualise the first 50 simulated stock price paths
# plt.figure(figsize=(10, 6))
# plt.plot(time_grid, paths[:, :50], alpha=0.3, linewidth=0.5)
# plt.xlabel("Time (years)")
# plt.ylabel("Stock Price (£)")
# plt.title("Monte Carlo Stock Price Simulation - 50 of 10,000 Paths")
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()


def european_call(S0, K, r, sigma, T, n_paths=10_000, seed=50):
    """Price a European call option using Monte Carlo simulation"""
    rng = np.random.default_rng(seed)
    
    #simulate terminal stock prices
    Z = rng.standard_normal(n_paths)
    ST = S0*np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0)
    price = np.exp(-r * T) * payoffs.mean()
    
    #standard error of the estimate
    std_error = np.exp(-r * T) * payoffs.std() / np.sqrt(n_paths)
    
    return price, std_error

#parameters for the European call option
# S0, K, r, sigma, T = 100, 110, 0.05, 0.2, 1.0
# mc_price, mc_std_error = european_call(S0, K, r, sigma, T)
# print(f"Monte Carlo Price: {mc_price:.4f}, Standard Error: {mc_std_error:.4f}")


def black_scholes_call(S0, K, r, sigma, T):
    """Exact Black-Scholes price for a European call."""
    sqrt_T = np.sqrt(T)
    discount = np.exp(-r * T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * sqrt_T
    call_value = S0 * stats.norm.cdf(d1) - K * discount * stats.norm.cdf(d2)
    return call_value

# #parameters for the Black-Scholes price
# bs_price = black_scholes_call(S0, K, r, sigma, T)
# print(f"Black-Scholes price: £{bs_price:.4f}")
# print(f"MC error: £{abs(mc_price - bs_price):.4f}")


def asian_call(S0, K, r, sigma, T, n_steps=252, n_paths=100_000, seed=50):
    """Price an arithmetic average Asian call option."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    
    Z = rng.standard_normal((n_steps, n_paths))
    
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(log_returns, axis=0)
    paths = S0 * np.exp(log_paths)
    
    #arithmetic average across all time steps for each path
    avg_prices = paths.mean(axis=0)
    
    payoffs = np.maximum(avg_prices - K, 0)
    discount = np.exp(-r * T)
    price = discount * np.mean(payoffs)
    se = discount * np.std(payoffs) / np.sqrt(len(payoffs))
    
    return price, se

# #parameters for the Asian call option
# asian_price, asian_se = asian_call(S0=100, K=105, r=0.05, sigma=0.25, T=1.0)
# print(f"Asian call price: £{asian_price:.4f} (SE: {asian_se:.4f})")

def up_and_out_call(S0, K, B, r, sigma, T, n_steps=252, n_paths=100_000, seed=50):
    """Price an up-and-out barrier call option."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    
    Z = rng.standard_normal((n_steps, n_paths))
    
    #simulate the stock price paths using Geometric Brownian Motion
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(log_returns, axis=0)
    paths = S0 * np.exp(log_paths)
    
    #check if any path crossed the barrier
    knocked_out = np.any(paths >= B, axis=0)
    
    #terminal payoff, zeroed out for knocked-out paths
    ST = paths[-1]
    payoffs = np.where(knocked_out, 0, np.maximum(ST - K, 0))
    
    #discounted expected payoff and standard error
    discount = np.exp(-r * T)
    price = discount * payoffs.mean()
    se = discount * payoffs.std() / np.sqrt(n_paths)
    
    return price, se

# #parameters for the up-and-out barrier call option
# barrier_price, barrier_se = up_and_out_call(S0=100, K=100, B=130, r=0.05, sigma=0.25, T=1.0)
# print(f"Up-and-out call price: £{barrier_price:.4f} (SE: {barrier_se:.4f})")

def monte_carlo_var(S0, mu, sigma, T, confidence = 0.95, n_paths = 100_000, seed = 50):
    """Estimate Value at Risk (VaR) and Conditional Value at Risk (CVaR) using Monte Carlo simulation."""
    rng = np.random.default_rng(seed)
    
    #simulate terminal stock prices
    Z = rng.standard_normal(n_paths)
    ST = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    #calculate losses
    profit_and_losses = ST - S0
    
    #calculate the VaR at the specified confidence level
    var = -np.percentile(profit_and_losses, (1 - confidence) * 100)
    
    #calculate the CVaR (expected shortfall) at the specified confidence level
    tail_losses = profit_and_losses[profit_and_losses <= -var]
    cvar = -tail_losses.mean() if len(tail_losses) > 0 else 0.0
    
    return var, cvar, profit_and_losses

# #parameters for the VaR calculation
# var, cvar, profit_and_losses = monte_carlo_var(S0=100, mu=0.05, sigma=0.25, T=1.0)
# print(f"Value at Risk (VaR): £{var:.4f}")
# print(f"Conditional Value at Risk (CVaR): £{cvar:.4f}")

##visualise the profit and loss distribution
# plt.figure(figsize=(10, 6))
# plt.hist(profit_and_losses, bins=200, density=True, alpha=0.7, color="steelblue", edgecolor="none")
# plt.axvline(-var, color="red", linestyle="--", linewidth=2, label=f"95% VaR: £{var:,.0f}")
# plt.axvline(-cvar, color="darkred", linestyle="--", linewidth=2, label=f"95% CVaR: £{cvar:,.0f}")
# plt.xlabel("Profit / Loss (£)")
# plt.ylabel("Density")
# plt.title("Monte Carlo P&L Distribution with VaR and CVaR")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

def simulate_correlated_portfolio(S0_vec, mu_vec, sigma_vec, corr_matrix, T, dt, n_paths=10_000, seed=50):
    """Simulate correlated GBM paths for multiple assets."""
    rng = np.random.default_rng(seed)
    n_assets = len(S0_vec)
    n_steps = int(T / dt)
    
    #build the covariance matrix from the correlation matrix and standard deviations
    D = np.diag(sigma_vec)
    cov_matrix = D @ corr_matrix @ D
    
    #cholesky decomposition to get lower triangular matrix for correlation
    L = np.linalg.cholesky(cov_matrix)
    
    #generate independent standard normal random variables and apply the correlation
    Z_indep = rng.standard_normal((n_steps, n_paths, n_assets))
    Z_corr = Z_indep @ L.T  # (n_steps, n_paths, n_assets)
    
    #simulate each asset
    drift = (mu_vec - 0.5 * sigma_vec**2) * dt
    diffusion = np.sqrt(dt) * Z_corr
    log_returns = drift + diffusion
    log_paths = np.cumsum(log_returns, axis=0)
    paths = S0_vec * np.exp(log_paths)
    
    return paths

#3 correlated assets
portfolio_S0 = np.array([100, 50, 200])
portfolio_mu = np.array([0.08, 0.06, 0.10])
portfolio_sigma = np.array([0.20, 0.15, 0.30])
corr = np.array([
    [1.0, 0.5, 0.3],
    [0.5, 1.0, 0.2],
    [0.3, 0.2, 1.0]
])
weights = np.array([0.4, 0.3, 0.3])

#simulate correlated portfolio paths
paths = simulate_correlated_portfolio(
    portfolio_S0,
    portfolio_mu,
    portfolio_sigma,
    corr,
    T=1.0,
    dt=1/252,
)
print(f"Shape: {paths.shape}")  #(252, 10000, 3)

#portfolio value paths
initial_investment = 100_000
shares = (weights * initial_investment) / portfolio_S0
portfolio_paths = paths @ shares  #(252, 10000)

#compute terminal portfolio values and P&L
terminal_values = portfolio_paths[-1]
pnl = terminal_values - initial_investment

var_99 = -np.percentile(pnl, 1)
cvar_99 = -pnl[pnl <= -var_99].mean()

# print(f"Portfolio 99% 1-year VaR:  £{var_99:,.2f}")
# print(f"Portfolio 99% 1-year CVaR: £{cvar_99:,.2f}")
# print(f"Mean return: {pnl.mean() / initial_investment:.2%}")

def call_antithetic(S0, K, r, sigma, T, n_paths=50_000, seed=50):
    """European call with antithetic variance reduction."""
    rng = np.random.default_rng(seed)
    
    Z = rng.standard_normal(n_paths)
    
    #simulate terminal stock prices for both the original and antithetic paths
    common_part = (r - 0.5 * sigma**2) * T
    volatility_part = sigma * np.sqrt(T)
    terminal_prices = S0 * np.exp(
    common_part + volatility_part * np.array([Z, -Z]))
    ST_pos, ST_neg = terminal_prices
    
    #calculate payoffs for the original and antithetic paths
    payoff_pos = np.maximum(ST_pos - K, 0)
    payoff_neg = np.maximum(ST_neg - K, 0)
    
    #average the payoffs from the original and antithetic paths
    paired_payoffs = 0.5 * (payoff_pos + payoff_neg)
    price = np.exp(-r * T) * paired_payoffs.mean()
    se = np.exp(-r * T) * paired_payoffs.std() / np.sqrt(n_paths)
    
    return price, se

# #compare standard Monte Carlo and antithetic variates for European call option pricing
# std_price, std_se = european_call(S0, K, r, sigma, T, n_paths=100_000)
# anti_price, anti_se = call_antithetic(S0, K, r, sigma, T, n_paths=50_000)

# print(f"Standard MC:  £{std_price:.4f} (SE: {std_se:.4f})")
# print(f"Antithetic:   £{anti_price:.4f} (SE: {anti_se:.4f})")
# print(f"SE reduction:  {(1 - anti_se/std_se):.1%}")

def call_control_variate(S0, K, r, sigma, T, n_paths=100_000, seed=50):
    """European call with control variate variance reduction."""
    rng = np.random.default_rng(seed)
    
    Z = rng.standard_normal(n_paths)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    payoffs = np.maximum(ST - K, 0)
    discount = np.exp(-r * T)
    
    #control variate: terminal stock price ST
    #known expected value of ST under risk-neutral measure
    expected_ST = S0 * np.exp(r * T)
    
    #estimate the covariance between payoffs and ST to compute beta
    cov_matrix = np.cov(payoffs, ST)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    
    #adjust payoffs using the control variate
    adjusted = payoffs - beta * (ST - expected_ST)
    price = discount * adjusted.mean()
    se = discount * adjusted.std() / np.sqrt(n_paths)
    
    return price, se

# #compare standard Monte Carlo and control variate for European call option pricing
# cv_price, cv_se = call_control_variate(S0, K, r, sigma, T)
#print(f"Control var:  £{cv_price:.4f} (SE: {cv_se:.4f})")
#print(f"SE reduction vs standard: {(1 - cv_se/std_se):.1%}")

def convergence_plot(S0, K, r, sigma, T, max_paths=200_000, seed=50):
    """Show the convergence of the Monte Carlo estimate for a European call option price."""
    rng = np.random.default_rng(seed)
    
    Z = rng.standard_normal(max_paths)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0) * np.exp(-r * T)
    
    #calculate cumulative means and standard errors for increasing path counts
    checkpoints = np.arange(100, max_paths + 1, 100)
    cumulative_sum = np.cumsum(payoffs)
    cumulative_squared_sum = np.cumsum(payoffs**2)
    means = cumulative_sum[checkpoints - 1] / checkpoints
    variances = cumulative_squared_sum[checkpoints - 1] / checkpoints - means**2
    ses = np.sqrt(np.maximum(variances, 0)) / np.sqrt(checkpoints)
    
    bs_price = black_scholes_call(S0, K, r, sigma, T)
    
    #plot the convergence of the Monte Carlo estimate with 95% confidence intervals
    plt.figure(figsize=(10, 6))
    plt.plot(checkpoints, means, linewidth=0.8, color="steelblue", label="MC estimate")
    plt.fill_between(
        checkpoints, means - 1.96 * ses, means + 1.96 * ses,
        alpha=0.2, color="steelblue", label="95% CI"
    )
    plt.axhline(bs_price, color="red", linestyle="--", label=f"Black-Scholes: £{bs_price:.4f}")
    plt.xlabel("Number of Paths")
    plt.ylabel("Option Price (£)")
    plt.title("Monte Carlo Convergence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# convergence_plot(S0, K, r, sigma, T)

def monte_carlo_pricer(S0, K, r, sigma, T, n_paths=200_000, seed=42):
    """
    European call option pricer with antithetic sampling and control variates.
    Returns price, standard error, and 95% confidence interval.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    
    #generate the original and antithetic terminal prices separately
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T)
    ST_positive = S0 * np.exp(drift + diffusion * Z)
    ST_negative = S0 * np.exp(drift - diffusion * Z)

    #average each antithetic pair
    payoff_pairs = 0.5 * (
        np.maximum(ST_positive - K, 0)
        + np.maximum(ST_negative - K, 0)
    )
    ST_pairs = 0.5 * (ST_positive + ST_negative)

    #adjust the paired payoffs with the terminal stock-price control variate
    expected_ST = S0 * np.exp(r * T)
    
    cov_est = np.cov(payoff_pairs, ST_pairs)
    beta = cov_est[0, 1] / cov_est[1, 1]
    
    adjusted = payoff_pairs - beta * (ST_pairs - expected_ST)
    
    discount = np.exp(-r * T)
    price = discount * adjusted.mean()
    se = discount * adjusted.std() / np.sqrt(n_paths)
    ci = (price - 1.96 * se, price + 1.96 * se)
    
    return price, se, ci

#run the full-featured Monte Carlo pricer and compare with Black-Scholes
price, se, ci = monte_carlo_pricer(S0=100, K=105, r=0.05, sigma=0.25, T=1.0)
bs = black_scholes_call(100, 105, 0.05, 0.25, 1.0)

print(f"MC Price:       £{price:.4f}")
print(f"Std Error:      £{se:.6f}")
print(f"95% CI:         [£{ci[0]:.4f}, £{ci[1]:.4f}]")
print(f"Black-Scholes:  £{bs:.4f}")
print(f"Absolute Error: £{abs(price - bs):.6f}")