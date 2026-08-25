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
returns = rng.normal(loc=mu, scale=sigma, size=10_000)
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

#parameters for the GBM simulation
S0 = 100  #initial stock price
mu = 0.05  #5% expected annual return
sigma = 0.2  #20% annual volatility
T = 1.0  #time horizon in years
dt = 1/252  #daily time steps
n_paths = 1000  #number of simulated paths

paths = simulate_gbm(S0, mu, sigma, T, dt, n_paths)
print(f"Simulated stock price paths shape: {paths.shape}")
print(f"Final stock prices: {paths[-1, :5]}")  #print the final stock prices of the first 5 paths

time_grid = np.linspace(0, T, paths.shape[0])

#visualise the first 50 simulated stock price paths
plt.figure(figsize=(10, 6))
plt.plot(time_grid, paths[:, :50], alpha=0.3, linewidth=0.5)
plt.xlabel("Time (years)")
plt.ylabel("Stock Price (£)")
plt.title("Monte Carlo Stock Price Simulation - 50 of 10,000 Paths")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


def price_european_call(S0, K, r, sigma, T, n_paths=10_000, seed=50):
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
S0, K, r, sigma, T = 100, 110, 0.05, 0.2, 1.0
mc_price, mc_std_error = price_european_call(S0, K, r, sigma, T)
print(f"Monte Carlo Price: {mc_price:.4f}, Standard Error: {mc_std_error:.4f}")


def black_scholes_call(S0, K, r, sigma, T):
    """Exact Black-Scholes price for a European call."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)

#parameters for the Black-Scholes price
bs_price = black_scholes_call(S0, K, r, sigma, T)
print(f"Black-Scholes price: £{bs_price:.4f}")
print(f"MC error: £{abs(mc_price - bs_price):.4f}")


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
    price = np.exp(-r * T) * payoffs.mean()
    se = np.exp(-r * T) * payoffs.std() / np.sqrt(n_paths)
    
    return price, se

#parameters for the Asian call option
asian_price, asian_se = asian_call(S0=100, K=105, r=0.05, sigma=0.25, T=1.0)
print(f"Asian call price: £{asian_price:.4f} (SE: {asian_se:.4f})")

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
    payoffs = np.maximum(ST - K, 0)
    payoffs[knocked_out] = 0.0
    
    #discounted expected payoff and standard error
    price = np.exp(-r * T) * payoffs.mean()
    se = np.exp(-r * T) * payoffs.std() / np.sqrt(n_paths)
    
    return price, se

#parameters for the up-and-out barrier call option
barrier_price, barrier_se = up_and_out_call(S0=100, K=100, B=130, r=0.05, sigma=0.25, T=1.0)
print(f"Up-and-out call price: £{barrier_price:.4f} (SE: {barrier_se:.4f})")

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

#parameters for the VaR calculation
var, cvar, profit_and_losses = monte_carlo_var(S0=100, mu=0.05, sigma=0.25, T=1.0)
print(f"Value at Risk (VaR): £{var:.4f}")
print(f"Conditional Value at Risk (CVaR): £{cvar:.4f}")

#visualise the profit and loss distribution
plt.figure(figsize=(10, 6))
plt.hist(profit_and_losses, bins=200, density=True, alpha=0.7, color="steelblue", edgecolor="none")
plt.axvline(-var, color="red", linestyle="--", linewidth=2, label=f"95% VaR: £{var:,.0f}")
plt.axvline(-cvar, color="darkred", linestyle="--", linewidth=2, label=f"95% CVaR: £{cvar:,.0f}")
plt.xlabel("Profit / Loss (£)")
plt.ylabel("Density")
plt.title("Monte Carlo P&L Distribution with VaR and CVaR")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()