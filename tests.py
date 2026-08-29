import numpy as np
import matplotlib.pyplot as plt
from constants import DEFAULT_N_PATHS, DEFAULT_N_PATHS_LARGE
from main import black_scholes_call, simulate_correlated_portfolio, monte_carlo_var, convergence_plot, call_antithetic, call_control_variate, simulate_gbm, up_and_out_call, asian_call, monte_carlo_pricer, european_call

#parameters for GBM simulation
S0 = 100  #initial stock price
mu = 0.05  #5% expected annual return
sigma = 0.2  #20% annual volatility
T = 1.0  #time horizon in years
dt = 1/252  #daily time steps
n_paths = DEFAULT_N_PATHS  #number of simulated paths

paths = simulate_gbm(S0, mu, sigma, T, dt, n_paths)
print(f"Simulated stock price paths shape: {paths.shape}")
print(f"Final stock prices: {paths[-1, :5]}")  #print the final stock prices of the first 5 paths

time_grid = np.linspace(0, T, paths.shape[0])

#visualise first 50 simulated stock price paths
plt.figure(figsize=(10, 6))
plt.plot(time_grid, paths[:, :50], alpha=0.3, linewidth=0.5)
plt.xlabel("Time (years)")
plt.ylabel("Stock Price (£)")
plt.title("Monte Carlo Stock Price Simulation - 50 of 10,000 Paths")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#parameters for the European call option
S0, K, r, sigma, T = 100, 110, 0.05, 0.2, 1.0
mc_price, mc_standard_error = european_call(S0, K, r, sigma, T)
print(f"Monte Carlo Price: {mc_price:.4f}, Standard Error: {mc_standard_error:.4f}")

# #parameters for the Black-Scholes price
bs_price = black_scholes_call(S0, K, r, sigma, T)
print(f"Black-Scholes price: £{bs_price:.4f}")
print(f"MC error: £{abs(mc_price - bs_price):.4f}")

# #parameters for the Asian call option
asian_price, asian_standard_error = asian_call(S0=100, K=105, r=0.05, sigma=0.25, T=1.0)
print(f"Asian call price: £{asian_price:.4f} (Standard Error: {asian_standard_error:.4f})")

#parameters for the up-and-out barrier call option
barrier_price, barrier_standard_error = up_and_out_call(S0=100, K=100, B=130, r=0.05, sigma=0.25, T=1.0)
print(f"Up-and-out call price: £{barrier_price:.4f} (Standard Error: {barrier_standard_error:.4f})")

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
profit_and_losses = terminal_values - initial_investment

var_99 = -np.percentile(profit_and_losses, 1)
cvar_99 = -profit_and_losses[profit_and_losses <= -var_99].mean()

print(f"Portfolio 99% 1-year VaR:  £{var_99:,.2f}")
print(f"Portfolio 99% 1-year CVaR: £{cvar_99:,.2f}")
print(f"Mean return: {profit_and_losses.mean() / initial_investment:.2%}")

#compare standard Monte Carlo and antithetic variates for European call option pricing
standard_price, std_standard_error = european_call(S0, K, r, sigma, T, n_paths=DEFAULT_N_PATHS_LARGE)
anti_price, anti_standard_error = call_antithetic(S0, K, r, sigma, T, n_paths=DEFAULT_N_PATHS)

print(f"Standard MC:  £{standard_price:.4f} (Standard Error: {std_standard_error:.4f})")
print(f"Antithetic:   £{anti_price:.4f} (Standard Error: {anti_standard_error:.4f})")
print(f"Standard error reduction:  {(1 - anti_standard_error/std_standard_error):.1%}")

#compare standard Monte Carlo and control variate for European call option pricing
cv_price, cv_standard_error = call_control_variate(S0, K, r, sigma, T)
print(f"Control var:  £{cv_price:.4f} (Standard Error: {cv_standard_error:.4f})")
print(f"Standard error reduction vs standard: {(1 - cv_standard_error/std_standard_error):.1%}")

#convergence plot for European call option pricing
convergence_plot(S0, K, r, sigma, T)