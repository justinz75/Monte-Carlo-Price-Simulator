import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from constants import DEFAULT_N_PATHS, DEFAULT_N_PATHS_LARGE
from main import (black_scholes_call, black_scholes_put, simulate_correlated_portfolio,
                  monte_carlo_var, convergence_plot, call_antithetic, call_control_variate,
                  simulate_gbm, up_and_out_call, asian_call, monte_carlo_pricer,
                  european_call, european_put, implied_volatility,
                  black_scholes_delta, monte_carlo_delta)
from volatility_smile import (drop_arbitrage_violations, implied_forward, out_of_the_money,
                              with_implied_vols)

#folder the charts are saved to for the README
PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

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
plt.savefig(PLOT_DIR / "gbm_paths.png", dpi=150)
plt.show()

#parameters for the European call option
S0, K, r, sigma, T = 100, 110, 0.05, 0.2, 1.0
mc_price, mc_standard_error = european_call(S0, K, r, sigma, T)
print(f"Monte Carlo Price: {mc_price:.4f}, Standard Error: {mc_standard_error:.4f}")

# #parameters for the Black-Scholes price
bs_price = black_scholes_call(S0, K, r, sigma, T)
print(f"Black-Scholes price: £{bs_price:.4f}")
print(f"MC error: £{abs(mc_price - bs_price):.4f}")

#check put-call parity for the Black-Scholes prices
bs_put_price = black_scholes_put(S0, K, r, sigma, T)
parity_rhs = S0 - K * np.exp(-r * T)
bs_parity_error = abs((bs_price - bs_put_price) - parity_rhs)
print(f"Black-Scholes put: £{bs_put_price:.4f}")
print(f"Parity residual (Black-Scholes): £{bs_parity_error:.2e}")
assert bs_parity_error < 1e-8, "Black-Scholes call and put break put-call parity"

#check put-call parity for the Monte Carlo prices, within the sampling error of the terminal price
mc_call_100k, _ = european_call(S0, K, r, sigma, T, n_paths=DEFAULT_N_PATHS_LARGE)
mc_put_100k, mc_put_standard_error = european_put(S0, K, r, sigma, T, n_paths=DEFAULT_N_PATHS_LARGE)
mc_parity_error = abs((mc_call_100k - mc_put_100k) - parity_rhs)
terminal_standard_error = (
    S0 * np.sqrt(np.expm1(sigma**2 * T)) / np.sqrt(DEFAULT_N_PATHS_LARGE)
)
print(f"Monte Carlo put:   £{mc_put_100k:.4f} (Standard Error: {mc_put_standard_error:.4f})")
print(f"Parity residual (Monte Carlo):   £{mc_parity_error:.4f} (tolerance £{5 * terminal_standard_error:.4f})")
assert mc_parity_error < 5 * terminal_standard_error, "MC call and put break put-call parity"

#a dividend yield should give the same price as a spot reduced by the dividends
dividend_yield = 0.02
for pricer in (black_scholes_call, black_scholes_put):
    with_yield = pricer(S0, K, r, sigma, T, q=dividend_yield)
    reduced_spot = pricer(S0 * np.exp(-dividend_yield * T), K, r, sigma, T)
    assert abs(with_yield - reduced_spot) < 1e-10, f"{pricer.__name__} mishandles the dividend yield"

#price options at known volatilities and check implied volatility recovers them
for option_type, pricer in (("call", black_scholes_call), ("put", black_scholes_put)):
    for strike in (80, 100, 120):
        for true_sigma in (0.1, 0.25, 0.6):
            model_price = pricer(100, strike, 0.05, true_sigma, 0.5, q=0.02)
            recovered = implied_volatility(model_price, 100, strike, 0.05, 0.5, q=0.02, option_type=option_type)
            assert abs(recovered - true_sigma) < 1e-6, f"{option_type} K={strike}: wanted {true_sigma}, got {recovered}"
print("Implied volatility: recovered all 18 volatilities to within 1e-6")

#a call priced below its intrinsic value has no implied volatility
assert np.isnan(implied_volatility(1.0, 100, 80, 0.05, 0.5)), "an impossible price should give nan"

#run the smile pipeline on a synthetic chain and check it recovers the forward and skew
spot_synthetic, rate, dividend, expiry_years = 5000.0, 0.04, 0.015, 0.25
strikes = np.arange(4000.0, 5801.0, 50.0)
true_vol_by_strike = {strike: 0.18 - 0.25 * (strike / spot_synthetic - 1) for strike in strikes}
synthetic_chain = pd.DataFrame(
    [{"option_type": "call", "strike": strike,
      "mid": black_scholes_call(spot_synthetic, strike, rate, vol, expiry_years, q=dividend)}
     for strike, vol in true_vol_by_strike.items()]
    + [{"option_type": "put", "strike": strike,
        "mid": black_scholes_put(spot_synthetic, strike, rate, vol, expiry_years, q=dividend)}
       for strike, vol in true_vol_by_strike.items()]
)
true_forward = spot_synthetic * np.exp((rate - dividend) * expiry_years)
recovered_forward = implied_forward(synthetic_chain, np.exp(-rate * expiry_years))
assert abs(recovered_forward - true_forward) < 1e-6, f"forward: wanted {true_forward}, got {recovered_forward}"

implied_dividend = rate - np.log(recovered_forward / spot_synthetic) / expiry_years
synthetic_smile = with_implied_vols(out_of_the_money(synthetic_chain, recovered_forward),
                                    spot_synthetic, rate, expiry_years, implied_dividend)
vol_errors = synthetic_smile["implied_vol"] - synthetic_smile["strike"].map(true_vol_by_strike)
assert vol_errors.abs().max() < 1e-6, "the smile pipeline did not recover the input skew"
print(f"Smile pipeline: recovered the forward and all {len(synthetic_smile)} input vols from prices alone")

#the filter should keep a clean chain and drop a planted stale quote, too cheap or too dear
assert len(drop_arbitrage_violations(synthetic_chain)) == len(synthetic_chain), "the filter dropped good quotes"
stale_row = synthetic_chain.index[(synthetic_chain["option_type"] == "put") & (synthetic_chain["strike"] == 4500.0)][0]
for mispricing in (0.5, 2.0):
    stale_chain = synthetic_chain.copy()
    stale_chain.loc[stale_row, "mid"] *= mispricing
    dropped = set(stale_chain.index) - set(drop_arbitrage_violations(stale_chain).index)
    assert dropped == {stale_row}, f"stale quote priced x{mispricing}: the filter dropped {dropped}"
print("No-arbitrage filter: kept the clean chain whole and caught both planted stale quotes")

#check the Black-Scholes delta against a finite difference of the Black-Scholes price
bump = 0.01
for option_type, pricer in (("call", black_scholes_call), ("put", black_scholes_put)):
    finite_difference = (pricer(S0 + bump, K, r, sigma, T) - pricer(S0 - bump, K, r, sigma, T)) / (2 * bump)
    exact_delta = black_scholes_delta(S0, K, r, sigma, T, option_type=option_type)
    assert abs(finite_difference - exact_delta) < 1e-6, f"{option_type}: formula {exact_delta}, finite difference {finite_difference}"

#call and put deltas should differ by exactly exp(-qT)
delta_gap = (black_scholes_delta(S0, K, r, sigma, T, q=0.02)
             - black_scholes_delta(S0, K, r, sigma, T, q=0.02, option_type="put"))
assert abs(delta_gap - np.exp(-0.02 * T)) < 1e-12, "call and put deltas should differ by exp(-qT)"

#compare the pathwise Monte Carlo delta with the Black-Scholes delta at a few strikes
print("\nDelta, pathwise Monte Carlo against Black-Scholes:")
for option_type in ("call", "put"):
    for strike in (90, 100, 110):
        mc_delta, mc_delta_standard_error = monte_carlo_delta(S0, strike, r, sigma, T, option_type=option_type)
        exact_delta = black_scholes_delta(S0, strike, r, sigma, T, option_type=option_type)
        print(f"  {option_type} K={strike}: {mc_delta:.4f} (Standard Error: {mc_delta_standard_error:.4f}), "
              f"Black-Scholes {exact_delta:.4f}")
        assert abs(mc_delta - exact_delta) < 4 * mc_delta_standard_error, f"{option_type} K={strike}: delta off by over 4 standard errors"

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
plt.savefig(PLOT_DIR / "var_distribution.png", dpi=150)
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

#compare the variance reduction methods using the same number of random draws
#antithetic methods use two draws per path so they get half the paths
n_draws = DEFAULT_N_PATHS_LARGE
standard_price, std_standard_error = european_call(S0, K, r, sigma, T, n_paths=n_draws)
anti_price, anti_standard_error = call_antithetic(S0, K, r, sigma, T, n_paths=n_draws // 2)
cv_price, cv_standard_error = call_control_variate(S0, K, r, sigma, T, n_paths=n_draws)
combined_price, combined_standard_error, combined_ci = monte_carlo_pricer(
    S0, K, r, sigma, T, n_paths=n_draws // 2
)

print(f"\nVariance reduction at {n_draws:,} normal draws (Black-Scholes = £{bs_price:.4f}):")
print(f"{'method':<28}{'price':>10}{'std error':>12}{'vs standard':>14}")
for label, price_estimate, standard_error in [
    ("standard", standard_price, std_standard_error),
    ("antithetic", anti_price, anti_standard_error),
    ("control variate", cv_price, cv_standard_error),
    ("antithetic + control", combined_price, combined_standard_error),
]:
    reduction = f"{1 - standard_error / std_standard_error:>13.1%}" if label != "standard" else f"{'-':>14}"
    print(f"{label:<28}£{price_estimate:>9.4f}{standard_error:>12.5f}{reduction}")
print("\nnote: methods that fit a control variate beta in-sample report a standard")
print("error roughly 10% below the spread actually seen across seeds.")

#convergence plot for European call option pricing
convergence_plot(S0, K, r, sigma, T, save_path=PLOT_DIR / "convergence.png")