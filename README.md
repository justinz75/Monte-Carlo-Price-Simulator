# Monte Carlo Price Simulator

Monte Carlo option pricing and portfolio risk in NumPy, benchmarked against closed-form
Black-Scholes. Covers European, Asian and barrier options, correlated multi-asset
portfolios, VaR/CVaR, and two variance reduction techniques.

## Running it

```bash
pip install -r requirements.txt
python main.py     # prices one European call, with a 95% confidence interval
python tests.py    # full demo: every pricer, the variance reduction table, and plots
```

`tests.py` opens matplotlib windows. To run it without them:

```bash
MPLBACKEND=Agg python tests.py
```

## What's implemented

| Area | Functions |
| --- | --- |
| Path simulation | `simulate_gbm`, `simulate_correlated_portfolio` (Cholesky-correlated assets) |
| European options | `european_call`, `european_put`, `european_call_batched`, `european_call_parallel` |
| Closed form | `black_scholes_call`, `black_scholes_put` |
| Path-dependent | `asian_call`, `asian_call_low_memory`, `up_and_out_call` |
| Risk | `monte_carlo_var` (VaR and CVaR) |
| Variance reduction | `call_antithetic`, `call_control_variate`, `monte_carlo_pricer` |
| Diagnostics | `convergence_plot` |

Paths are simulated in `float32` to halve memory traffic, but every sum, mean and
variance accumulates in `float64` — `float32` accumulators lose meaningful precision
over hundreds of thousands of paths.

## Results

### Variance reduction

European call, `S0=100, K=110, r=0.05, sigma=0.2, T=1.0`. Black-Scholes gives **£6.0401**.

All four methods get the same budget of 100,000 standard normal draws. This matters:
antithetic sampling consumes two draws per path, so it is given half the path count.
Comparing 100k standard paths against 10k antithetic *pairs* is a 5x budget difference
and makes antithetic look about 90% worse than it actually is.

| method | price | std error | vs standard |
| --- | ---: | ---: | ---: |
| standard | £6.0906 | 0.03692 | — |
| antithetic | £6.0283 | 0.03137 | 15.0% |
| control variate | £6.0450 | 0.01940 | 47.5% |
| antithetic + control | £6.0433 | 0.00704 | **80.9%** |

Antithetic sampling alone is weak here because the call payoff is convex and this strike
is out of the money — one leg of most pairs contributes nothing. The control variate
(terminal stock price, whose risk-neutral expectation `S0·exp(rT)` is known exactly)
does most of the work, and the two compose well.

### Barrier monitoring frequency

`up_and_out_call` checks the barrier at `n_steps` discrete dates, so the price is a
function of the monitoring frequency. Up-and-out call, `S0=100, K=100, B=130,
r=0.05, sigma=0.25, T=1.0`, 200,000 paths:

| n_steps | price |
| ---: | ---: |
| 50 | 2.7304 |
| 252 | 2.4467 |
| 1008 | 2.3296 |
| 4032 | 2.3020 |

That is a 19% swing, roughly 35 standard errors — discretization bias, not noise. More
monitoring dates means more chances to knock out, so the price falls. Daily monitoring
(252) is a real contract specification, but it is *not* an approximation to the
continuously monitored price, and this function does not claim to be one.

## Known limitations

- **The barrier is discretely monitored.** See the table above. Converting to a
  continuous-barrier price needs a continuity correction (Broadie-Glasserman-Kou).
- **Control variate betas are fitted in-sample.** `call_control_variate` and
  `monte_carlo_pricer` estimate beta from the same paths they price with. Measured over
  40 seeds at 100,000 paths, `monte_carlo_pricer` reports a standard error about 10%
  below the spread actually observed (0.00497 reported vs 0.00558 realised). The prices
  themselves show no detectable bias. Fitting beta on a held-out pilot batch would fix it.
- **GBM only** — constant volatility, no jumps, no stochastic vol.
- **`european_call_parallel` is unbenchmarked** and seeds workers with `seed + worker_id`,
  which does not guarantee independent streams. `np.random.SeedSequence(seed).spawn(n)`
  is the correct approach. It also ships full payoff arrays back through pickle rather
  than partial sums.

## Roadmap

- Quasi-Monte Carlo with scrambled Sobol sequences. Measured on this pricer, Sobol is
  ~320x more accurate than pseudorandom sampling at 65,536 paths, and the gap widens
  with path count — a far bigger win than any of the current optimisations.
- A pytest suite replacing the demo script: MC within 3 standard errors of Black-Scholes,
  put-call parity, geometric Asian against its closed form, barrier ≤ vanilla.
- Greeks: pathwise delta, and gamma by finite differences with common random numbers.
- Collapse the six near-duplicate terminal-price samplers into one engine taking a
  payoff function, so variance reduction and QMC apply to every product.

## Correctness checks

`tests.py` asserts put-call parity in two places:

- **Black-Scholes**: `C - P = S0 - K·exp(-rT)` holds to ~1e-15.
- **Monte Carlo**: the call and put share a seed, so the only parity error is the
  sampling error in the mean terminal price. The tolerance is derived from that
  quantity rather than picked by hand.
