"""Plot the implied volatility smile from live S&P 500 option prices."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.ticker import PercentFormatter

from main import implied_volatility

TICKER = "^SPX"
CONTRACT_ROOT = "SPXW"  #PM-settled contracts only, monthly expiries also list AM-settled ones
RATE_TICKER = "^IRX"  #13-week US Treasury bill yield, in percent
FALLBACK_RATE = 0.04
NEW_YORK = "America/New_York"
MAX_RELATIVE_SPREAD = 0.25  #drop quotes whose bid-ask spread is over 25% of the mid
EXPIRY_SEARCH_WINDOW = 12  #days either side of the target when picking an expiry
SECONDS_PER_YEAR = 365 * 24 * 60 * 60
PLOT_DIR = Path(__file__).parent / "plots"

#chart colours
PUT_COLOR = "#2a78d6"
CALL_COLOR = "#eb6834"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def fetch_quotes(ticker, expiry):
    """Fetch the calls and puts for one expiry and add a mid price."""
    chain = ticker.option_chain(expiry)
    frames = []
    for option_type, frame in (("call", chain.calls), ("put", chain.puts)):
        frame = frame[frame["contractSymbol"].str.startswith(CONTRACT_ROOT)].copy()
        frame["option_type"] = option_type
        frames.append(frame)
    quotes = pd.concat(frames, ignore_index=True)
    quotes["mid"] = (quotes["bid"] + quotes["ask"]) / 2
    return quotes


def usable(quotes):
    """Keep quotes with both a bid and an ask and a reasonable spread."""
    two_sided = (quotes["bid"] > 0) & (quotes["ask"] > quotes["bid"])
    tight = (quotes["ask"] - quotes["bid"]) <= MAX_RELATIVE_SPREAD * quotes["mid"]
    return quotes[two_sided & tight]


def drop_arbitrage_violations(quotes):
    """Drop stale quotes where a put is cheaper than a lower strike put, or the reverse for calls."""
    def distance_off_line(strikes, prices, j):
        #end quotes only have one neighbour so they are dropped first
        if j in (0, len(prices) - 1):
            return np.inf
        line = np.interp(strikes[j], strikes[[j - 1, j + 1]], prices[[j - 1, j + 1]])
        return abs(prices[j] - line)

    kept = []
    for option_type, side in quotes.groupby("option_type"):
        side = side.sort_values("strike")
        direction = 1 if option_type == "put" else -1  #puts rise with strike, calls fall
        while True:
            strikes = side["strike"].to_numpy()
            prices = direction * side["mid"].to_numpy()
            broken = np.flatnonzero(np.diff(prices) < 0)
            if len(broken) == 0:
                break
            i = broken[0]
            #drop whichever of the two is further from the line through its neighbours
            stale = max((i, i + 1), key=lambda j: distance_off_line(strikes, prices, j))
            side = side.drop(side.index[stale])
        kept.append(side)
    return pd.concat(kept)


def choose_expiry(ticker, target_days, today):
    """Pick the expiry near target_days with the most usable quotes, returned with its quotes."""
    candidates = [expiry for expiry in ticker.options
                  if abs((pd.Timestamp(expiry) - today).days - target_days) <= EXPIRY_SEARCH_WINDOW]
    if not candidates:
        raise SystemExit(f"no {TICKER} expiry within {EXPIRY_SEARCH_WINDOW} days of {target_days} days out")
    fetched = [(expiry, fetch_quotes(ticker, expiry)) for expiry in candidates]
    return max(fetched, key=lambda pair: len(usable(pair[1])))


def overnight_session(t):
    """Check if t is in Cboe's overnight session for SPX options, roughly 20:15 to 09:15 New York."""
    minutes = 60 * t.hour + t.minute
    if minutes >= 20 * 60 + 15:
        return t.dayofweek in (6, 0, 1, 2, 3)  #the session opens Sunday to Thursday evenings
    if minutes < 9 * 60 + 15:
        return t.dayofweek in (0, 1, 2, 3, 4)  #and runs into Monday to Friday mornings
    return False


def risk_free_rate():
    """Latest 13-week T-bill yield as a continuously compounded rate, or a fallback."""
    try:
        closes = yf.Ticker(RATE_TICKER).history(period="5d")["Close"].dropna()
        return float(np.log1p(closes.iloc[-1] / 100))
    except Exception as error:
        print(f"warning: could not fetch {RATE_TICKER} ({error}), using {FALLBACK_RATE:.2%}")
        return FALLBACK_RATE


def implied_forward(quotes, discount, n_strikes=5):
    """Forward price from put-call parity at the strikes where calls and puts are closest in price."""
    calls = quotes[quotes["option_type"] == "call"].set_index("strike")["mid"]
    puts = quotes[quotes["option_type"] == "put"].set_index("strike")["mid"]
    parity_gap = (calls - puts).dropna()
    nearest = parity_gap.abs().nsmallest(n_strikes).index
    forwards = nearest.to_numpy() + parity_gap[nearest].to_numpy() / discount
    return float(np.median(forwards))


def out_of_the_money(quotes, forward):
    """Keep puts struck below the forward and calls struck at or above it."""
    puts = (quotes["option_type"] == "put") & (quotes["strike"] < forward)
    calls = (quotes["option_type"] == "call") & (quotes["strike"] >= forward)
    return quotes[puts | calls]


def with_implied_vols(quotes, spot, r, T, q):
    """Add the Black-Scholes implied volatility of each option."""
    quotes = quotes.copy()
    quotes["implied_vol"] = [
        implied_volatility(row.mid, spot, row.strike, r, T, q, row.option_type)
        for row in quotes.itertuples()
    ]
    return quotes


def call_put_vol_gap(quotes, spot, r, T, q, forward, band=0.03):
    """Median gap in vol points between call and put implied vols at strikes near the forward."""
    near = quotes[(quotes["strike"] / forward - 1).abs() <= band]
    vols = with_implied_vols(near, spot, r, T, q).pivot(
        index="strike", columns="option_type", values="implied_vol")
    return 100 * float((vols["call"] - vols["put"]).abs().median())


def vol_at(smile, moneyness):
    """Interpolate the implied vol at a given moneyness, nan outside the data."""
    x = smile["moneyness"].to_numpy()
    y = smile["implied_vol"].to_numpy()
    if not x.min() <= moneyness <= x.max():
        return np.nan
    return float(np.interp(moneyness, x, y))


def plot_smile(smile, atm_vol, expiry_time, valuation_time, path, overnight=False, show=False):
    """Plot implied vol against strike with the flat Black-Scholes line."""
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
        "xtick.labelcolor": TEXT_MUTED,
        "ytick.labelcolor": TEXT_MUTED,
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)

        for option_type, color, label in (("put", PUT_COLOR, "Out-of-the-money puts"),
                                          ("call", CALL_COLOR, "Out-of-the-money calls")):
            side = smile[smile["option_type"] == option_type]
            #no marker edges as the dense strikes overlap
            ax.scatter(100 * side["moneyness"], 100 * side["implied_vol"], s=14, color=color,
                       linewidths=0, label=label, zorder=3)

        #the at-the-money vol at every strike, which is what Black-Scholes assumes
        ax.axhline(100 * atm_vol, color=TEXT_SECONDARY, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
        ax.annotate(f"Black-Scholes: {atm_vol:.1%} at every strike",
                    xy=(0, 100 * atm_vol), xycoords=("axes fraction", "data"),
                    xytext=(4, 5), textcoords="offset points", ha="left", va="bottom",
                    color=TEXT_SECONDARY, fontsize=9.5)

        ax.axvline(100, color=BASELINE, linewidth=0.8, zorder=1)
        ax.annotate("forward", xy=(100, 0), xycoords=("data", "axes fraction"),
                    xytext=(4, 4), textcoords="offset points", ha="left", va="bottom",
                    color=TEXT_MUTED, fontsize=9)

        #label the put nearest 90% of the forward
        puts = smile[smile["option_type"] == "put"]
        if puts["moneyness"].min() <= 0.90:
            anchor = puts.iloc[(puts["moneyness"] - 0.90).abs().argmin()]
            ax.annotate(f"{anchor['implied_vol']:.1%} at {anchor['moneyness']:.0%} of the forward",
                        xy=(100 * anchor["moneyness"], 100 * anchor["implied_vol"]),
                        xytext=(18, 18), textcoords="offset points", ha="left", va="bottom",
                        color=TEXT_PRIMARY, fontsize=9.5,
                        arrowprops={"arrowstyle": "-", "color": TEXT_MUTED, "linewidth": 0.8})

        days = (expiry_time - valuation_time).total_seconds() / 86400
        ax.text(0, 1.10, "The S&P 500 volatility smile", transform=ax.transAxes,
                fontsize=15, fontweight="semibold", color=TEXT_PRIMARY, ha="left", va="bottom")
        session = ", overnight session" if overnight else ""
        ax.text(0, 1.035, f"SPX options expiring {expiry_time:%d %b %Y} ({days:.0f} days), "
                          f"quoted {valuation_time:%d %b %Y %H:%M} New York{session}",
                transform=ax.transAxes, fontsize=10, color=TEXT_SECONDARY, ha="left", va="bottom")

        ax.set_xlabel("Strike as a percentage of the forward price", color=TEXT_SECONDARY, labelpad=8)
        ax.set_ylabel("Implied volatility", color=TEXT_SECONDARY, labelpad=8)
        ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", color=BASELINE)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.legend(frameon=False, loc="upper right", labelcolor=TEXT_SECONDARY, fontsize=9.5)

        fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot the SPX implied volatility smile from live option prices.")
    parser.add_argument("--expiry", help="expiry date as YYYY-MM-DD (default: the most liquid expiry near --days)")
    parser.add_argument("--days", type=int, default=50, help="target days to expiry (default: 50)")
    parser.add_argument("--show", action="store_true", help="open the chart as well as saving it")
    args = parser.parse_args()

    ticker = yf.Ticker(TICKER)
    if args.expiry:
        expiry, quotes = args.expiry, fetch_quotes(ticker, args.expiry)
    else:
        today = pd.Timestamp.now(tz=NEW_YORK).tz_localize(None).normalize()
        expiry, quotes = choose_expiry(ticker, args.days, today)

    #use the last trade time so weekend runs don't shorten T, except overnight when quotes still move
    fetched_at = pd.Timestamp.now(tz=NEW_YORK)
    overnight = overnight_session(fetched_at)
    valuation_time = fetched_at if overnight else quotes["lastTradeDate"].max().tz_convert(NEW_YORK)
    expiry_time = pd.Timestamp(f"{expiry} 16:00").tz_localize(NEW_YORK)
    T = (expiry_time - valuation_time).total_seconds() / SECONDS_PER_YEAR

    spot = float(ticker.history(period="5d")["Close"].dropna().iloc[-1])
    r = risk_free_rate()
    liquid = usable(quotes)
    #forward from put-call parity so dividends don't need looking up
    forward = implied_forward(liquid, np.exp(-r * T))
    #dividend yield that matches the forward, which also absorbs any move since the index close
    q = r - np.log(forward / spot) / T

    #only filter out of the money quotes, stale in the money ones come in runs and fool the filter
    otm = out_of_the_money(liquid, forward)
    clean = drop_arbitrage_violations(otm)
    smile = with_implied_vols(clean, spot, r, T, q)
    smile = smile.dropna(subset=["implied_vol"])
    smile["moneyness"] = smile["strike"] / forward
    smile = smile.sort_values("moneyness")
    atm_vol = vol_at(smile, 1.0)

    gap = call_put_vol_gap(liquid, spot, r, T, q, forward)
    naive_gap = call_put_vol_gap(liquid, spot, r, T, 0.0, forward)

    n_puts = int((smile["option_type"] == "put").sum())
    n_calls = int((smile["option_type"] == "call").sum())
    print(f"SPX options expiring {expiry}: {T * 365:.1f} days, quoted {valuation_time:%Y-%m-%d %H:%M} New York"
          + (" in the overnight session" if overnight else ""))
    if overnight:
        print("note: SPX options trade overnight while the index is shut, so these quotes have moved on "
              "from the index close. The vols are anchored to the options' own forward, so they still "
              "hold, but for a regular-hours snapshot run this between 09:30 and 20:15 New York.")
    print(f"forward {forward:,.2f} from put-call parity   index close {spot:,.2f}   risk-free rate {r:.2%}")
    print(f"used {n_puts} puts and {n_calls} calls of {len(quotes)} listed contracts "
          f"(dropped one-sided, wide-spread and in-the-money quotes)")
    print(f"dropped {len(otm) - len(clean)} stale quotes that broke the no-arbitrage ordering in strike")
    print(f"call and put vol at the same strike differ by a median {gap:.2f} vol points, "
          f"or {naive_gap:.2f} priced off the index close with no dividends")
    for moneyness in (0.80, 0.90, 0.95, 1.00, 1.05):
        print(f"  implied vol at {moneyness:.0%} of the forward: {vol_at(smile, moneyness):.1%}")

    PLOT_DIR.mkdir(exist_ok=True)
    #include the time so a new snapshot doesn't overwrite an old one
    path = PLOT_DIR / f"spx_smile_{valuation_time:%Y-%m-%d_%H%M}_{expiry}.png"
    columns = ["option_type", "strike", "moneyness", "bid", "ask", "mid", "implied_vol",
               "volume", "openInterest", "lastTradeDate"]
    smile[columns].to_csv(path.with_suffix(".csv"), index=False)
    plot_smile(smile, atm_vol, expiry_time, valuation_time, path, overnight=overnight, show=args.show)
    print(f"saved {path.relative_to(Path(__file__).parent)} and its data as .csv")


if __name__ == "__main__":
    main()
