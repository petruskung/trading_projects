import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import integrate
import yfinance as yf

### Line 98 is crucial for MTM value ###
st.title("Advanced Options Pricing Dashboard")

# ------------------------------------------------------
# Section 1: To input different option strategies
# ------------------------------------------------------

strategies = [
    "Single Call", "Single Put", 
    "Call Spread", "Put Spread", "Straddle", "Strangle",
    "Iron Condor", "Butterfly Call", "Butterfly Put"
]
strategy = st.selectbox("Select Option Strategy", strategies)

# This just shows # of legs for each strategy
leg_config = {
    "Single Call": 1, "Single Put": 1,
    "Call Spread": 2, "Put Spread": 2, 
    "Straddle": 1, "Strangle": 2,
    "Iron Condor": 4, "Butterfly Call": 3, "Butterfly Put": 3
}
num_legs = leg_config[strategy]
option_data = []

# Lets the user input the strikes for each leg
for i in range(num_legs):
    col1, col2 = st.columns(2)
    with col1:
        strike = st.number_input(f"Strike {i+1}", min_value=0.0, value=500.0 + i*5, key=f"strike_{i}")
    option_data.append({"strike": strike})
    
st.subheader("Underlying & IV Source")

# 1) User chooses ticker
ticker = st.text_input("Ticker (e.g., AAPL, SPY, TSLA)", value="AAPL").strip().upper()

# 2) The user chooses how we get IV from the nearest expiry:
#  (A) "ATM option": pick the strike closest to spot and use its IV
#  (B) "Highest-volume option": pick the most traded option and use its IV
iv_mode = st.radio(
    "Choose IV from nearest expiry",
    ["ATM option", "Highest-volume option"],
    index=0,
    help="Both choices look at the nearest listed expiry."
)

@st.cache_data(ttl=60)
# Cache for 60s to avoid repeated network calls on every rerun
def fetch_spot_and_iv(tkr: str, mode: str):
    import yfinance as yf
    t = yf.Ticker(tkr)
    s0 = None
    try:
        fi = getattr(t, "fast_info", {}) or {}
        s0 = fi.get("last_price", None)
    except Exception:
        pass
    if s0 is None:
        try:
            info = t.info or {}
            s0 = info.get("regularMarketPrice", None)
        except Exception:
            pass
    if s0 is None:
        # if could not get a price for this ticker
        return None, None, None, None
    s0 = float(s0)

    # Nearest expiry
    try:
        exps = t.options
    except Exception:
        exps = []
    if not exps:
        return s0, None, None, None
    expiry = exps[0]

    # Option chain
    try:
        chain = t.option_chain(expiry)
    except Exception:
        return s0, None, expiry, None

    base = chain.calls.copy()
    if base is None or base.empty:
        base = chain.puts.copy()
    if base is None or base.empty:
        return s0, None, expiry, None

    base = base.dropna(subset=["impliedVolatility"])
    if base.empty:
        return s0, None, expiry, None

    if mode == "ATM option":
        base["atm_dist"] = (base["strike"] - s0).abs()
        row = base.sort_values("atm_dist").iloc[0]
    else:  
        if "volume" in base:
            base["volume"] = base["volume"].fillna(0)
            row = base.sort_values("volume", ascending=False).iloc[0]
        else:
            base["atm_dist"] = (base["strike"] - s0).abs()
            row = base.sort_values("atm_dist").iloc[0]

    sigma = float(row["impliedVolatility"])   # σ as decimal (e.g., 0.24)
    chosen_strike = float(row["strike"])
    return s0, sigma, expiry, chosen_strike

# Fallback just in case --> Of course we can adjust later
if "s0" not in st.session_state:
    st.session_state.s0 = 500.0
if "iv" not in st.session_state:
    st.session_state.iv = 0.20

# Fetch from yfinance
S0_live, sigma_live, expiry_live, strike_for_iv = (None, None, None, None)
if ticker:
    S0_live, sigma_live, expiry_live, strike_for_iv = fetch_spot_and_iv(ticker, iv_mode)

# Update session state if we got live values
if S0_live is not None:
    st.session_state.s0 = float(S0_live)
if sigma_live is not None:
    st.session_state.iv = float(sigma_live)

# Variables used everywhere below
S0 = float(st.session_state.s0)         
sigma_input = float(st.session_state.iv) 

# Display what we're using
st.info(
    f"**Live Spot (S₀):** ${S0:,.2f}  \n"
    f"**IV source:** {iv_mode}  ·  "
    f"**Nearest expiry:** {expiry_live if expiry_live else 'N/A'}  ·  "
    f"**Chosen option strike:** {f'${strike_for_iv:,.2f}' if strike_for_iv else 'N/A'}  \n"
    f"**Implied Vol (σ):** {sigma_input:.2%}"
)

## This rf rate is extracted from the 13-week Treasury Yield Curve from YFinance, and computes the 5-day average
st.subheader("Market Parameters")
tbill = yf.Ticker("^IRX")
hist = tbill.history(period="5d")
risk_free_rate = hist['Close'].mean() / 100
st.metric("Live Risk-Free Rate", f"{risk_free_rate:.4%}", "5-Day Avg of ^IRX")

# Slider to choose TTM --> min = expiry; max = 2 years (just adjust "2.0" to any time you like)
T = st.slider("Time-to-Maturity (Years)", 0.0, 2.0, 0.25, key="time_slider")

# ------------------------------------------------------
# Section 2: Code for Heston Model --> only need to adjust for stochastic vol
# ------------------------------------------------------
# IMPORTANT: This function expects v0 = variance (σ^2), not volatility σ.

def heston_price(S0, K, T, r, v0, theta, kappa, sigma_v, rho_sv, option_type="call"):
    T_eff = max(float(T), 1e-8)  # guard near expiry
    if S0 <= 0 or K <= 0:
        return 0.0
    x0 = np.log(S0)
    def char_func(u):
        iu = 1j * u
        a = kappa * theta
        b = kappa
        d = np.sqrt((rho_sv * sigma_v * iu - b)**2 + sigma_v**2 * (iu + u**2))
        g = (b - rho_sv * sigma_v * iu - d) / (b - rho_sv * sigma_v * iu + d)
        # log branch-safe
        log_term = np.log((1 - g * np.exp(-d * T_eff)) / (1 - g))
        C = iu * (x0 + r * T_eff) + (a / (sigma_v**2)) * ((b - rho_sv * sigma_v * iu - d) * T_eff - 2.0 * log_term)
        D = ((b - rho_sv * sigma_v * iu - d) / (sigma_v**2)) * ((1 - np.exp(-d * T_eff)) / (1 - g * np.exp(-d * T_eff)))
        return np.exp(C + D * v0)
    # Normalization for P1
    phi_minus_i = char_func(-1j)
    def integrand_P1(u):
        return np.real(np.exp(-1j * u * np.log(K)) * char_func(u - 1j) / (1j * u * phi_minus_i))
    def integrand_P2(u):
        return np.real(np.exp(-1j * u * np.log(K)) * char_func(u) / (1j * u))
    upper = 150.0
    P1 = 0.5 + (1 / np.pi) * integrate.quad(integrand_P1, 0.0, upper, limit=200)[0]
    P2 = 0.5 + (1 / np.pi) * integrate.quad(integrand_P2, 0.0, upper, limit=200)[0]
    call_price = S0 * P1 - K * np.exp(-r * T_eff) * P2
    call_price = float(np.real(call_price))
    if option_type == "call":
        return call_price
    else:
        # put via parity
        return call_price - S0 + K * np.exp(-r * T_eff)

def calculate_strategy_price(S, v0, strategy, strikes, T, r):
    # note: v0 is variance (σ^2). Keep names consistent.
    args = dict(T=T, r=r, v0=v0, theta=0.04, kappa=1.0, sigma_v=0.2, rho_sv=-0.7)
    if strategy == "Single Call":
        return heston_price(S, strikes[0], option_type="call", **args)
    elif strategy == "Single Put":
        return heston_price(S, strikes[0], option_type="put", **args)
    elif strategy == "Call Spread":
        return heston_price(S, strikes[0], option_type="call", **args) - heston_price(S, strikes[1], option_type="call", **args)
    elif strategy == "Put Spread":
        return heston_price(S, strikes[1], option_type="put", **args) - heston_price(S, strikes[0], option_type="put", **args)
    elif strategy == "Straddle":
        return heston_price(S, strikes[0], option_type="call", **args) + heston_price(S, strikes[0], option_type="put", **args)
    elif strategy == "Strangle":
        return heston_price(S, strikes[1], option_type="call", **args) + heston_price(S, strikes[0], option_type="put", **args)
    elif strategy == "Iron Condor":
        put_spread = heston_price(S, strikes[1], option_type="put", **args) - heston_price(S, strikes[0], option_type="put", **args)
        call_spread = heston_price(S, strikes[3], option_type="call", **args) - heston_price(S, strikes[2], option_type="call", **args)
        return put_spread + call_spread
    elif "Butterfly" in strategy:
        typ = "call" if "Call" in strategy else "put"
        return (heston_price(S, strikes[0], option_type=typ, **args)
                - 2*heston_price(S, strikes[1], option_type=typ, **args)
                + heston_price(S, strikes[2], option_type=typ, **args))

# ------------------------------------------------------
# Section 3: For the MTM value 
# ------------------------------------------------------

st.subheader("Current Mark-to-Market Value")

live_spot_price = S0                 # Spot from the ticker
live_sigma = sigma_input             # σ from the selected option of nearest expiry
initial_variance = live_sigma ** 2   # v0 for Heston 

st.info(f"""
Calculating MTM using Heston model with these live parameters:
- **Live Spot Price (S0):** ${live_spot_price:,.2f}
- **Initial Variance (v0):** {initial_variance:.4f} (from selected option IV of {live_sigma:.2%})
""")

current_value = calculate_strategy_price(
    live_spot_price,                  
    initial_variance,               
    strategy,
    [leg["strike"] for leg in option_data],
    T,
    risk_free_rate
)

st.metric("Strategy MTM Value", f"${current_value:,.2f}")

# ------------------------------------------------------
# Section 4: Heatmap with $0.5 Stock Price increments 
# ------------------------------------------------------

st.subheader("Price Sensitivity Heatmap")
stock_range = np.arange(S0-40, S0+40, 0.5)  ## If you want larger increments --> just adjust "0.5" to "1.0" or anything
iv_range = np.linspace(0.01, 0.6, 20)
prices = np.zeros((len(iv_range), len(stock_range)))

for i, iv in enumerate(iv_range):
    for j, S in enumerate(stock_range):
        prices[i,j] = calculate_strategy_price(
            S, iv, strategy, 
            [leg["strike"] for leg in option_data],
            T, risk_free_rate)

heatmap = go.Heatmap(
    z=prices,
    x=stock_range,
    y=iv_range,
    colorscale="Viridis",
    hoverongaps=False,
    hovertext=[[f"Stock: ${S:.0f}<br>IV: {iv:.1%}<br>Value: ${val:.2f}" 
               for S, val in zip(stock_range, row)] 
               for iv, row in zip(iv_range, prices)]
)
fig = go.Figure(data=[heatmap])
fig.update_layout(
    title=f"{strategy} Strategy Value",
    xaxis_title="Underlying Price ($0.5 increments)",
    yaxis_title="Implied Volatility"
)
st.plotly_chart(fig)

live_volatility = st.session_state.iv 
initial_variance = live_volatility ** 2

# ------------------------------------------------------
## Section 5: Show Payoff & PnL section
# ------------------------------------------------------

st.subheader("Payoff Diagram & P&L Curves")

# 1) Create a space that has a grid of 200 stock prices
spot_grid = np.linspace(S0 * 0.5, S0 * 1.5, 200)

# 2) Payoff at expiry
def payoff_at_expiry(S, strategy, strikes):
    K = strikes
    if strategy == "Single Call":
        return np.maximum(S - K[0], 0)
    if strategy == "Single Put":
        return np.maximum(K[0] - S, 0)
    if strategy == "Call Spread":
        return np.maximum(S - K[0], 0) - np.maximum(S - K[1], 0)
    if strategy == "Put Spread":
        return np.maximum(K[1] - S, 0) - np.maximum(K[0] - S, 0)
    if strategy == "Straddle":
        return np.abs(S - K[0])
    if strategy == "Strangle":
        return np.maximum(S - K[1], 0) + np.maximum(K[0] - S, 0)
    if strategy == "Iron Condor":
        return (np.maximum(K[1] - S, 0) - np.maximum(K[0] - S, 0) +
                np.maximum(S - K[2], 0) - np.maximum(S - K[3], 0))
    if "Butterfly" in strategy:
        if "Call" in strategy:
            return (np.maximum(S - K[0], 0) -
                    2*np.maximum(S - K[1], 0) +
                    np.maximum(S - K[2], 0))
        else:
            return (np.maximum(K[0] - S, 0) -
                    2*np.maximum(K[1] - S, 0) +
                    np.maximum(K[2] - S, 0))
    return np.zeros_like(S)

K_list = [leg["strike"] for leg in option_data]
payoffs = payoff_at_expiry(spot_grid, strategy, K_list)

# 3) Defining the time-points for P&L curves
taus = {
    f"Current (T={T:.2f}y)":      T,                                  
    "1 Month to Expiry":          max(T - 1/12, 0.0),                
    "3 Months to Expiry":         max(T - 3/12, 0.0),                
    "6 Months to Expiry":         max(T - 6/12, 0.0),                
    "1 year to Expiry":           max(T - 1,    0.0),               
    "Expiry (T=0)":               0.0                               
}

# 4) Base premium at selected TTM
base_premium = calculate_strategy_price(
    S0, initial_variance, strategy, K_list, T, risk_free_rate
)

# 5) Build P&L curves 
pnl_curves = {}
for label, tau_rem in taus.items():
    if tau_rem == 0:
        pnl_curves[label] = payoffs - base_premium
    else:
        vals = np.array([
            calculate_strategy_price(
                S, initial_variance, strategy, K_list, tau_rem, risk_free_rate
            ) for S in spot_grid
        ])
        pnl_curves[label] = vals - base_premium

# 6) Plot payoff + P&L
fig = go.Figure()

# payoff at expiry --> Used "black" and "dash" to make it more distinguishable
fig.add_trace(go.Scatter(
    x=spot_grid, y=payoffs,
    mode="lines", name="Payoff at Expiry",
    line=dict(color="black", dash="dash")
))

# plot P&L curves for each tau --> Each has diff color --> Had to ask chatgpt for the names of these colors 
colors = ["firebrick","royalblue","forestgreen","goldenrod"]
for (label, pnl), color in zip(pnl_curves.items(), colors):
    fig.add_trace(go.Scatter(
        x=spot_grid, y=pnl,
        mode="lines", name=f"P&L ({label})",
        line=dict(color=color)
    ))
    
## One vital thing to note: Note we have 3M, 6M expiries. Suppose your TTM is 2M to expiry --> 3M and 6M lines will still appear, just in wiggly lines (can ignore them)
fig.update_layout(
    title="Strategy Payoff & P&L Curves",
    xaxis_title="Spot Price",
    yaxis_title="Payoff / P&L",
    legend_title="Curves",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------
## Section 6: Show Greeks Calculations
# ------------------------------------------------------

st.subheader("Strategy Greeks")

def compute_strategy_greeks(S0, iv, strategy, strikes, T, r):
    h = 0.01  # Perturbation size
    base_price = calculate_strategy_price(S0, iv, strategy, strikes, T, r)
    ## Note, I used Central Differences Method (within FDM) to calculate them

    price_up = calculate_strategy_price(S0 + h, iv, strategy, strikes, T, r)
    price_down = calculate_strategy_price(S0 - h, iv, strategy, strikes, T, r)
    delta = (price_up - price_down) / (2 * h)
    
    gamma = (price_up - 2 * base_price + price_down) / (h ** 2)
    
    vega_up = calculate_strategy_price(S0, iv + h, strategy, strikes, T, r)
    vega_down = calculate_strategy_price(S0, iv - h, strategy, strikes, T, r)
    vega = (vega_up - vega_down) / (2 * h)
    
    theta = (calculate_strategy_price(S0, iv, strategy, strikes, T + h, r) - base_price) / h
    
    volga = (vega_up - 2 * base_price + vega_down) / (h ** 2)
    
    vanna = (calculate_strategy_price(S0 + h, iv + h, strategy, strikes, T, r) -
            calculate_strategy_price(S0 + h, iv - h, strategy, strikes, T, r) -
            calculate_strategy_price(S0 - h, iv + h, strategy, strikes, T, r) +
            calculate_strategy_price(S0 - h, iv - h, strategy, strikes, T, r)) / (4 * h ** 2)
    
    return {
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega,
        "Theta": theta,
        "Volga": volga,
        "Vanna": vanna
    }

# Calculate and display Greeks
greeks = compute_strategy_greeks(
    S0, initial_variance, strategy, 
    [leg["strike"] for leg in option_data], 
    T, risk_free_rate
)

# Create columns for better display
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Delta", f"{greeks['Delta']:.4f}")
    st.metric("Gamma", f"{greeks['Gamma']:.4f}")
with col2:
    st.metric("Vega", f"{greeks['Vega']:.4f}")
    st.metric("Theta", f"{greeks['Theta']:.4f}")
with col3:
    st.metric("Volga", f"{greeks['Volga']:.4f}")
    st.metric("Vanna", f"{greeks['Vanna']:.4f}")

# Display P&L decomposition formula with current Greeks
st.latex(r'''
\delta V \approx \Theta \delta t + \Delta \delta S + v \delta\sigma + \frac{1}{2} \Gamma (\delta S)^2 + \frac{1}{2} Volga (\delta\sigma)^2 + Vanna (\delta S \delta\sigma)
''')

st.markdown("**Using Computed Greeks:**")
st.write(f"""
- Θ (Theta) = {greeks['Theta']:.4f} per year
- Δ (Delta) = {greeks['Delta']:.4f} per $1
- v (Vega) = {greeks['Vega']:.4f} per 1% IV change
- Γ (Gamma) = {greeks['Gamma']:.4f} per $1²
- Volga = {greeks['Volga']:.4f} per 1% IV²
- Vanna = {greeks['Vanna']:.4f} per $1·1% IV
""")

# ------------------------------------------------------
# Section 7: Interactive P&L Calculator
# ------------------------------------------------------

with st.expander("Calculate Hypothetical P&L"):
    col1, col2, col3 = st.columns(3)
    with col1:
        delta_S = st.number_input("ΔS ($ change)", value=0.0)
    with col2:
        delta_sigma = st.number_input("Δσ (IV change %)", value=0.0) / 100
    with col3:
        delta_t = st.number_input("Δt (years)", value=0.0)
    
    pnl = (greeks['Theta'] * delta_t +
          greeks['Delta'] * delta_S +
          greeks['Vega'] * delta_sigma +
          0.5 * greeks['Gamma'] * delta_S**2 +
          0.5 * greeks['Volga'] * delta_sigma**2 +
          greeks['Vanna'] * delta_S * delta_sigma)
    
    st.metric("Estimated P&L", f"${pnl:.2f}", delta_color="off")

# ------------------------------------------------------
# Section 8: Strategy Value Surface 
# ------------------------------------------------------

st.subheader("Strategy Value Surface")

# Build axes --> ttm, underlying which becomes surface panel
stock_range = np.arange(S0 * 0.7, S0 * 1.3, S0 / 100) # Dynamic range
times = np.linspace(0.01, T if T > 0 else 1.0, 20)
surface = np.zeros((len(times), len(stock_range)))
strikes = [leg["strike"] for leg in option_data]

# Generate surface data using the live initial_variance
for i, t in enumerate(times):
    for j, S in enumerate(stock_range):
        surface[i, j] = calculate_strategy_price(
            S, initial_variance, strategy, strikes, t, risk_free_rate
        )

# Create the surface plot
surface_trace = go.Surface(
    z=surface,
    x=stock_range,
    y=times,
    colorscale="Viridis",
    hovertemplate=(
        "Stock Price: %{x:,.0f}<br>"
        "TTM: %{y:.2f} yrs<br>"
        "Strategy Value: $%{z:,.2f}<extra></extra>"
    )
)
fig_surface = go.Figure(data=[surface_trace])

fig_surface.update_layout(
    title=f"{strategy} Value Surface",
    scene=dict(
        xaxis_title="Underlying Price ($)",
        yaxis_title="Time to Maturity (yrs)",
        zaxis_title="Strategy Value ($)",
    ),
    height=600
)

st.plotly_chart(fig_surface)

# ------------------------------------------------------
# Section 9: Building vol skew and vol term structure
# ------------------------------------------------------

# build moneyness and TTM axes
moneyness = stock_range / S0
times = np.linspace(0.1, 1.0, surface.shape[0])

# here, i am basically finding the TTM row, and the strike column in the IV surface I computed
ttm_idx = np.abs(times - T).argmin()
atm_idx = np.abs(stock_range - S0).argmin()

# slice out the two curves
iv_vs_k = surface[ttm_idx, :]        # IV slice at time T
iv_vs_t = surface[:, atm_idx]      # IV slice at S = S0

st.subheader("Volatility Slices")

# IV vs Moneyness --> IV Vol Skew plot
fig_k = go.Figure(go.Scatter(
    x=moneyness,
    y=iv_vs_k,
    mode="lines",
    line=dict(color="red"),
    name=f"T = {times[ttm_idx]:.2f} yrs"
))
fig_k.update_layout(
    title="IV vs Moneyness",
    xaxis_title="Moneyness (K / S₀)",
    yaxis_title="Implied Volatility",
    showlegend=False
)
st.plotly_chart(fig_k)

# IV vs TTM --> IV Term Structure plot
fig_t = go.Figure(go.Scatter(
    x=times,
    y=iv_vs_t,
    mode="lines",
    line=dict(color="red"),
    name=f"K ≈ {stock_range[atm_idx]:.0f}"
))
fig_t.update_layout(
    title="IV vs Time to Maturity",
    xaxis_title="Time to Maturity (yrs)",
    yaxis_title="Implied Volatility",
    showlegend=False
)
st.plotly_chart(fig_t)
