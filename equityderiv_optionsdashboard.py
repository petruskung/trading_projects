import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import integrate
import yfinance as yf

### Line 98 is crucial for MTM value ###
st.title("Advanced Options Pricing Dashboard")
# ------------------------------------------------------
# Section 1: To input different option strategies
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

# Code to enter requested strike, underlying price, rf rate
for i in range(num_legs):
    col1, col2 = st.columns(2)
    with col1:
        strike = st.number_input(f"Strike {i+1}", min_value=0.0, value=500.0 + i*5, key=f"strike_{i}")
    option_data.append({"strike": strike})


S0 = st.number_input("Underlying Price", min_value=0.0, value=500.0, key="underlying_price")

## This rf rate is extracted from the 13-week Treasury Yield Curve from YFinance 
# In case it could not be extracted, the function will fall back to a default rate of 3%
tbill = yf.Ticker("^IRX")
hist = tbill.history(period="5d")
risk_free_rate = hist["Close"].iloc[-1]/100 if not hist.empty else 0.03
st.write(f"Risk-Free Rate: {risk_free_rate:.4f}")

# Slider to choose TTM --> min = expiry; max = 2 years (just adjust "2.0" to any time you like)
T = st.slider("Time-to-Maturity (Years)", 0.0, 2.0, 0.25, key="time_slider")

# ------------------------------------------------------
# Section 2: Code for Heston Model --> only need to adjust for stochastic vol
def heston_price(S0, K, T, r, v0, theta, kappa, sigma, rho, option_type="call"):
    def char_func(u, t, v0, theta, kappa, sigma, rho):
        xi = kappa - sigma * rho * 1j * u
        d = np.sqrt(xi**2 + sigma**2 * (u**2 + 1j * u))
        g = (xi - d) / (xi + d)
        C = r * 1j * u * t + (kappa * theta / sigma**2) * ((xi - d) * t - 2 * np.log((1 - g * np.exp(-d * t)) / (1 - g)))
        D = (xi - d) / sigma**2 * (1 - np.exp(-d * t)) / (1 - g * np.exp(-d * t))
        return np.exp(C + D * v0 + 1j * u * np.log(S0))

# These are norm functions
    P1 = 0.5 + (1 / np.pi) * integrate.quad(lambda u: np.real(np.exp(-1j * u * np.log(K)) * char_func(u - 1j, T, v0, theta, kappa, sigma, rho) / (1j * u * S0)), 0, 100)[0]
    P2 = 0.5 + (1 / np.pi) * integrate.quad(lambda u: np.real(np.exp(-1j * u * np.log(K)) * char_func(u, T, v0, theta, kappa, sigma, rho) / (1j * u)), 0, 100)[0]
    
    call_price = S0 * P1 - K * np.exp(-r * T) * P2
    return call_price if option_type == "call" else call_price - S0 + K * np.exp(-r * T)

# Compute the mechanism for specific strategies
def calculate_strategy_price(S, iv, strategy, strikes, T, r):
    if strategy == "Single Call":
        return heston_price(S, strikes[0], T, r, iv, 0.04, 1.0, 0.2, -0.7, "call")
    elif strategy == "Single Put":
        return heston_price(S, strikes[0], T, r, iv, 0.04, 1.0, 0.2, -0.7, "put")
    elif strategy == "Call Spread":
        return heston_price(S, strikes[0], T, r, iv, 0.04, 1.0, 0.2, -0.7, "call") - heston_price(S, strikes[1], T, r, iv, 0.04, 1.0, 0.2, -0.7, "call")
    elif strategy == "Put Spread":
        return heston_price(S, strikes[1], T, r, iv, 0.04, 1.0, 0.2, -0.7, "put") - heston_price(S, strikes[0], T, r, iv, 0.04, 1.0, 0.2, -0.7, "put")
    elif strategy == "Straddle":
        return heston_price(S, strikes[0], T, r, iv, 0.04, 1.0, 0.2, -0.7, "call") + heston_price(S, strikes[0], T, r, iv, 0.04, 1.0, 0.2, -0.7, "put")
    elif strategy == "Strangle":
        return heston_price(S, strikes[1], T, r, iv, 0.04, 1.0, 0.2, -0.7, "call") + heston_price(S, strikes[0], T, r, iv, 0.04, 1.0, 0.2, -0.7, "put")
    elif strategy == "Iron Condor":
        put_spread = heston_price(S, strikes[1], T, r, iv, 0.04, 1.0, 0.2, -0.7, "put") - heston_price(S, strikes[0], T, r, iv, 0.04, 1.0, 0.2, -0.7, "put")
        call_spread = heston_price(S, strikes[3], T, r, iv, 0.04, 1.0, 0.2, -0.7, "call") - heston_price(S, strikes[2], T, r, iv, 0.04, 1.0, 0.2, -0.7, "call")
        return put_spread + call_spread
# Butterflies are a bit more troublesome, needed Chatgpt for some help lol
    elif "Butterfly" in strategy:
        if "Call" in strategy:
            return (heston_price(S, strikes[0], T, r, iv, 0.04, 1.0, 0.2, -0.7, "call") -
                    2*heston_price(S, strikes[1], T, r, iv, 0.04, 1.0, 0.2, -0.7, "call") +
                    heston_price(S, strikes[2], T, r, iv, 0.04, 1.0, 0.2, -0.7, "call"))
        else:
            return (heston_price(S, strikes[0], T, r, iv, 0.04, 1.0, 0.2, -0.7, "put") -
                    2*heston_price(S, strikes[1], T, r, iv, 0.04, 1.0, 0.2, -0.7, "put") +
                    heston_price(S, strikes[2], T, r, iv, 0.04, 1.0, 0.2, -0.7, "put"))

# ------------------------------------------------------
# Section 3: For the MTM value --> I used a default IV 4% to solve for the rf rate --> An extra step will be to use Newton-Raphson method to find IV --> Then invert to find 
# option price --> This will be much more computationally-intensive, and the solution stability depends on deviation of initial guess from fair value
current_iv = 0.04
current_value = calculate_strategy_price(S0, current_iv, strategy, 
                                       [leg["strike"] for leg in option_data], 
                                       T, risk_free_rate)
st.subheader(f"Current Mark-to-Market Value: ${current_value:.2f}")

# ------------------------------------------------------
# Section 4: Heatmap with $0.5 Stock Price increments 
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

# ------------------------------------------------------
## Section 5: Show Payoff & PnL section
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
    S0, current_iv, strategy, K_list, T, risk_free_rate
)

# 5) Build P&L curves
pnl_curves = {}
for label, tau_rem in taus.items():
    if tau_rem == 0:
        # at expiry, payoff minus premium
        pnl_curves[label] = payoffs - base_premium
    else:
        vals = np.array([
            calculate_strategy_price(
                S, current_iv, strategy, K_list, tau_rem, risk_free_rate
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

# plot P&L curves for each tau --> Each has diff color --> Had to use chatgpt for the names of these colors lol
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
## Section 6:
# Greeks Calculation Section
st.subheader("Strategy Greeks")

def compute_strategy_greeks(S0, iv, strategy, strikes, T, r):
    h = 0.01  # Perturbation size
    base_price = calculate_strategy_price(S0, iv, strategy, strikes, T, r)
    ## Note, I used Central Differences Method (within FDM) to calculate them
    # Delta
    price_up = calculate_strategy_price(S0 + h, iv, strategy, strikes, T, r)
    price_down = calculate_strategy_price(S0 - h, iv, strategy, strikes, T, r)
    delta = (price_up - price_down) / (2 * h)
    
    # Gamma
    gamma = (price_up - 2 * base_price + price_down) / (h ** 2)
    
    # Vega
    vega_up = calculate_strategy_price(S0, iv + h, strategy, strikes, T, r)
    vega_down = calculate_strategy_price(S0, iv - h, strategy, strikes, T, r)
    vega = (vega_up - vega_down) / (2 * h)
    
    # Theta
    theta = (calculate_strategy_price(S0, iv, strategy, strikes, T + h, r) - base_price) / h
    
    # Volga
    volga = (vega_up - 2 * base_price + vega_down) / (h ** 2)
    
    # Vanna
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

## Note this also uses the default IV (4%) to compute the greeks
# Calculate and display Greeks
greeks = compute_strategy_greeks(S0, current_iv, strategy, 
                                [leg["strike"] for leg in option_data], 
                                T, risk_free_rate)


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

# Calculate current MTM value
current_iv = 0.04  # Using model's volatility parameter
current_mtm = calculate_strategy_price(S0, current_iv, strategy, 
                                      [leg["strike"] for leg in option_data], 
                                      T, risk_free_rate)

st.subheader("Mark-to-Market & P&L Decomposition")
st.markdown(f"**The PnL Change to the Option Position is: ${current_mtm:.2f}**")

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
# Section 7: 
# Interactive P&L Calculator
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
# Section 8:
from plotly.subplots import make_subplots

# Build axes --> ttm, underlying which becomes surface panel
times   = np.linspace(0.1, 1.0, 20)
surface = np.zeros((len(times), len(stock_range)))
strikes = [leg["strike"] for leg in option_data]
for i, t in enumerate(times):
    for j, S in enumerate(stock_range):
        surface[i, j] = calculate_strategy_price(
            S, current_iv, strategy, strikes, t, risk_free_rate
        )

fig = make_subplots(rows=1, cols=1, specs=[[{"type": "surface"}]])

# Giving credits to chatgpt for lines 358-382, this helped with the presentation for IV surface
# Create the surface with a custom hovertemplate:
surface_trace = go.Surface(
    z=surface,
    x=stock_range,
    y=times,
    colorscale="Viridis",
    hovertemplate=(
        "Stock Price: %{x:.0f}<br>"
        "TTM: %{y:.4f} yrs<br>"
        "IV: %{z:.4f}<extra></extra>"
    )
)
fig_surface = go.Figure(data=[surface_trace])

fig_surface.update_layout(
    title="Implied Volatility Surface",
    scene=dict(
        xaxis_title="Underlying Price",
        yaxis_title="Time to Maturity (yrs)",
        zaxis_title="IV (%)",
        # turn off the “floor” and side‐walls so panels don’t connect:
        xaxis=dict(showbackground=False, showgrid=True),
        yaxis=dict(showbackground=False, showgrid=True),
        zaxis=dict(showbackground=False, showgrid=True),
    )
)

st.plotly_chart(fig_surface)

# ------------------------------------------------------
# Section 9:

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
