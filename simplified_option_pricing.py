import streamlit as st
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go

# --- Black‑Scholes Formula ---
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# --- First‑Order Greeks ---
def delta_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def delta_put(S, K, T, r, sigma):
    return delta_call(S, K, T, r, sigma) - 1

def gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def theta_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    term1 = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    term2 = - r * K * np.exp(-r * T) * norm.cdf(d2)
    return term1 + term2

def theta_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    term1 = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
    return term1 + term2

def rho_call(S, K, T, r, sigma):
    d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return K * T * np.exp(-r * T) * norm.cdf(d2)

def rho_put(S, K, T, r, sigma):
    d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)

# --- Second‑Order Greeks ---
def volga(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T) * d1 * (d1 - sigma * np.sqrt(T))

def vanna(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return T * S * norm.pdf(d1) * (1 - d1 / (sigma * np.sqrt(T)))

# --- Streamlit App ---
st.set_page_config(layout='wide')
st.title('European Option Pricing Dashboard')

# Input sliders
col1, col2 = st.columns(2)
with col1:
    S = st.slider('Stock Price (S)', 10.0, 500.0, 100.0)
    K = st.slider('Strike Price (K)', 10.0, 500.0, 100.0)
    T = st.slider('Time to Maturity (T, years)', 0.01, 5.0, 1.0)
with col2:
    r = st.slider('Risk-Free Rate (r)', 0.0, 0.2, 0.05)
    sigma = st.slider('Volatility (σ)', 0.01, 1.0, 0.2)

# Compute prices and Greeks
call_price = black_scholes_call(S, K, T, r, sigma)
put_price  = black_scholes_put(S, K, T, r, sigma)

delta_c = delta_call(S, K, T, r, sigma)
delta_p = delta_put(S, K, T, r, sigma)
gamma_v = gamma(S, K, T, r, sigma)
vega_v  = vega(S, K, T, r, sigma)
theta_c = theta_call(S, K, T, r, sigma)
theta_p = theta_put(S, K, T, r, sigma)
rho_c   = rho_call(S, K, T, r, sigma)
rho_p   = rho_put(S, K, T, r, sigma)
volga_v = volga(S, K, T, r, sigma)
vanna_v = vanna(S, K, T, r, sigma)

# Display values
st.subheader('Option Prices & Greeks')
col3, col4 = st.columns(2)
with col3:
    st.write(f'Call Price: {call_price:.2f}')
    st.write(f'Put Price:  {put_price:.2f}')
    st.write(f'Call Delta: {delta_c:.4f}')
    st.write(f'Put Delta:  {delta_p:.4f}')
with col4:
    st.write(f'Gamma:      {gamma_v:.4f}')
    st.write(f'Vega:       {vega_v:.4f}')
    st.write(f'Theta Call: {theta_c:.4f}')
    st.write(f'Theta Put:  {theta_p:.4f}')
    st.write(f'Rho Call:   {rho_c:.4f}')
    st.write(f'Rho Put:    {rho_p:.4f}')
    st.write(f'Volga:      {volga_v:.4f}')
    st.write(f'Vanna:      {vanna_v:.4f}')

# --- 3D Surface Plot ---
# Generate grid
S_vals = np.linspace(10, 500, 50)
sigma_vals = np.linspace(0.01, 1.0, 50)
S_grid, sig_grid = np.meshgrid(S_vals, sigma_vals)
price_grid = np.vectorize(black_scholes_call)(S_grid, K, T, r, sig_grid)

# Plotly surface
fig = go.Figure(data=[
    go.Surface(
        x=S_vals, y=sigma_vals, z=price_grid,
        colorscale='Viridis', opacity=0.8
    )
])
fig.update_layout(
    title='Call Price Surface (S vs σ)',
    scene=dict(
        xaxis_title='Stock Price (S)',
        yaxis_title='Volatility (σ)',
        zaxis_title='Call Price'
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

st.plotly_chart(fig, use_container_width=True)