# portfolio_optimizer.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Union
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from pypfopt import risk_models, expected_returns
from pypfopt.efficient_frontier import EfficientFrontier, EfficientCVaR
from pypfopt.objective_functions import L2_reg, ex_ante_tracking_error, ex_post_tracking_error
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.risk_models import CovarianceShrinkage

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Optimizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Data Loaders ────────────────────────────────────────────────────────

@st.cache_data
def load_european_tickers() -> List[str]:
    """
    Scrape constituents from major European index Wiki pages,
    append the correct Yahoo suffix, and return a sorted unique list.
    """
    idx_info = {
        "FTSE100":  ("https://en.wikipedia.org/wiki/FTSE_100",           ["Ticker"],           [".L"]),
        "DAX":      ("https://en.wikipedia.org/wiki/DAX",                ["Ticker"],           [""]),
        "CAC40":    ("https://en.wikipedia.org/wiki/CAC_40",             ["Ticker"],           [""]),
        "AEX":      ("https://en.wikipedia.org/wiki/AEX_index",          ["Ticker symbol"],    [".AS"]),
        "FTSEMIB":  ("https://en.wikipedia.org/wiki/FTSE_MIB",           ["Ticker"],           [""]),
        "SMI":      ("https://en.wikipedia.org/wiki/Swiss_Market_Index", ["Ticker"],           [""]),
    }

    all_tickers: List[str] = []

    for name, (url, col_candidates, suffixes) in idx_info.items():
        tables = pd.read_html(url, header=0)
        found = False
        for df in tables:
            # normalize column names
            df = df.rename(columns=lambda c: str(c).strip())
            # try each candidate until one matches
            for col in col_candidates:
                if col in df.columns:
                    syms = (
                        df[col]
                        .dropna()
                        .astype(str)
                        .str.strip()
                        .unique()
                    )
                    # combine each raw symbol with each suffix
                    for sym in syms:
                        for suf in suffixes:
                            all_tickers.append(f"{sym}{suf}")
                    found = True
                    break  # no need to try other col names

            if found:
                break  
        if not found:
            st.warning(f"⚠️ Could not find any of {col_candidates} on {name} page")
    return sorted(set(all_tickers))


@st.cache_data
def load_sp500_tickers() -> List[str]:
    """
    Scrape the S&P 500 constituents from Wikipedia and return a sorted list of tickers.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(url, header=0)[0]
    return df["Symbol"].sort_values().tolist()


@st.cache_data
def fetch_price_data(
    tickers: Union[List[str], str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Download adjusted prices via yfinance with auto_adjust=True,
    then extract the 'Close' (adjusted close) series for each ticker,
    handling both MultiIndex (group_by='ticker') and single-index cases.
    """
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True,
        group_by="ticker"
    )
    if data.empty:
        st.error("⚠️ No price data found for those dates/tickers.")
        return pd.DataFrame()

    # MultiIndex: first level is ticker
    if isinstance(data.columns, pd.MultiIndex):
        symbols = data.columns.levels[0]
        try:
            prices = pd.concat(
                [data[sym]["Close"].rename(sym) for sym in symbols],
                axis=1
            )
        except KeyError:
            st.error("⚠️ Downloaded data lacks a ‘Close’ column.")
            return pd.DataFrame()
    else:
        if "Close" in data.columns:
            name = tickers if isinstance(tickers, str) else tickers[0]
            prices = data["Close"].to_frame(name=name)
        else:
            st.error("⚠️ Downloaded data has no ‘Close’ field.")
            return pd.DataFrame()

    prices.dropna(how="all", inplace=True)
    prices.fillna(method="ffill", inplace=True)
    return prices


# ─── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Petrus Kung")
    st.markdown("📧  [petruskung@hotmail.com](mailto:petruskung@hotmail.com)")
    st.markdown("📱  +44 7799 451 519")
    st.markdown("---")

    st.markdown("## Portfolio Inputs")

    # 1) load both universes
    sp500_tickers = load_sp500_tickers()
    euro_tickers  = load_european_tickers()
    all_tickers   = sp500_tickers + euro_tickers

    # 2) let user multiselect from the combined list
    tickers = st.multiselect(
        "Choose Stock Tickers:",
        options=all_tickers,
        default=[],
        help="Type to search among US & major European tickers"
    )

    # 3) Benchmark, dates, etc.
    benchmark = st.selectbox(
        "Choose a Benchmark:",
        [
            "^GSPC", "^IXIC", "^DJI",
            "SPY", "QQQ", "DIA"
        ],
        index=0
    )
    start_date = st.date_input("Start Date", value=pd.to_datetime("2010-01-01"))
    end_date   = st.date_input("End Date",   value=pd.to_datetime("2025-04-09"))

    st.markdown("---")
    st.markdown("#### Quick Guide")
    st.markdown("📊 Asset Analysis  |  📈 Portfolio Comparison  |  ⚖️ Mean-Risk")

# ─── Validation ──────────────────────────────────────────────────────────────
if not tickers:
    st.warning("Select at least one ticker in the sidebar to begin.")
    st.stop()

prices = fetch_price_data(tickers, start_date, end_date)
if prices.empty:
    st.stop()

# ─── Main Tabs ────────────────────────────────────────────────────────────────
asset_tab, comp_tab, mr_tab = st.tabs([
    "Asset Analysis",
    "Portfolio Comparison",
    "Mean-Risk"
])

# -------------------- Asset Analysis --------------------
@st.cache_data(show_spinner=False)
def load_asset_info(tickers: list[str]) -> pd.DataFrame:
    info_dict = {
        "Sector": {}, "Industry": {}, "Market Cap": {},
        "Country": {}, "P/E Ratio": {}
    }
    for t in tickers:
        tk   = yf.Ticker(t)
        info = tk.info
        info_dict["Sector"][t]     = info.get("sector", "N/A")
        info_dict["Industry"][t]   = info.get("industry", "N/A")

        mc = info.get("marketCap", np.nan)
        if pd.notna(mc):
            if mc >= 1e12:
                info_dict["Market Cap"][t] = f"${mc/1e12:.2f}T"
            else:
                info_dict["Market Cap"][t] = f"${mc/1e9:.2f}B"
        else:
            info_dict["Market Cap"][t] = "N/A"

        info_dict["Country"][t]    = info.get("country", "N/A")

        pe = info.get("trailingPE", np.nan)
        info_dict["P/E Ratio"][t]  = f"{pe:.2f}" if pd.notna(pe) else "N/A"

    return pd.DataFrame(info_dict).T

@st.cache_data(show_spinner=False)
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()

@st.cache_data(show_spinner=False)
def compute_correlation(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr() * 100

with asset_tab:
    st.header("Asset Analysis")
    st.markdown(
        "💡 Gain insights into performance, risk metrics, sector mix, correlations, and their relationship with benchmarks."
    )
    analyses = st.multiselect(
        "Select analyses to display:",
        ["Asset Information", "Correlation Matrix", "Sector Allocation",
         "Beta vs Benchmark", "Distribution of Returns"],
        default = []
    )
    
    # 🔧 Asset Information
    if "Asset Information" in analyses:
        st.subheader("Asset Information")
        info_df = load_asset_info(tickers)
        st.dataframe(
            info_df,
            use_container_width=True,
            height=200
        )

    # Precompute returns once
    returns = compute_returns(prices)

    if "Correlation Matrix" in analyses:
        st.subheader("Correlation Matrix")
        corr = compute_correlation(returns)
        fig = px.imshow(
            corr,
            text_auto=".1f",
            color_continuous_scale="RdBu_r",
            origin="lower",
            aspect="auto",
            labels=dict(x="Ticker", y="Ticker", color="Corr (%)"),
        )
        fig.update_layout(height=500, margin=dict(l=40, r=40, t=40, b=40))
        st.plotly_chart(fig, use_container_width=True)

    if "Sector Allocation" in analyses:
        st.subheader("Sector Allocation")
        info_df = load_asset_info(tickers).T  # now index=ticker
        sector_counts = info_df["Sector"].value_counts()
        fig = px.pie(
            names=sector_counts.index,
            values=sector_counts.values,
            hole=0,
            labels={"names": "Sector", "values": "% of Tickers"},
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    if "Beta vs Benchmark" in analyses:
        st.subheader("Beta vs Benchmark")

        # 1. Fetch benchmark prices on its own
        bm_prices = fetch_price_data(benchmark, start_date, end_date)
        if bm_prices.empty:
            st.error(f"No data for benchmark {benchmark}.")
        else:
        # If multi‐column, take the first column
            bm_series = bm_prices.iloc[:, 0].pct_change().dropna()

            betas = {}
            for t in tickers:
            # Align the dates
                tr = prices[t].pct_change().dropna()
                dfb = pd.concat([bm_series, tr], axis=1).dropna()

            # Linear regression of ticker returns vs. benchmark returns
                slope, _, _, _, _ = stats.linregress(dfb.iloc[:, 0], dfb.iloc[:, 1])
                betas[t] = slope

        # Build a DataFrame for plotting
            beta_df = pd.DataFrame.from_dict(betas, orient="index", columns=["Beta"])
            beta_df["Ticker"] = beta_df.index

        # Plot with a “hot” color scale so higher Beta → yellow
            fig = px.bar(
                beta_df,
                x="Ticker",
                y="Beta",
                color="Beta",
                color_continuous_scale="Plasma",
                text=beta_df["Beta"].round(2),
                labels={"Beta": ""}
        )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                yaxis_title="Beta",
                coloraxis_colorbar=dict(title="Beta"),
                uniformtext_minsize=8,
                uniformtext_mode="hide",
                height=450,
                margin=dict(l=40, r=40, t=40, b=40)
        )
            st.plotly_chart(fig, use_container_width=True)

    # 4) Distribution of Returns
    if "Distribution of Returns" in analyses:
        st.subheader("Distribution of Returns vs Normal Distribution")
        filt = st.multiselect(
            "Select Tickers to Display:",
            options=tickers,
            default=tickers[:3]
        )
        fig = go.Figure()
        for t in filt:
            r = returns[t]
            # Histogram (density)
            fig.add_trace(
                go.Histogram(
                    x=r,
                    histnorm="probability density",
                    name=f"{t} Returns",
                    opacity=0.4
                )
            )
            # Normal fit
            mu, sigma = r.mean(), r.std()
            x_axis = np.linspace(r.min(), r.max(), 200)
            y_norm = stats.norm.pdf(x_axis, mu, sigma)
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=y_norm,
                    mode="lines",
                    name=f"{t} Normal Fit",
                    line=dict(width=2)
                )
            )
        fig.update_layout(
            xaxis_title="Returns",
            yaxis_title="Density",
            legend_title="",
            height=450,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistical properties
        st.subheader("Statistical Properties of Returns")
        stats_df = pd.DataFrame(index=filt)
        stats_df["Mean"] = returns[filt].mean()
        stats_df["Standard Deviation"] = returns[filt].std()
        stats_df["Skewness"] = returns[filt].skew()
        stats_df["Kurtosis"] = returns[filt].kurt()
        st.dataframe(stats_df.style.format({
            "Mean": "{:.4f}",
            "Standard Deviation": "{:.4f}",
            "Skewness": "{:.4f}",
            "Kurtosis": "{:.4f}"
        }), use_container_width=True, height=200)

# ---------------- Portfolio Comparison -------------------
with comp_tab:
    st.header("Portfolio Comparison")
    st.markdown("💡 Compare multiple portfolio optimization models on a Train/Test split.")

    # 1) Model selector
    model_opts = [
        "Equal Weighted",
        "Inverse Volatility",
        "Random",
        "Minimize Portfolio Volatility",
        "Mean-Risk – Maximum Sharpe Ratio",
        "Mean-Risk – Minimum CVaR",
        "Risk Parity – Variance",
        "Risk Budgeting – CVaR",
        "Risk Parity – Covariance Shrinkage"
    ]
    selected_models = st.multiselect(
        "📌 Select models to compare:",
        options=model_opts,
        default=model_opts[:4],
        help="Pick any of these portfolio construction models"
    )

    # 2) Global Parameters: Train/Test split slider
    test_frac = st.slider("Select Test Set Percentage", 0.10, 0.90, 0.33, 0.01)
    train_frac = 1 - test_frac
    st.markdown(f"<span style='color:pink'>Train: {train_frac:.0%} | Test: {test_frac:.0%}</span>",
                unsafe_allow_html=True)

    # 3) Run button
    if st.button("🚀 Run Portfolio Comparison"):
        # 3a) Split data
        split_idx = int(len(prices) * train_frac)
        train_prices = prices.iloc[:split_idx]
        test_prices  = prices.iloc[split_idx:]
        train_rets   = train_prices.pct_change().dropna()
        test_rets    = test_prices.pct_change().dropna()
        split_date   = train_prices.index[-1]

        # 3b) Helper: compute weights
        def get_weights(model: str) -> dict[str, float]:
            n = len(tickers)
        # 1) Equal‐Weighted
            if model == "Equal Weighted":
                return {t: 1/n for t in tickers}

    # 2) Inverse Volatility
            if model == "Inverse Volatility":
                vols = train_rets.std()
                inv = 1 / vols
                return (inv / inv.sum()).to_dict()

    # 3) Random
            if model == "Random":
                r = np.random.rand(n)
                return dict(zip(tickers, r/r.sum()))

    # 4) Minimize Portfolio Volatility
            if model == "Minimize Portfolio Volatility":
                μ = expected_returns.mean_historical_return(train_prices)
                Σ = CovarianceShrinkage(train_prices).ledoit_wolf()
                ef = EfficientFrontier(μ, Σ)
                ef.min_volatility()
                return ef.clean_weights()

    # 5) Mean-Risk – Maximum Sharpe Ratio
            if model == "Mean-Risk – Maximum Sharpe Ratio":
                μ = expected_returns.mean_historical_return(train_prices)
                Σ = CovarianceShrinkage(train_prices).ledoit_wolf()
                ef = EfficientFrontier(μ, Σ)
                ef.max_sharpe()
                return ef.clean_weights()

    # 6) Mean-Risk – Minimum CVaR
            if model == "Mean-Risk – Minimum CVaR":
                μ = expected_returns.mean_historical_return(train_prices)
                Σ = CovarianceShrinkage(train_prices).ledoit_wolf()
                ef = EfficientCVaR(μ, Σ)
                ef.min_cvar()
                return ef.clean_weights()
    
    # 7) Risk Parity – Variance
            if model == "Risk Parity – Variance":
        # 1) compute sample covariance on train set
                Σ = CovarianceShrinkage(train_prices).ledoit_wolf()

        # 2) drop any tickers with zero variance or missing values
                Σ = Σ.replace(0, np.nan).dropna(axis=0, how="any").dropna(axis=1, how="any")
        
        # 3) run HRP on that cleaned covariance
                hrp = HRPOpt(Σ)
                raw_w = hrp.optimize()

        # 4) re‐insert zero weights for any dropped tickers
                for t in tickers:
                    raw_w.setdefault(t, 0.0)
                return raw_w
    # 8) Risk Budgeting – CVaR
            if model == "Risk Budgeting – CVaR":
        # as a placeholder, equal weights—you’ll replace this
                return {t: 1/n for t in tickers}

    # 9) Risk Parity – Covariance Shrinkage
            if model == "Risk Parity – Covariance Shrinkage":
        # 1) compute & shrink
                cov_mat = CovarianceShrinkage(train_prices).ledoit_wolf()
                cov = pd.DataFrame(cov_mat, index=train_prices.columns, columns=train_prices.columns)

        # 2) clean as above
                cov.replace([np.inf, -np.inf], np.nan, inplace=True)
                cov_clean = cov.dropna(axis=0, how="any").dropna(axis=1, how="any")

        # 3) HRP on the shrunk cov
                hrp = HRPOpt(cov_clean)
                raw_w = hrp.optimize()

        # 4) re‐insert zeros for dropped tickers
                for t in train_prices.columns:
                    raw_w.setdefault(t, 0.0)
                return raw_w
    
    # Fallback (shouldn't happen)
            return {t: 1/n for t in tickers}

        # 3c) Compute all weights
        weights_dict = {m: get_weights(m) for m in selected_models}

        # 4) Portfolio Composition chart
        comp_df = pd.DataFrame(weights_dict).T  # rows=models, cols=tickers
        fig_comp = px.bar(
            comp_df,
            x=comp_df.index,
            y=comp_df.columns,
            title="Portfolio Composition",
            labels={"value": "Weight", "variable": "Asset"},
            text_auto=".2f"
        )
        fig_comp.update_layout(barmode="stack", height=400, margin=dict(t=50))
        st.plotly_chart(fig_comp, use_container_width=True)

        # 5) Cumulative Returns
        #   non-compounded = cumulative sum of daily portfolio returns
        cum_train = pd.DataFrame({
            m: train_rets[list(w.keys())].mul(list(w.values())).sum(axis=1).cumsum()
            for m, w in weights_dict.items()
        })
        cum_test = pd.DataFrame({
            m: test_rets[list(w.keys())].mul(list(w.values())).sum(axis=1).cumsum()
            for m, w in weights_dict.items()
        })

        # 5a) Combined Train & Test with shading
        fig_all = go.Figure()
        # train traces
        for m in selected_models:
            fig_all.add_trace(go.Scatter(
                x=cum_train.index, y=cum_train[m],
                mode="lines", name=f"Train – {m}"
            ))
        # test traces
        for m in selected_models:
            fig_all.add_trace(go.Scatter(
                x=cum_test.index, y=cum_test[m],
                mode="lines", name=f"Test – {m}"
            ))
        # shading
        fig_all.add_vrect(
            x0=split_date, x1=prices.index[-1],
            fillcolor="goldenrod", opacity=0.2, layer="below",
            annotation_text="Test Set", annotation_position="top left"
        )
        fig_all.add_vrect(
            x0=prices.index[0], x1=split_date,
            fillcolor="lightblue", opacity=0.1, layer="below",
            annotation_text="Train Set", annotation_position="bottom right"
        )
        fig_all.update_layout(
            title="Cumulative Returns (non-compounded) – Train & Test – All Portfolios",
            xaxis_title="Observations",
            yaxis_title="Cumulative Returns",
            height=450,
            margin=dict(t=50)
        )
        st.plotly_chart(fig_all, use_container_width=True)

        # 5b) Train-only
        fig_tr = px.line(
            cum_train,
            title="Cumulative Returns (non-compounded) – Train Set – All Portfolios",
            labels={"index": "Observations", "value": "Cumulative Returns", "variable": "Portfolio"}
        )
        fig_tr.update_layout(height=350, margin=dict(t=40))
        st.plotly_chart(fig_tr, use_container_width=True)

        # 5c) Test-only
        fig_te = px.line(
            cum_test,
            title="Cumulative Returns (non-compounded) – Test Set – All Portfolios",
            labels={"index": "Observations", "value": "Cumulative Returns", "variable": "Portfolio"}
        )
        fig_te.update_layout(height=350, margin=dict(t=40))
        st.plotly_chart(fig_te, use_container_width=True)

        # 6) Portfolio Summary tables
        def summarize(ps: pd.Series, alpha=0.05):
            m = ps.mean()
            v = ps.var()
            sv = ps[ps < 0].var()
            sd = ps.std()
            sdev = ps[ps < 0].std()
            mad = (ps - ps.mean()).abs().mean()
            # CVaR @95%
            var95 = ps.quantile(alpha)
            cvar = -ps[ps <= var95].mean()
            # EVaR @95% (grid search)
            xs = ps.values
            def evar_obj(t):
                return (np.log(np.mean(np.exp(t * xs))) - np.log(alpha)) / t
            ts = np.linspace(0.01, 10, 50)
            evar = max(evar_obj(t) for t in ts)
            return {
                "Mean": m,
                "Annualized Mean": m * 252,
                "Variance": v,
                "Annualized Variance": v * 252,
                "Semi-Variance": sv,
                "Ann. Semi-Variance": sv * 252,
                "Standard Deviation": sd,
                "Ann. Standard Deviation": sd * np.sqrt(252),
                "Semi-Deviation": sdev,
                "Ann. Semi-Deviation": sdev * np.sqrt(252),
                "Mean Absolute Deviation": mad,
                "CVaR @95%": cvar,
                "EVaR @95%": evar
            }

        # Build summary DataFrames
        sum_train = pd.DataFrame({
            m: summarize(cum_train[m].diff().dropna()) for m in selected_models
        })
        sum_test  = pd.DataFrame({
            m: summarize(cum_test[m].diff().dropna()) for m in selected_models
        })

        st.subheader("Portfolio Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Train Set**")
            st.dataframe(sum_train.style.format("{:.2%}"), height=350)
        with col2:
            st.markdown("**Test Set**")
            st.dataframe(sum_test.style.format("{:.2%}"), height=350)

# ---------------------- Mean-Risk ------------------------

with mr_tab:
    st.header("Mean-Risk – Efficient Frontier")

    # 1) TRAIN/TEST SPLIT fraction (you already have train_frac from sidebar)
    split_idx   = int(len(prices) * train_frac)
    train_p     = prices.iloc[:split_idx]
    test_p      = prices.iloc[split_idx:]
    train_r     = train_p.pct_change().dropna()
    test_r      = test_p.pct_change().dropna()
    split_date  = train_p.index[-1]

    # 2) UI controls
    n_pts = st.slider(
        "Select Number of Portfolios on the Efficient Frontier", 
        min_value=5, max_value=50, value=10
    )

    if st.button("🚀 Train & Test – Efficient Frontier"):

        st.success("✅ Efficient frontier computed!")

        # 3) Estimate μ & Σ (no need to annualize returns; PyPortfolioOpt does internally)
        mu_train    = expected_returns.mean_historical_return(train_p)
        Sigma_train = risk_models.sample_cov(train_p)
        mu_test     = expected_returns.mean_historical_return(test_p)
        Sigma_test  = risk_models.sample_cov(test_p)

        # 4) Helper: compute N portfolios along the frontier
        def compute_frontier_weights(mu: pd.Series, Sigma: pd.DataFrame, n: int):
            eps     = 1e-4
            r_min   = mu.min() + eps
            r_max   = mu.max() - eps
            targets = np.linspace(r_min, r_max, n)

            weights_dict = {}
            for i, tgt in enumerate(targets):
                ef = EfficientFrontier(mu, Sigma)
                w  = ef.efficient_return(target_return=tgt)
                weights_dict[f"ptf{i}"] = pd.Series(w)
            return weights_dict

        comp_w_train = compute_frontier_weights(mu_train, Sigma_train, n_pts)
        comp_w_test  = compute_frontier_weights(mu_test,  Sigma_test,  n_pts)

        # 5) Portfolio Composition (stacked bar)
        df_comp = pd.DataFrame(comp_w_train).T
        fig_comp = px.bar(
            df_comp,
            barmode="stack",
            labels={"value":"Weight","index":"Portfolio","variable":"Asset"},
            title="Portfolio Composition – Efficient Frontier",
            template="plotly_dark"
        )
        fig_comp.update_layout(margin=dict(t=50,l=40,r=40,b=40))
        st.plotly_chart(fig_comp, use_container_width=True)

        # 6) Efficient Frontier scatter
        records = []
        for tag, (mu, Sigma, wdict) in [
            ("Train", (mu_train,    Sigma_train,    comp_w_train)),
            ("Test",  (mu_test,     Sigma_test,     comp_w_test ))
        ]:
            for ptf, w in wdict.items():
                w_vec = w.reindex(mu.index).fillna(0).values
                ann_ret = mu.dot(w_vec)
                ann_vol = np.sqrt(w_vec.T @ Sigma @ w_vec)
                ann_shp = ann_ret / ann_vol
                records.append({
                    "Portfolio": ptf,
                    "Ann Std Dev": ann_vol,
                    "Ann Mean":    ann_ret,
                    "Sharpe":      ann_shp,
                    "Tag":         tag
                })

        ef_df = pd.DataFrame(records)
        fig_ef = px.scatter(
            ef_df,
            x="Ann Std Dev",
            y="Ann Mean",
            color="Sharpe",
            symbol="Tag",
            hover_data=["Portfolio"],
            labels={
                "Ann Std Dev":"Annualized Standard Deviation",
                "Ann Mean":"Annualized Mean",
                "Sharpe":"Annualized Sharpe Ratio"
            },
            color_continuous_scale="Blues",
            template="plotly_dark",
            title="Mean-Risk – Efficient Frontier"
        )
        fig_ef.update_xaxes(tickformat=".1%")
        fig_ef.update_yaxes(tickformat=".1%")
        fig_ef.update_traces(marker=dict(size=12, line_width=1))
        st.plotly_chart(fig_ef, use_container_width=True)

        # 7) Cumulative Returns
        # build a DataFrame of pct‐change returns for each frontier‐ptf
        cum_train = pd.DataFrame({
            ptf: (train_r.dot(w) + 1).cumprod() - 1
            for ptf, w in comp_w_train.items()
        })
        cum_test  = pd.DataFrame({
            ptf: (test_r.dot(comp_w_train[ptf]) + 1).cumprod() - 1
            for ptf in comp_w_train
        })

        # 7a) Train only
        st.plotly_chart(
            px.line(
                cum_train,
                labels={"value":"Cumulative Returns (%)","index":"Observations"},
                title="Cumulative Returns – Train Set"
            ).update_layout(template="plotly_dark"),
            use_container_width=True
        )

        # 7b) Test only
        st.plotly_chart(
            px.line(
                cum_test,
                labels={"value":"Cumulative Returns (%)","index":"Observations"},
                title="Cumulative Returns – Test Set"
            ).update_layout(template="plotly_dark"),
            use_container_width=True
        )

        # 8) Portfolio Summary tables 
        def compute_portfolio_summary(cum_returns: pd.DataFrame):
            rets = cum_returns.diff().dropna()  # per‐period returns
            annual_factor = 252
            summary = pd.DataFrame(index=rets.columns)
            summary["Mean"]                    = rets.mean()
            summary["Annualized Mean"]         = rets.mean() * annual_factor
            summary["Variance"]                = rets.var()
            summary["Annualized Variance"]     = rets.var() * annual_factor
            summary["Semi-Variance"]           = (rets[rets < 0]**2).mean()
            summary["Ann. Semi-Variance"]      = (rets[rets < 0]**2).mean() * annual_factor
            summary["Std Dev"]                 = rets.std()
            summary["Ann. Std Dev"]            = rets.std() * np.sqrt(annual_factor)
            summary["Semi-Dev"]                = rets[rets < 0].std()
            summary["Ann. Semi-Dev"]           = rets[rets < 0].std() * np.sqrt(annual_factor)
    # CVaR at 95%
            summary["CVaR 95%"]                = rets.quantile(0.05)
            for col in summary.columns:
                summary[col] = summary[col]
            return summary
    # Format as percentages:
        summary_train = compute_portfolio_summary(cum_train)
        summary_test  = compute_portfolio_summary(cum_test)    

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Portfolio Summary – Train Set")
            st.dataframe(summary_train.style.format("{:.2%}"), use_container_width=True)
        with col2:
            st.subheader("Portfolio Summary – Test Set")
            st.dataframe(summary_test.style.format("{:.2%}"), use_container_width=True)
