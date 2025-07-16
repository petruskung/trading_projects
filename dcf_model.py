import requests
import pandas as pd
import numpy as np
import yfinance as yf
from io import StringIO
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Configure retries for all requests
session = requests.Session()
retry_strategy = Retry(
    total=5,
    backoff_factor=0.3,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

# Import ticker
ticker = 'aapl'
# Map each URL to descriptive sheet name
url_sheet_map = {
    f"http://stockanalysis.com/stocks/{ticker}/": "Overview",
    f"http://stockanalysis.com/stocks/{ticker}/financials/": "Income Statement",
    f"http://stockanalysis.com/stocks/{ticker}/financials/balance-sheet/": "Balance Sheet",
    f"http://stockanalysis.com/stocks/{ticker}/financials/cash-flow-statement/": "Cash Flow",
    f"http://stockanalysis.com/stocks/{ticker}/financials/ratios/": "Ratios"
}

# Create excel writer
with pd.ExcelWriter(f"{ticker}_financial_statements.xlsx") as writer:
    # Loop through each URL and corresponding sheet name
    for url, sheet_name in url_sheet_map.items():
        print(f"Processing: {url}")
        response = session.get(url)
        response.raise_for_status()  # Ensure request was successful
        
        # Use StringIO to handle HTML content
        html_content = StringIO(response.text)
        
        # Parse tables from current URL
        tables = pd.read_html(html_content)
        print(f"Found {len(tables)} tables at {url}.")
        
        # If there are multiple tables, write them sequentially
        startrow = 0  # Initialize row for writing
        for idx, table in enumerate(tables):
            header = pd.DataFrame({f"Table {idx} from {sheet_name}": []})
            header.to_excel(writer, sheet_name=sheet_name, startrow=startrow)
            startrow += 1  # Move down 1 row for table data
            
            # Write table to current sheet
            table.to_excel(writer, sheet_name=sheet_name, startrow=startrow)
            
            # Update startrow for next table (2 extra rows as spacer)
            startrow += len(table.index) + 2

print(f"All tables have been saved to '{ticker}_financial_statements.xlsx'")

# Parameters
TICKER = "AAPL"
EXCEL = f"{TICKER}_financial_statements.xlsx"
FY_COL = "FY2024"

def parse_value(val):
    if isinstance(val, str):
        val = val.replace(",", "").strip()
        if val in ['-', '', 'NA', 'N/A']:
            return np.nan
        if "%" in val:
            try:
                return float(val.replace("%", "").strip()) / 100
            except:
                return np.nan
        multipliers = {'B': 1e9, 'M': 1e6, 'T': 1e12}
        if val[-1] in multipliers:
            try:
                return float(val[:-1].strip()) * multipliers[val[-1]]
            except:
                return np.nan
        try:
            return float(val) * 1e6 if val[-1].isdigit() else np.nan
        except:
            return np.nan
    return np.nan if pd.isna(val) else val

def clean_sheet(sheet, file):
    df = pd.read_excel(file, sheet_name=sheet, header=None).iloc[4:].reset_index(drop=True)
    
    # Remove index column if it's just row numbers
    if pd.api.types.is_numeric_dtype(df.iloc[:, 0]) and (df.iloc[:, 0].fillna(-1) == pd.Series(range(len(df)))).all():
        df = df.iloc[:, 1:]
    
    n = df.shape[1]  # Number of columns
    if n == 7:
        df.columns = ["Item", FY_COL, "FY2023", "FY2022", "FY2021", "FY2020", "Notes"]
    elif n == 8:
        df.columns = ["Item", FY_COL, "FY2023", "FY2022", "FY2021", "FY2020", "Extra", "Notes"]
    else:
        df.columns = [f"Col{i}" for i in range(n)]
    
    # Convert numeric columns
    for col in df.columns:
        if col not in ["Item", "Notes"]:
            df[col] = df[col].apply(parse_value)
    
    return df

# Load Data
print("Loading and cleaning financial data...")
fin = clean_sheet("Income Statement", EXCEL)
bal = clean_sheet("Balance Sheet", EXCEL).set_index("Item")
cf = clean_sheet("Cash Flow", EXCEL)

# Function to extract values from DataFrames
def get_val(df, key, col=FY_COL, default=None):
    row = df[df["Item"].str.contains(key, case=False, na=False)]
    return row[col].values[0] if not row.empty else default

# Extract parameters for FCF
print("Calculating Free Cash Flow...")
EBIT = get_val(fin, "EBIT|Operating Income")
tax = 0.21  # Default tax rate
depre = get_val(cf, "Depreciation", default=0)
capex = abs(get_val(cf, "Capital Expenditure", default=0))

# Calculate Working Capital Change
if "Working Capital" in bal.index:
    wc = bal.loc["Working Capital"]
    delta_wc = wc.iloc[0] - (wc.iloc[1] if len(wc) > 1 else 0)
else:
    delta_wc = 0

# Calculate FCF with proper default parameter
FCF = get_val(cf, "Free Cash Flow", default=EBIT*(1-tax) + depre - capex - delta_wc)

# Forecasting future FCF
print("Forecasting future cash flows...")
DEFAULT_GROWTH = 0.15  # Assume 15% growth
FORECAST_YEARS = 5
growth = DEFAULT_GROWTH

if FCF and not np.isnan(FCF):
    forecast = [FCF * (1+growth) ** t for t in range(1, FORECAST_YEARS + 1)]
else:
    print("Warning: FCF is missing or invalid. Using default value.")
    FCF = 10**9  # Default to $1 billion
    forecast = [FCF * (1+growth) ** t for t in range(1, FORECAST_YEARS + 1)]

# Calculating WACC
print("Calculating WACC...")

# Set up Yahoo Finance with retries
yf_session = Session()
yf_session.mount("https://", adapter)
yf_ticker = yf.Ticker(TICKER, session=yf_session)

# Get stock info with error handling
try:
    info = yf_ticker.info
    price = info.get("currentPrice", None)
    beta = info.get("beta", 1.0)
    shares = info.get("sharesOutstanding", None)
    
    if shares is None:
        print("Warning: Shares outstanding not found. Using last reported value.")
        shares = bal.loc["Shares Outstanding (Basic)", FY_COL] if "Shares Outstanding (Basic)" in bal.index else None
except Exception as e:
    print(f"Error fetching Yahoo Finance data: {e}")
    # Fallback values
    beta = 1.0
    shares = bal.loc["Shares Outstanding (Basic)", FY_COL] if "Shares Outstanding (Basic)" in bal.index else None
    price = None

# Risk-free rate from 10-Year Treasury
try:
    rf = yf.Ticker("^TNX", session=yf_session).history(period="1d")["Close"].iloc[-1] / 100
except:
    print("Warning: Using default risk-free rate")
    rf = 0.04  # Fallback rate

# Market risk premium
mrp = 0.055  # Standard historical average

# Cost of equity
ce = rf + beta * mrp

# Cost of debt
de = rf + 0.02  # Add typical debt premium

# Calculate market values
if shares and price:
    market_equity = shares * price
else:
    print("Warning: Market equity not available")
    market_equity = None

if "Total Debt" in bal.index:
    market_debt = bal.loc["Total Debt", FY_COL]
else:
    market_debt = 0

# Capital structure weights
if market_equity and market_debt and (market_equity + market_debt) > 0:
    we = market_equity / (market_equity + market_debt)
    wd = market_debt / (market_equity + market_debt)
else:
    print("Warning: Using default capital structure weights")
    we, wd = 0.9, 0.1  # Conservative defaults

# WACC computation
WACC = we * ce + wd * de * (1 - tax)

# Terminal Value and DCF valuation
print("Calculating terminal value...")
TERM_GROWTH = 0.04  # Terminal growth rate assumption

# Discount forecasted FCFs
disc_FCF = [f / ((1+WACC) ** t) for t, f in enumerate(forecast, start=1)]

# Calculate terminal value
term_val = forecast[-1] * (1 + TERM_GROWTH) / (WACC - TERM_GROWTH)
disc_term = term_val / ((1+WACC) ** FORECAST_YEARS)

# Enterprise Value (EV)
EV = sum(disc_FCF) + disc_term

# Calculate net debt
if "Net Cash (Debt)" in bal.index:
    net_debt = bal.loc["Net Cash (Debt)", FY_COL]
elif "Total Debt" in bal.index and "Cash & Equivalents" in bal.index:
    net_debt = bal.loc["Total Debt", FY_COL] - bal.loc["Cash & Equivalents", FY_COL]
else:
    net_debt = 0

# Equity Value and Intrinsic Share Price
eq_value = EV - net_debt

if shares:
    intrinsic = eq_value / shares
else:
    intrinsic = None

# Final output
if intrinsic and price:
    print(f"\nValuation Results for {TICKER}:")
    print(f"Free Cash Flow: ${FCF/1e9:,.2f}B")
    print(f"WACC: {WACC*100:.2f}%")
    print(f"Terminal Value: ${term_val/1e9:,.2f}B")
    print(f"Enterprise Value: ${EV/1e9:,.2f}B")
    print(f"Equity Value: ${eq_value/1e9:,.2f}B")
    print(f"Intrinsic Price: ${intrinsic:,.2f}")
    print(f"Current Price: ${price:,.2f}")
    print(f"Margin of Safety: {(intrinsic/price - 1)*100:+.2f}%")
elif price:
    print(f"\nCurrent Price: ${price:,.2f}")
    print("Could not calculate intrinsic value due to missing data")
else:
    print("Valuation failed due to missing data")