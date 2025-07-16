import requests
import pandas as pd
import numpy as np
import yfinance as yf

# import ticker
ticker = 'nvda'
# map each URL to descriptive sheet name

url_sheet_map = {
    f"http://stockanalysis.com/stocks/{ticker}/": "Overview",
    f"http://stockanalysis.com/stocks/{ticker}/financials/": "Income Statement",
    f"http://stockanalysis.com/stocks/{ticker}/financials/balance-sheet/": "Balance Sheet",
    f"http://stockanalysis.com/stocks/{ticker}/financials/cash-flow-statement/": "Cash Flow",
    f"http://stockanalysis.com/stocks/{ticker}/financials/ratios/": "Ratios"
}

# Create excel writer
with pd.ExcelWriter(f"{ticker}_financial_statements.xlsx") as writer:
    # loop through each URL and corresponding sheet name
    for url, sheet_name in url_sheet_map.items():
        print(f"Processing: {url}")
        response = requests.get(url)
        response.raise_for_status() #ensure request was successful

        # parse all tables from current url
        tables = pd.read_html(response.text)
        print(f"Found {len(tables)} tables at {url}.")

        # If there are multiple tables, we write them sequentially in the same sheet
        startrow = 0 #initialize row for writing
        for idx, table in enumerate(tables):
            header = pd.DataFrame({f"Table {idx} from {sheet_name}": []})
            header.to_excel(writer, sheet_name=sheet_name, startrow=startrow)
            startrow += 1 #move down 1 row for the table data
            # now, we write the table to the current sheet starting at the designated row
            table.to_excel(writer, sheet_name=sheet_name, startrow=startrow)
            # then. we update the startrow for the next stable (2 extra rows as spacer)
            startrow += len(table.index) + 2
print("All tables have now been saved into 'tables_by_url.xlsx', each URL in its own sheet." )

# Parameters
TICKER = "NVDA"
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
        m = {'B': 1e9, 'M': 1e6, 'T': 1e12}
        if val[-1] in m:
            try:
                return float(val[:-1].strip()) * m[val[-1]]
            except:
                return np.nan
        try:
            return float(val) * 1e6 if val[-1].isdigit() else np.nan
        except:
            return np.nan
    return np.nan if pd.isna(val) else val

def clean_sheet(sheet, file):
    df = pd.read_excel(file, sheet_name=sheet,
                       header=None).iloc[4:].reset_index(drop=True)
    if pd.api.types.is_numeric_dtype(df.iloc[:, 0]) and (df.iloc[:, 0].fillna(-1) == pd.Series(range(len(df)))).all():
        df = df.iloc[:, 1:]
    n = df.shape[1]
    if n== 7:
        df.columns = ["Item", FY_COL, "FY2023", "FY2022", "FY2021", "FY2020", "Notes"]
    elif n==8:
        df.columns = ["Item", FY_COL, "FY2023", "FY2022", "FY2021", "FY2020", "Extra", "Notes"]
    else:
        df.columns = [f"Col{i}" for i in range(n)]
    for c in df.columns:
        if c not in ["Item", "Notes"]:
            df[c] = df[c].apply(parse_value)
    return df

# Load Data
fin = clean_sheet("Income Statement", EXCEL)
bal = clean_sheet("Balance Sheet", EXCEL).set_index("Item")
cf = clean_sheet("Cash Flow", EXCEL)

# computing FCF to firm
# function to extract different parameters 
def get_val(df, key, col=FY_COL, default=None):
    row = df[df["Item"].str.contains(key, case=False, na=False)]
    return row[col].values[0] if not row.empty else default
# Extract parameters for FCF
EBIT = get_val(fin, "EBIT|Operating Income")
tax = 0.21 #Default tax rate
depre = get_val(cf, "Depreciation", default=0)
capex = abs(get_val(cf, "Capital Expenditure", default=0))
# Calculate WC Change if available
if "Working Capital" in bal.index:
    wc = bal.loc["Working Capital"]
    delta_wc = wc.iloc[0] - (wc.iloc[1] if len(wc) > 1 else 0)
else:
    delta_wc = 0
# Calculate net debt
if "Net Cash (Debt)" in bal.index:
    net_debt = bal.loc["Net Cash (Debt)", FY_COL]
else:
    net_debt = (bal.loc["Total Debt", FY_COL] - bal.loc["Cash & Equivalents", FY_COL] 
                if "Total Debt" in bal.index and "Cash & Equivalents" in bal.index else 0)
# Calculate FCF
FCF = get_val(cf, "Free Cash Flow", default=EBIT*(1-tax) + depre - capex - delta_wc)

# Forecasting future FCF over set period with assumed growth rate
# set period often 5-10 years
DEFAULT_GROWTH = 0.60 #assume 15% growth
FORECAST_YEARS = 5

growth = DEFAULT_GROWTH
forecast = [FCF * (1+growth) ** t for t in range(1, FORECAST_YEARS + 1)] if FCF else [None] * FORECAST_YEARS

# Calculating WACC (cost of equity -> CAPM)
# Retrieve data
info = yf.Ticker(TICKER).info
beta = info.get("beta", 1.0)
shares = info.get("sharesOutstanding", None)
price = info.get("currentPrice", None)

# Risk-free rate from TLT
rf = yf.Ticker("TLT").info.get("yield", 0.022)
spy_hist = yf.Ticker("SPY").history(period="20y")["Close"].resample("Y").last()
if len(spy_hist) >= 2:
    spy_cagr = (spy_hist.iloc[-1] / spy_hist.iloc[0]) ** (1/(len(spy_hist)-1)) -1
else:
    spy_cagr = 0.08
mrp = spy_cagr - rf
# cost of equity
ce = rf + beta * mrp
# cost of debt
de = rf

# determine market value weights
market_equity = shares * price if shares and price else None
market_debt = bal.loc["Total Debt", FY_COL] if "Total Debt" in bal.index else 0
if market_equity and (market_equity + market_debt) > 0:
    we = market_equity / (market_equity + market_debt)
    wd = market_debt / (market_equity + market_debt)
else:
    we, wd = 1, 0
# WACC computation
WACC = we * ce + wd * de * (1-tax)

# Terminal Value and DCF valuation
TERM_GROWTH = 0.60 # Terminal growth rate assumption
# Discount forecasted FCFs
disc_FCF = [f / ((1+WACC) ** t) for t, f in enumerate(forecast, start=1)]
# calculate terminal value using last year's FCF
term_val = forecast[-1] * (1 + TERM_GROWTH) / (WACC - TERM_GROWTH)
disc_term = term_val / ((1+WACC) ** FORECAST_YEARS)
# Enterprise Value (EV)
EV = sum(disc_FCF) + disc_term
# Equity Value and Intrinsic Share Price
eq_value = EV - net_debt
intrinsic = eq_value / shares if shares else None

info = yf.Ticker(TICKER).info
price = info.get("currentPrice", None)
print(f"Intrinsic Price: {intrinsic:,.2f}, Current Price: {price:,.2f} Done")



