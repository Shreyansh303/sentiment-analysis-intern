"""
universe.py — the expanded ticker universe (Nifty 50) for cross-sectional tests.

Maps each NSE ticker to a plain-English query name used to pull its news. With
~50 names the cross-sectional rank IC / long-short book becomes statistically
meaningful (4 names made per-window IC swing -0.80..+0.80). Constituents drift
over time; this is a representative mid-2020s list, good enough for the backtest.
"""

NIFTY_50 = {
    "RELIANCE.NS": "Reliance Industries", "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank", "INFY.NS": "Infosys", "ICICIBANK.NS": "ICICI Bank",
    "HINDUNILVR.NS": "Hindustan Unilever", "ITC.NS": "ITC Limited",
    "SBIN.NS": "State Bank of India", "BHARTIARTL.NS": "Bharti Airtel",
    "KOTAKBANK.NS": "Kotak Mahindra Bank", "LT.NS": "Larsen and Toubro",
    "AXISBANK.NS": "Axis Bank", "BAJFINANCE.NS": "Bajaj Finance",
    "ASIANPAINT.NS": "Asian Paints", "MARUTI.NS": "Maruti Suzuki",
    "HCLTECH.NS": "HCL Technologies", "SUNPHARMA.NS": "Sun Pharmaceutical",
    "TITAN.NS": "Titan Company", "WIPRO.NS": "Wipro",
    "ULTRACEMCO.NS": "UltraTech Cement", "NESTLEIND.NS": "Nestle India",
    "ONGC.NS": "Oil and Natural Gas Corporation", "NTPC.NS": "NTPC Limited",
    "POWERGRID.NS": "Power Grid Corporation", "M&M.NS": "Mahindra and Mahindra",
    "TATAMOTORS.NS": "Tata Motors", "TATASTEEL.NS": "Tata Steel",
    "JSWSTEEL.NS": "JSW Steel", "ADANIENT.NS": "Adani Enterprises",
    "ADANIPORTS.NS": "Adani Ports", "COALINDIA.NS": "Coal India",
    "BAJAJFINSV.NS": "Bajaj Finserv", "GRASIM.NS": "Grasim Industries",
    "HINDALCO.NS": "Hindalco Industries", "DRREDDY.NS": "Dr Reddy's Laboratories",
    "CIPLA.NS": "Cipla", "BRITANNIA.NS": "Britannia Industries",
    "EICHERMOT.NS": "Eicher Motors", "HEROMOTOCO.NS": "Hero MotoCorp",
    "BAJAJ-AUTO.NS": "Bajaj Auto", "DIVISLAB.NS": "Divi's Laboratories",
    "TECHM.NS": "Tech Mahindra", "INDUSINDBK.NS": "IndusInd Bank",
    "APOLLOHOSP.NS": "Apollo Hospitals", "TATACONSUM.NS": "Tata Consumer Products",
    "SBILIFE.NS": "SBI Life Insurance", "HDFCLIFE.NS": "HDFC Life Insurance",
    "LTIM.NS": "LTIMindtree", "SHRIRAMFIN.NS": "Shriram Finance",
    "BPCL.NS": "Bharat Petroleum",
}

TICKERS = list(NIFTY_50.keys())

MACRO_QUERIES = ["Nifty Sensex RBI", "FII flows Indian markets", "Fed rate decision"]

# Per-month fetch windows (full month so both A and B sub-windows are covered).
# June is the current month -> only data through ~today, so Jun-B is skipped later.
MONTH_FETCH = [
    ("jan", "2026-01-01", "2026-01-31"),
    ("feb", "2026-02-01", "2026-02-28"),
    ("mar", "2026-03-01", "2026-03-31"),
    ("apr", "2026-04-01", "2026-04-30"),
    ("may", "2026-05-01", "2026-05-31"),
    ("jun", "2026-06-01", "2026-06-18"),
]
