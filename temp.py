import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Fetch financial data
cipla = yf.Ticker("CIPLA.NS")
balance_sheet = cipla.balance_sheet
income_statement = cipla.financials

# Transpose data for easier manipulation
balance_sheet = balance_sheet.T
income_statement = income_statement.T

# Step 2: Common size calculation
# Common size balance sheet as percentage of total assets
balance_sheet_common_size = balance_sheet.div(balance_sheet['Total Assets'], axis=0) * 100

# Common size income statement as percentage of revenue
income_statement_common_size = income_statement.div(income_statement['Total Revenue'], axis=0) * 100

# Step 3: Calculate some financial ratios
# Liquidity Ratios
current_ratio = balance_sheet['Total Current Assets'] / balance_sheet['Total Current Liabilities']
quick_ratio = (balance_sheet['Total Current Assets'] - balance_sheet['Inventory']) / balance_sheet['Total Current Liabilities']

# Profitability Ratios
gross_margin = (income_statement['Gross Profit'] / income_statement['Total Revenue']) * 100
operating_margin = (income_statement['Operating Income'] / income_statement['Total Revenue']) * 100
net_margin = (income_statement['Net Income'] / income_statement['Total Revenue']) * 100

# Step 4: Plotting the ratios
plt.figure(figsize=(10, 8))

# Plot Liquidity Ratios
plt.subplot(2, 2, 1)
plt.plot(current_ratio.index, current_ratio, marker='o', label='Current Ratio')
plt.plot(quick_ratio.index, quick_ratio, marker='o', label='Quick Ratio')
plt.title("Liquidity Ratios")
plt.legend()

# Plot Profitability Ratios
plt.subplot(2, 2, 2)
plt.plot(gross_margin.index, gross_margin, marker='o', label='Gross Margin')
plt.plot(operating_margin.index, operating_margin, marker='o', label='Operating Margin')
plt.plot(net_margin.index, net_margin, marker='o', label='Net Margin')
plt.title("Profitability Ratios")
plt.legend()

plt.tight_layout()
plt.show()
