import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(4)
risk_free_rate=.0687

tickers = "SPY BTC-USD BANDHANBNK.NS GILLETTE.NS NCC.NS FORTIS.NS RECLTD.NS DIVISLAB.NS COROMANDEL.NS"
start_date = "2023-01-01"
end_date = "2023-12-31"
data = yf.download(tickers, start=start_date, end=end_date)
print(f"data:\n {data}")

df=data[[('Adj Close', 'BANDHANBNK.NS'),
         ('Adj Close', 'COROMANDEL.NS'),
            ('Adj Close',   'DIVISLAB.NS'),
            ('Adj Close',     'FORTIS.NS'),
            ('Adj Close',   'GILLETTE.NS'),
            ('Adj Close',        'NCC.NS'),
            ('Adj Close',     'RECLTD.NS'),]]
df.dropna(inplace=True)
print(f"df: \n{df}")

returns = np.log(df / df.shift(1)).dropna()
returns.columns = returns.columns.get_level_values(1)
mean_returns = returns.mean()
annualised_mean_returns=(1+mean_returns)**252 -1
print(f"annualised_mean_returns: \n {annualised_mean_returns}")

cov_matrix = returns.cov()
print(f"cov_matrix: {cov_matrix*252}")

variance=returns.var()
annualised_variance=variance*252
standard_deviation=np.sqrt(annualised_variance)
print(f"standard_deviation: \n {standard_deviation}")

# Equally Weighted Portfolio
x=1/7
weights=np.array([x for i in range(0,7)])
portfolio_return=np.dot(annualised_mean_returns,weights)
portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))*252
portfolio_risk=np.sqrt(portfolio_variance)

print(f"portfolio_variance: {portfolio_variance}")
print(f"portfolio_risk: {portfolio_risk}")
print(f"portfolio_return: {portfolio_return}")

Sharpe_ratio=portfolio_return-risk_free_rate
Sharpe_ratio=Sharpe_ratio/portfolio_risk
print(f"Sharpe_ratio: {Sharpe_ratio}")

# Minimum Variance Portfolio
def make_random_portfolio(n,returns, covariance_matrix):
    random_weights=np.random.rand(n)
    random_weights=random_weights/np.sum(random_weights)
    portfolio_return=np.dot(random_weights,returns)
    portfolio_variance=np.dot(random_weights.T,np.dot(covariance_matrix,random_weights))*252

    return random_weights,portfolio_return,portfolio_variance


min_variance=100
er=[]
sigma=[]
sharpe_ratio=0
min_weights=np.ones(7)
for i in range(0,100000):
    x,y,z=make_random_portfolio(7,annualised_mean_returns,cov_matrix)
    er.append(y)
    sigma.append(np.sqrt(z))
    if(z<min_variance):
        min_variance=z
        min_weights=x
        min_return=y
    sharpe_temp=(y-risk_free_rate)/np.sqrt(z)
    if(sharpe_temp>sharpe_ratio):
        sharpe_ratio=sharpe_temp
        sharpe_weights=x
        sharpe_return=y
        sharpe_variance=z

print(f"min_var_weights: {min_weights}")
print(f"min_var_return: {min_return}")
print(f"min_variance: {min_variance}")
print(f"max_sharpe_ratio: {sharpe_ratio}")
print(f"max_sharpe_weights: {sharpe_weights}")
print(f"max_sharpe_return: {sharpe_return}")
print(f"max_sharpe_variance: {sharpe_variance}")

# Efficient Frontier and Tangency Portfolio
plt.figure(figsize=(10,10))
plt.scatter(sigma,er)
plt.scatter(np.sqrt(sharpe_variance),sharpe_return,c="r")
plt.scatter(np.sqrt(min_variance),min_return, c='y')

plt.annotate('Tangency Portfolio', 
             xy=(np.sqrt(sharpe_variance),sharpe_return), 
             xytext=(np.sqrt(sharpe_variance) + 0.01,sharpe_return + 0.01),
             fontsize=10)

plt.annotate('Minimum Variance', 
             xy=(np.sqrt(min_variance),min_return), 
             xytext=(np.sqrt(min_variance) + 0.01,min_return + 0.01),
             fontsize=10)

x = np.linspace(0, max(sigma), 100)
y = risk_free_rate + sharpe_ratio * x
plt.title("Efficient Frontier and Tangency Portfolio, Domestic")
plt.xlabel('Portfolio Risk (Standard Deviation)')
plt.ylabel('Portfolio Return')
plt.plot(x,y,c='purple',label='CAL',linewidth=2)
plt.legend()
plt.show()
# International Portfolio with SPY 

df=data[[('Adj Close', 'BANDHANBNK.NS'),
            ('Adj Close', 'COROMANDEL.NS'),
            ('Adj Close',   'DIVISLAB.NS'),
            ('Adj Close',     'FORTIS.NS'),
            ('Adj Close',   'GILLETTE.NS'),
            ('Adj Close',        'NCC.NS'),
            ('Adj Close',     'RECLTD.NS'),
            ('Adj Close',           'SPY'),]]
df.dropna(inplace=True)
print(f"df: \n {df}")

returns = np.log(df / df.shift(1)).dropna()
returns.columns = returns.columns.get_level_values(1)
mean_returns = returns.mean()
annualised_mean_returns=(1+mean_returns)**252 -1
print(f"annualised_mean_returns: \n {annualised_mean_returns}")

cov_matrix = returns.cov()
print(f"cov_matrix: {cov_matrix}")

variance=returns.var()
annualised_variance=variance*252
standard_deviation=np.sqrt(annualised_variance)
print(f"standard_deviation: \n {standard_deviation}")

# Minimum Variance Portfolio
min_variance=100
er=[]
sigma=[]
sharpe_ratio=0
min_weights=np.ones(8)
for i in range(0,100000):
    x,y,z=make_random_portfolio(8,annualised_mean_returns,cov_matrix)
    er.append(y)
    sigma.append(np.sqrt(z))
    if(z<min_variance):
        min_variance=z
        min_weights=x
        min_return=y
    sharpe_temp=(y-risk_free_rate)/np.sqrt(z)
    if(sharpe_temp>sharpe_ratio):
        sharpe_ratio=sharpe_temp
        sharpe_weights=x
        sharpe_return=y
        sharpe_variance=z

print(f"min_var_weights: {min_weights}")
print(f"min_var_return: {min_return}")
print(f"min_variance: {min_variance}")
print(f"max_sharpe_ratio: {sharpe_ratio}")
print(f"max_sharpe_weights: {sharpe_weights}")
print(f"max_sharpe_return: {sharpe_return}")
print(f"max_sharpe_variance: {sharpe_variance}")

# Efficient Frontier and Tangency Portfolio
plt.figure(figsize=(10,10))
plt.scatter(sigma,er)
plt.scatter(np.sqrt(sharpe_variance),sharpe_return,c="r")
plt.scatter(np.sqrt(min_variance),min_return, c='y')

plt.annotate('Tangency Portfolio', 
             xy=(np.sqrt(sharpe_variance),sharpe_return), 
             xytext=(np.sqrt(sharpe_variance) + 0.01,sharpe_return + 0.01),
             fontsize=10)

plt.annotate('Minimum Variance', 
             xy=(np.sqrt(min_variance),min_return), 
             xytext=(np.sqrt(min_variance) + 0.01,min_return + 0.01),
             fontsize=10)

x = np.linspace(0, max(sigma), 100)
y = risk_free_rate + sharpe_ratio * x
plt.title("Efficient Frontier and Tangency Portfolio, Domestic + SPY")
plt.xlabel('Portfolio Risk (Standard Deviation)')
plt.ylabel('Portfolio Return')
plt.plot(x,y,c='purple',label='CAL',linewidth=2)
plt.legend()
plt.show()

# International Portfolio with SPY and Bitcoin

df=data[[('Adj Close', 'BANDHANBNK.NS'),
            ('Adj Close',       'BTC-USD'),
            ('Adj Close', 'COROMANDEL.NS'),
            ('Adj Close',   'DIVISLAB.NS'),
            ('Adj Close',     'FORTIS.NS'),
            ('Adj Close',   'GILLETTE.NS'),
            ('Adj Close',        'NCC.NS'),
            ('Adj Close',     'RECLTD.NS'),
            ('Adj Close',           'SPY'),]]
df.dropna(inplace=True)
print(f"df: \n{df}")

returns = np.log(df / df.shift(1)).dropna()
returns.columns = returns.columns.get_level_values(1)
mean_returns = returns.mean()
annualised_mean_returns=(1+mean_returns)**252 -1
print(f"annualised_mean_returns: \n{annualised_mean_returns}")

cov_matrix = returns.cov()
print(f"cov_matrix: {cov_matrix}")

variance=returns.var()
annualised_variance=variance*252
standard_deviation=np.sqrt(annualised_variance)
print(f"standard_deviation: \n {standard_deviation}")

# Minimum Variance Portfolio
min_variance=100
er=[]
sigma=[]
sharpe_ratio=0
min_weights=np.ones(9)
for i in range(0,100000):
    x,y,z=make_random_portfolio(9,annualised_mean_returns,cov_matrix)
    er.append(y)
    sigma.append(np.sqrt(z))
    if(z<min_variance):
        min_variance=z
        min_weights=x
        min_return=y
    sharpe_temp=(y-risk_free_rate)/np.sqrt(z)
    if(sharpe_temp>sharpe_ratio):
        sharpe_ratio=sharpe_temp
        sharpe_weights=x
        sharpe_return=y
        sharpe_variance=z

print(f"min_var_weights: {min_weights}")
print(f"min_var_return: {min_return}")
print(f"min_variance: {min_variance}")
print(f"max_sharpe_ratio: {sharpe_ratio}")
print(f"max_sharpe_weights: {sharpe_weights}")
print(f"max_sharpe_return: {sharpe_return}")
print(f"max_sharpe_variance: {sharpe_variance}")

# Efficient Frontier and Tangency Portfolio
plt.figure(figsize=(10,10))
plt.scatter(sigma,er)
plt.scatter(np.sqrt(sharpe_variance),sharpe_return,c="r")
plt.scatter(np.sqrt(min_variance),min_return, c='y')

plt.annotate('Tangency Portfolio', 
             xy=(np.sqrt(sharpe_variance),sharpe_return), 
             xytext=(np.sqrt(sharpe_variance) + 0.01,sharpe_return + 0.01),
             fontsize=10)

plt.annotate('Minimum Variance', 
             xy=(np.sqrt(min_variance),min_return), 
             xytext=(np.sqrt(min_variance) + 0.01,min_return + 0.01),
             fontsize=10)

x = np.linspace(0, max(sigma), 100)
y = risk_free_rate + sharpe_ratio * x
plt.title("Efficient Frontier and Tangency Portfolio, Domestic + SPY + BTC")
plt.xlabel('Portfolio Risk (Standard Deviation)')
plt.ylabel('Portfolio Return')
plt.plot(x,y,c='purple',label='CAL',linewidth=2)
plt.legend()
plt.show()
