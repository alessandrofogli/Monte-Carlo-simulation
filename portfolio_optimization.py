import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.dates as mdates

# Set a fixed seed for reproducibility
np.random.seed(42)

def download_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

def calculate_portfolio_statistics(returns):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return mean_returns, cov_matrix

def generate_random_portfolios(returns, num_portfolios):
    num_assets = len(returns.columns)
    results = np.zeros((3, num_portfolios))

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        
        results[0,i] = portfolio_return
        results[1,i] = portfolio_std_dev
        results[2,i] = portfolio_return / portfolio_std_dev

    return results

def plot_efficient_frontier(returns, num_portfolios, portfolios, optimal_portfolio):
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(portfolios[1,:], portfolios[0,:], c=portfolios[2,:], marker='o', cmap='viridis')
    ax.set_title('Efficient Frontier')
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Return')
    fig.colorbar(sc, label='Sharpe Ratio')
    ax.grid(True)

    ax.scatter(optimal_portfolio['volatility'], optimal_portfolio['return'], marker='*', color='r', s=200, label='Optimal Portfolio')
    ax.legend()

    return fig

def calculate_underwater_curve(cumulative_returns):
    high_water_mark = np.zeros_like(cumulative_returns)
    drawdown = np.zeros_like(cumulative_returns)

    for t in range(1, len(cumulative_returns)):
        high_water_mark[t] = np.maximum(high_water_mark[t-1], cumulative_returns[t])
        drawdown[t] = (1 + cumulative_returns[t]) / (1 + high_water_mark[t]) - 1
    
    return drawdown

def plot_underwater_curve(drawdown, dates):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dates, drawdown, color='red', alpha=0.5)
    ax.fill_between(dates, drawdown, 0, color='red', alpha=0.5)
    ax.set_title('Underwater Curve')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown')
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the date on x-axis
    fig.autofmt_xdate()  # Auto-format the x-axis date labels for better readability
    return fig


def main():
    st.title('Efficient Frontier Portfolio Analysis')

    tickers_input = st.text_input('Enter tickers (comma separated)', 'AAPL,GOOGL,MSFT,AMZN,META')
    tickers = [x.strip() for x in tickers_input.split(',')]

    start_date = st.date_input('Start Date', value=pd.to_datetime('2020-01-01'))
    end_date = st.date_input('End Date', value=pd.to_datetime('2023-01-01'))

    if st.button('Calculate'):
        data = download_data(tickers, start_date, end_date)
        returns = data.pct_change().dropna()
        dates = returns.index


        num_portfolios = 10000
        portfolios = generate_random_portfolios(returns, num_portfolios)

        max_sharpe_idx = np.argmax(portfolios[2])
        optimal_return = portfolios[0, max_sharpe_idx]
        optimal_volatility = portfolios[1, max_sharpe_idx]
        optimal_sharpe_ratio = portfolios[2, max_sharpe_idx]

        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)

        # Retrieve full names of tickers and create DataFrame
        ticker_infos = [yf.Ticker(ticker).info for ticker in tickers]
        tickers_full_names = [info['longName'] for info in ticker_infos]
        optimal_portfolio = pd.DataFrame({
            'tickers': tickers,
            'full_name': tickers_full_names,
            'weights': weights,
            'return': optimal_return,
            'volatility': optimal_volatility
        })
        
        st.header('Optimal Portfolio', divider='rainbow')
        st.write('Return:', round(optimal_return * 100, 2), '%')
        st.write('Volatility:', round(optimal_volatility * 100, 2), '%')
        st.write('Sharpe Ratio:', round(optimal_sharpe_ratio, 2))
        st.write(optimal_portfolio[['tickers', 'full_name', 'weights']].round(4))
        st.write('---')

        fig_efficient_frontier = plot_efficient_frontier(returns, num_portfolios, portfolios, optimal_portfolio)
        st.pyplot(fig_efficient_frontier)

        # Calculate cumulative returns of the optimal portfolio
        optimal_cumulative_returns = (returns * weights).sum(axis=1).cumsum()

        drawdown = calculate_underwater_curve(optimal_cumulative_returns)

        # Plot the underwater curve
        fig_underwater_curve = plot_underwater_curve(drawdown, optimal_cumulative_returns.index)
        st.pyplot(fig_underwater_curve)
        
        # Plot cumulative return of optimal portfolio
        fig_cumulative_returns = plt.figure(figsize=(10, 6))
        plt.plot(optimal_cumulative_returns.index, optimal_cumulative_returns.values, color='blue')
        plt.title('Cumulative Returns of Optimal Portfolio')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        st.pyplot(fig_cumulative_returns)

if __name__ == "__main__":
    main()
