# SmartFolio - Portfolio Optimisation App

SmartFolio is a Streamlit-based application that helps investors and analysts optimize their portfolios using both classical and quantum machine learning techniques. The app retrieves historical stock data for S&P 500 companies, computes financial metrics, and applies various optimisation strategies to help you diversify your portfolio. Users can choose between classical optimisation methods (such as maximizing Sharpe ratio, maximizing returns, minimizing volatility, or targeting specific risk/return levels) and quantum approaches using D-Wave's hybrid solvers.

## Features

- **Data Retrieval:**  
  Fetches S&P 500 company data from Wikipedia and downloads historical stock data from Yahoo Finance.

- **Financial Metrics:**  
  Calculates expected returns and sample covariance using the PyPortfolioOpt library to build an Efficient Frontier.

- **Optimisation Strategies (Classical):**

  - Maximize Sharpe Ratio
  - Maximize Returns
  - Minimize Volatility
  - Target Volatility
  - Target Returns

- **Optimisation Strategies (Quantum):**  
  Utilizes D-Wave’s LeapHybridCQMSampler to perform portfolio optimisation with quantum solvers for:

  - Maximizing Sharpe Ratio
  - Maximizing Returns
  - Minimizing Volatility

- **Detailed Analysis:**  
  Displays detailed stock information (e.g., industry, CEO, country, revenue) and provides interactive visualizations (line charts, scatter plots, pie charts) for further insights.

- **Interactive Interface:**  
  A sidebar navigation allows you to choose between Classical and Quantum machine learning backends, and various optimisation strategies.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Jnan-py/Portfolio-Optimisation.git
   cd Portfolio-Optimisation
   ```

2. **Create a Virtual Environment (Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Application:**

   ```bash
   streamlit run opt.py
   ```

2. **Select Backend & Strategy:**

   - Use the sidebar to choose between **Classical Machine Learning** and **Quantum Machine Learning**.
   - Depending on your choice, select the desired optimisation strategy (e.g., Maximize Sharpe, Maximize Returns, Minimize Volatility, Target Volatility, or Target Returns).

3. **Input Parameters:**

   - Enter the number of tickers, choose the tickers from the S&P 500 list, and input your investment budget and risk rate.
   - For target-based strategies, input the target volatility or target returns as needed.

4. **View Optimisation Results:**
   - The app will display an optimised portfolio allocation table along with key metrics (Sharpe Ratio, Annual Volatility, Expected Returns) and interactive graphs of the stock price data and allocation pie charts.
   - In the quantum backend, D-Wave’s sampler is used to solve the portfolio optimisation problem with a constrained quadratic model.

## Project Structure

```
smartfolio/
│
├── opt.py                 # Main Streamlit application
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies

```

## Technologies Used

- **Streamlit:** Interactive web application framework.
- **Pandas & NumPy:** Data manipulation and numerical computations.
- **Yahoo Finance (yfinance):** Historical stock data retrieval.
- **PyPortfolioOpt:** Financial optimisation, risk models, and efficient frontier construction.
- **Prophet:** Time series forecasting and visualization (if used).
- **Plotly:** Interactive plotting and data visualization.
- **SciPy:** Numerical optimisation routines.
- **D-Wave System & Dimod:** Quantum optimisation using LeapHybridCQMSampler.
- **Streamlit Option Menu:** For intuitive sidebar navigation.
