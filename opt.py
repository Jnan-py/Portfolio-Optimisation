import streamlit as st
import pandas as pd 
import yfinance as yf 
import datetime
from dimod import *
from streamlit_option_menu import option_menu
from scipy.optimize import minimize
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import plotly.express as px
from dwave.samplers import *
from dwave.system import *


def main():
    st.title('PortFoliO Optimisation')

    with st.sidebar:
        st.title("Portfolio Optimisation")
        with st.expander("Select your ML backend",expanded=True):
            page = st.selectbox(
            label="Types",
            options=["Classical Machine Learning", "Quantum Machine Learning"]
            )

    def load_df():
        tk_list = pd.read_html("https://en.wikipedia.org/wiki/list_of_S%26P_500_companies",header=0)
        df= tk_list[0]
        return df

    def get_metrics(tkr,close_df):
        for i in tkr:
            data = yf.download(i,start,end)
            close_df[i] = data['Adj Close']
            
        close_df = close_df.fillna(method="ffill")
        mu = expected_returns.mean_historical_return(close_df)
        s = risk_models.sample_cov(close_df)
        ef = EfficientFrontier(mu,s)
        return mu,s,ef

    def get_details():
        with st.expander("**Details of Stock**",expanded=False):
            tr = st.selectbox("Choose the Stock Ticker",tkr)
            if tr:
                st.session_state.ticker=True
                if st.session_state.ticker:
                    resp = yf.Ticker(tr)
                    info = resp.info
                    name = info.get('longName')
                    country = info.get('country')
                    ceo = info.get('companyOfficers')[0]['name']
                    currency = info.get('currency')
                    summ = info.get('longBusinessSummary')
                    ind = info.get('industry')
                    website = info.get('website')
                    rev = info.get('totalRevenue')

                    st.subheader(name)
                    st.write(f'**Industry** : {ind}')
                    st.write(F'**Chief Executive Officer**: {ceo}')
                    st.write(f'**Country** : {country}')
                    st.write(f'**Currency** : {currency}')
                    st.write(f'**Total Revenue** : {rev}')
                    st.write(f'**Summary** : {summ}')

                    
                    if st.button("View Price Table and Graph"):
                        st.subheader("Price Table")
                        stock = yf.download(tr,start,end)
                        stock.reset_index(inplace=True)
                        st.write(stock)
                        
                        fig=go.Figure()
                        fig.add_trace(go.Scatter(x=stock['Date'],y=stock['Open'],name="Stock Open Price"))
                        fig.add_trace(go.Scatter(x=stock['Date'],y=stock['Close'],name="Stock Close Price "))
                        fig.layout.update(title_text="Stock Data Graph",xaxis_rangeslider_visible=True)
                        fig.update_layout(xaxis_title="Price",yaxis_title="Date")        
                        st.plotly_chart(fig)

    def maxim_sharpe(mu,s,ef,budget):
        weights1 = ef.max_sharpe()
        clnd = ef.clean_weights()

        main_wghts = [clnd[i] for i in clnd]
        av_keys = [i for i in clnd.keys() if clnd[i]!=0]
        weights = [clnd[i] for i in av_keys]
        budgets = [i*budget for i in weights]

        fin_df = pd.DataFrame()
        fin_df["Tickers"] = av_keys
        fin_df["Weights Assigned"] = weights
        fin_df["Investment Amount"] = budgets

        st.subheader("Optimised Table")
        st.write(fin_df)

        return main_wghts,av_keys,weights

    def min_volatility(ef,budget):
        weights1 = ef.min_volatility()
        clnd = ef.clean_weights()

        main_wghts = [clnd[i] for i in clnd]
        av_keys = [i for i in clnd.keys() if clnd[i]!=0]
        weights = [clnd[i] for i in av_keys]
        budgets = [i*budget for i in weights]

        fin_df = pd.DataFrame()
        fin_df["Tickers"] = av_keys
        fin_df["Weights Assigned"] = weights
        fin_df["Investment Amount"] = budgets

        st.subheader("Optimised Table")
        st.write(fin_df)

        return main_wghts,av_keys,weights

    def max_returns(ef,budget):
        weights1 = ef.max_quadratic_utility()
        clnd = ef.clean_weights()

        main_wghts = [clnd[i] for i in clnd]
        av_keys = [i for i in clnd.keys() if clnd[i]!=0]
        weights = [clnd[i] for i in av_keys]
        budgets = [i*budget for i in weights]

        fin_df = pd.DataFrame()
        fin_df["Tickers"] = av_keys
        fin_df["Weights Assigned"] = weights
        fin_df["Investment Amount"] = budgets

        return main_wghts,av_keys,weights,fin_df

    def targ_vol(ef,budget,target_vol):
        weights1 = ef.efficient_risk(target_vol)
        clnd = ef.clean_weights()

        main_wghts = [clnd[i] for i in clnd]
        av_keys = [i for i in clnd.keys() if clnd[i]!=0]
        weights = [clnd[i] for i in av_keys]
        budgets = [i*budget for i in weights]

        fin_df = pd.DataFrame()
        fin_df["Tickers"] = av_keys
        fin_df["Weights Assigned"] = weights
        fin_df["Investment Amount"] = budgets

        st.subheader("Optimised Table")
        st.write(fin_df)

        return main_wghts,av_keys,weights

    def targ_ret(ef,budget,target_ret):
        weights1 = ef.efficient_return(target_ret)
        clnd = ef.clean_weights()

        main_wghts = [clnd[i] for i in clnd]
        av_keys = [i for i in clnd.keys() if clnd[i]!=0]
        weights = [clnd[i] for i in av_keys]
        budgets = [i*budget for i in weights]

        fin_df = pd.DataFrame()
        fin_df["Tickers"] = av_keys
        fin_df["Weights Assigned"] = weights
        fin_df["Investment Amount"] = budgets

        st.subheader("Optimised Table")
        st.write(fin_df)

        return main_wghts,av_keys,weights

    def std(weights,s):
        var = weights.T @ s @ weights
        return np.sqrt(var)

    def exp_r(weights,mu):
        return np.sum(mu*weights)

    def sharpe(weights,mu,s,risk_rate):
        return (exp_r(weights,mu) - risk_rate)/std(weights,s)

    def further_details(main_wghts,av_keys,weights,mu,s,rrate,close_df,tkr):
        with st.expander("**FURTHER DETAILS**",expanded=True):
            st.subheader("Metrics")
            st.write(
                    f"**Sharpe Ratio** : {round(sharpe(np.array(main_wghts),mu,s,rrate),5)}"
                )
            st.write(
                    f"**Annual Volatility** : {round(std(np.array(main_wghts),s),5)}"
                )
            st.write(
                    f"**Expected Returns** : {round(exp_r(np.array(main_wghts),mu),5)}"
                )

            st.subheader("Graphs")
            fig1 = go.Figure()
            for i in tkr:
                fig1.add_trace(go.Scatter(x=close_df.index,y=close_df[i],name=i))

            fig1.layout.update(title_text="Stock Data Graph",xaxis_rangeslider_visible=True)
            fig1.update_layout(xaxis_title="Date",yaxis_title="Price")
            st.plotly_chart(fig1)

            pie_chart = px.pie(names = av_keys,values = weights,title = "Pie Chart")
            st.plotly_chart(pie_chart)
        
    def sum(a):
        l = 0
        for i in a:
            l+=i
        return l

    def q_max_sharpe(tkr,n,mu,sigma,budget):
        cqm = ConstrainedQuadraticModel()
        bin_vars = [Binary(i) for i in range(n)]
        obj1 = -quicksum(mu[i]*bin_vars[i] for i in range(n))
        obj2 = 0.834*quicksum(bin_vars[i]*bin_vars[j]*sigma[i][j] for i in range(n) for j in range(n))
        obj = obj1+obj2
        cqm.set_objective(obj)

        sampler = LeapHybridCQMSampler(token="your-dwave-token")
        samples = sampler.sample_cqm(cqm,time_limit=10)
        sol_lst = []
        for i in samples:
            if i not in sol_lst:
                sol_lst.append(i)

        shp_ratio = []
        wghts = []
        for i in sol_lst:
            dct=i
            lsta = [i for i in dct.values()]
            n_wts1=[0 for i in range(len(lsta))]
            for i,wt in enumerate(lsta):
                if mu[i]>=0:
                    n_wts1[i]=wt*mu[i]
            tot = sum(n_wts1)
            nwts = [i/tot for i in n_wts1]
            wghts.append(nwts)
            shp_ratio.append(sharpe(np.array(nwts),mu,sigma,0.02))

        sol_idx = shp_ratio.index(max(shp_ratio))
        fin_wts = wghts[sol_idx]

        ndct = dict()
        for i,v in enumerate(tkr):
            ndct[v]=fin_wts[i]

        tkrs = [tkr for tkr in ndct.keys() if ndct[tkr] != 0]
        wts = [ndct[i] for i in tkrs]
        budgets = [i*budget for i in wts]

        fin_df = pd.DataFrame()
        fin_df["Tickers"] = tkrs
        fin_df["Weights Assigned"] = wts
        fin_df["Investment Amount"] = budgets

        st.subheader("Optimised Table")
        st.write(fin_df)

        return fin_wts,tkrs,wts

    def q_max_returns(global_min_vol,tkr,n,mu,sigma,budget):
        cqm1  = ConstrainedQuadraticModel()
        bin_vars = [Binary(i) for i in range(n)]
        obj1 = -quicksum(mu[i]*bin_vars[i] for i in range(n))
        obj2 = 0.834*quicksum(bin_vars[i]*bin_vars[j]*sigma[i][j] for i in range(n) for j in range(n))
        cqm1.set_objective(obj1)
        cqm1.add_constraint(obj2 , '==' , rhs = global_min_vol)

        sampler = LeapHybridCQMSampler(token="your-dwave-token")
        samp = sampler.sample_cqm(cqm1)
        solll = []
        for i in samp:
            if i not in solll:
                solll.append(i)
            
        rtrns = []
        wghts = []
        for i in solll:
            dct=i
            lsta = [i for i in dct.values()]
            n_wts1=[0 for i in range(len(lsta))]
            for i,wt in enumerate(lsta):
                if mu[i]>=0:
                    n_wts1[i]=wt*mu[i]
            tot = sum(n_wts1)
            nwts = [i/tot for i in n_wts1]
            wghts.append(nwts)
            rtrns.append(exp_r(np.array(nwts),mu))

        if len(rtrns)!=1:
            if min(rtrns)==max(rtrns):
                if max(rtrns).astype(int) == min(rtrns).astype(int):
                    rtrns.remove(max(rtrns))
                    wghts.pop(rtrns.index(max(rtrns)))
        
        sol_idx_r = rtrns.index(max(rtrns))
        fin_wts = wghts[sol_idx_r]

        ndct = dict()
        for i,v in enumerate(tkr):
            ndct[v]=fin_wts[i]

        tkrs = [tkr for tkr in ndct.keys() if ndct[tkr] != 0]
        wts = [ndct[i] for i in tkrs]
        budgets = [i*budget for i in wts]

        fin_df = pd.DataFrame()
        fin_df["Tickers"] = tkrs
        fin_df["Weights Assigned"] = wts
        fin_df["Investment Amount"] = budgets

        st.subheader("Optimised Table")
        st.write(fin_df)

        return fin_wts,tkrs,wts

    def q_min_vol(global_max_return,tkr,n,mu,sigma,budget,close_df):
        cqm2  = ConstrainedQuadraticModel()
        bin_vars = [Binary(i) for i in range(n)]
        obj1 = -quicksum(mu[i]*bin_vars[i] for i in range(n))
        obj2 = 0.834*quicksum(bin_vars[i]*bin_vars[j]*sigma[i][j] for i in range(n) for j in range(n))
        cqm2.set_objective(obj2)
        cqm2.add_constraint(-obj1 , '==' , rhs = global_max_return)

        sampler = LeapHybridCQMSampler(token="your-dwave-token")
        samp = sampler.sample_cqm(cqm2)
        soll = []
        for i in samp:
            if i not in soll:
                soll.append(i) 

        stds = []
        wghts = []
        for i in soll:
            dct=i
            lsta = [i for i in dct.values()]
            n_wts1=[0 for i in range(len(lsta))]
            for i,wt in enumerate(lsta):
                if np.std(close_df[tkr[i]])>=0:
                    n_wts1[i]=wt*np.std(close_df[tkr[i]])
            tot = sum(n_wts1)
            nwts = [i/tot for i in n_wts1]
            wghts.append(nwts)
            stds.append(std(np.array(nwts),sigma))

        if len(stds)!=1:
            if min(stds)!=max(stds):
                if min(stds).astype(int) == max(stds).astype(int):
                    stds.remove(min(stds))
                    wghts.pop(stds.index(min(stds)))
        
        sol_idx = stds.index(min(stds))           
        fin_wts = wghts[sol_idx]

        ndct = dict()
        for i,v in enumerate(tkr):
            ndct[v]=fin_wts[i]

        tkrs = [tkr for tkr in ndct.keys() if ndct[tkr] != 0]
        wts = [ndct[i] for i in tkrs]
        budgets = [i*budget for i in wts]

        fin_df = pd.DataFrame()
        fin_df["Tickers"] = tkrs
        fin_df["Weights Assigned"] = wts
        fin_df["Investment Amount"] = budgets

        st.subheader("Optimised Table")
        st.write(fin_df)

        return fin_wts,tkrs,wts


    if page == "Classical Machine Learning":
        st.header("Classical Machine Learning")
        n = st.number_input("Enter the number of tickers : ",min_value=0,step=1)
        strategy = st.selectbox("Select Strategy", options=[
            "Maximize Sharpe", 
            "Maximize Returns", 
            "Minimize Volatility",
            "Target Volatility",
            "Target Returns"
            ])
        budget = st.number_input("Enter the budget : ",min_value = 0.00)
        rrate = st.number_input("Enter the risk rate : ",min_value=0.00)

        df = load_df()
        close_df = pd.DataFrame()

        if strategy == "Maximize Sharpe":
            tkr = st.multiselect("Tickers",df["Symbol"].unique(),max_selections=n)
            if tkr and budget and rrate and len(tkr)==n: 
                end = datetime.datetime.today()
                start = end - datetime.timedelta(days=(3650))
                get_details()
                mu,s,ef = get_metrics(tkr,close_df)
                
                if st.button("Get Optimisation Results"):
                    st.header("Maximizing Sharpe")
                    main_wghts,av_keys,weights = maxim_sharpe(mu,s,ef,budget)
                    further_details(main_wghts=main_wghts,av_keys=av_keys,weights=weights,mu=mu,s=s,rrate=rrate,close_df=close_df,tkr=tkr)

        elif strategy == "Maximize Returns":
            tkr = st.multiselect("Tickers",df["Symbol"].unique(),max_selections=n)
            if tkr and budget and rrate and len(tkr)==n: 
                end = datetime.datetime.today()
                start = end - datetime.timedelta(days=(3650))
                get_details()
                mu,s,ef = get_metrics(tkr,close_df)

                if st.button("Get Optimisation Results"):
                    st.header("Maximizing Returns")
                    main_wghts,av_keys,weights,fin_df = max_returns(ef,budget)
                    st.subheader("Optimised Table")
                    st.write(fin_df)
                    further_details(main_wghts=main_wghts,av_keys=av_keys,weights=weights,mu=mu,s=s,rrate=rrate,close_df=close_df,tkr=tkr)

        elif strategy == "Minimize Volatility":
            tkr = st.multiselect("Tickers",df["Symbol"].unique(),max_selections=n)
            if tkr and budget and rrate and len(tkr)==n: 
                end = datetime.datetime.today()
                start = end - datetime.timedelta(days=(3650))
                get_details()
                mu,s,ef = get_metrics(tkr,close_df)

                if st.button("Get Optimisation Results"):
                    st.header("Minimize Volatility")
                    main_wghts,av_keys,weights = min_volatility(ef,budget)
                    further_details(main_wghts=main_wghts,av_keys=av_keys,weights=weights,mu=mu,s=s,rrate=rrate,close_df=close_df,tkr=tkr)

        elif strategy == "Target Volatility":
            target_vol = st.number_input("Enter the target volatility : ",min_value=0.00)
            tkr = st.multiselect("Tickers",df["Symbol"].unique(),max_selections=n)
            if tkr and budget and rrate and len(tkr)==n and target_vol: 
                end = datetime.datetime.today()
                start = end - datetime.timedelta(days=(3650))
                mu,s,ef = get_metrics(tkr,close_df)
                global_min_vol = np.sqrt(1 / np.sum(np.linalg.pinv(s)))
                if target_vol < global_min_vol:    
                    st.warning(f"The given target volatility should be greater than the minimum volatility {global_min_vol}")

                elif target_vol > global_min_vol:    
                    get_details()
                    if st.button("Get Optimisation Results"):
                        st.header("Target Volatility")
                        main_wghts,av_keys,weights = targ_vol(ef,budget,target_vol)
                        further_details(main_wghts=main_wghts,av_keys=av_keys,weights=weights,mu=mu,s=s,rrate=rrate,close_df=close_df,tkr=tkr)

        elif strategy == "Target Returns":
            target_ret = st.number_input("Enter the target returns : ",min_value=0.00)
            tkr = st.multiselect("Tickers",df["Symbol"].unique(),max_selections=n)
            if tkr and budget and rrate and len(tkr)==n and target_ret: 
                end = datetime.datetime.today()
                start = end - datetime.timedelta(days=(3650))
                mu,s,ef = get_metrics(tkr,close_df)
                m_wghts,av,wghts,fin_df = max_returns(ef,budget)
                global_max_return = round(exp_r(m_wghts,mu),5)

                if target_ret > global_max_return:    
                    st.warning(f"The given target return should be lesser than the maximum returns {global_max_return}")

                elif target_ret < global_max_return:    
                    get_details()
                    if st.button("Get Optimisation Results"):
                        st.header("Target Returns")
                        mu,s,ef = get_metrics(tkr,close_df)
                        main_wghts,av_keys,weights = targ_ret(ef,budget,target_ret)
                        further_details(main_wghts=main_wghts,av_keys=av_keys,weights=weights,mu=mu,s=s,rrate=rrate,close_df=close_df,tkr=tkr)

    elif page == "Quantum Machine Learning":
        st.header("Portfolio Optimisation using Quantum Machine Learning")
        n = st.number_input("Enter the number of tickers : ",min_value=0,step=1)
        strategy = st.selectbox("Select Strategy", options=[
            "Maximize Sharpe", 
            "Maximize Returns", 
            "Minimize Volatility",
            ])
        budget = st.number_input("Enter the budget : ",min_value = 0.00)
        rrate = st.number_input("Enter the risk rate : ",min_value=0.00)

        df = load_df()
        close_df = pd.DataFrame()

        if strategy == "Maximize Sharpe":
            tkr = st.multiselect("Tickers",df["Symbol"].unique(),max_selections=n)
            if tkr and budget and rrate and len(tkr)==n: 
                end = datetime.datetime.today()
                start = end - datetime.timedelta(days=(3650))
                get_details()
                mu,s,ef = get_metrics(tkr,close_df)

                sigma = np.zeros((n,n))
                for i in tkr:
                    for j in tkr:
                        sigma[tkr.index(i)][tkr.index(j)]=s[i][j]
                                    
                if st.button("Get Optimisation Results"):
                    st.header("Maximizing Sharpe")
                    main_wghts,av_keys,weights = q_max_sharpe(tkr,n,mu,sigma,budget)
                    further_details(main_wghts=main_wghts,av_keys=av_keys,weights=weights,mu=mu,s=s,rrate=rrate,close_df=close_df,tkr=tkr)

        elif strategy == "Maximize Returns":
            tkr = st.multiselect("Tickers",df["Symbol"].unique(),max_selections=n)
            if tkr and budget and rrate and len(tkr)==n: 
                end = datetime.datetime.today()
                start = end - datetime.timedelta(days=(3650))
                get_details()
                mu,s,ef = get_metrics(tkr,close_df)
                global_min_vol = np.sqrt(1 / np.sum(np.linalg.pinv(s)))

                sigma = np.zeros((n,n))
                for i in tkr:
                    for j in tkr:
                        sigma[tkr.index(i)][tkr.index(j)]=s[i][j]

                if st.button("Get Optimisation Results"):
                    st.header("Maximizing Returns")
                    main_wghts,av_keys,weights = q_max_returns(global_min_vol,tkr,n,mu,sigma,budget)
                    further_details(main_wghts=main_wghts,av_keys=av_keys,weights=weights,mu=mu,s=s,rrate=rrate,close_df=close_df,tkr=tkr)

        elif strategy == "Minimize Volatility":
            tkr = st.multiselect("Tickers",df["Symbol"].unique(),max_selections=n)
            if tkr and budget and rrate and len(tkr)==n: 
                end = datetime.datetime.today()
                start = end - datetime.timedelta(days=(3650))
                get_details()
                mu,s,ef = get_metrics(tkr,close_df)
                m_wghts,av,wghts,fin_df = max_returns(ef,budget)
                global_max_return = round(exp_r(m_wghts,mu),5)

                sigma = np.zeros((n,n))
                for i in tkr:
                    for j in tkr:
                        sigma[tkr.index(i)][tkr.index(j)]=s[i][j]

                if st.button("Get Optimisation Results"):
                    st.header("Minimize Volatility")
                    main_wghts,av_keys,weights = q_min_vol(global_max_return,tkr,n,mu,sigma,budget,close_df)
                    further_details(main_wghts=main_wghts,av_keys=av_keys,weights=weights,mu=mu,s=s,rrate=rrate,close_df=close_df,tkr=tkr)


if __name__ == '__main__':
    main()
