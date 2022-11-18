# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 09:59:42 2022

@author: paulw
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import QuantLib as ql
    
    
def forecast_pandi(loan, cpr):
  note_rate = loan['note_rate']/1200
  coupon = loan['coupon']/1200
  term = loan['term'].astype(int)
  age = loan['age'].astype(int)
  initial_bal = loan['curr_bal']

  smmfactors = np.power(1-cpr/1200,range(1,term-age+1))
  amfactors = ((1+note_rate)**term - (1+note_rate)**(range(age+1,term+1)))/((1+note_rate)**term - 1)
  initial_amfactor = ((1+note_rate)**term - (1+note_rate)**age)/((1+note_rate)**term - 1)
  amfactors = smmfactors*amfactors/initial_amfactor
  amfactors = np.insert(amfactors,0,1)
  balance = initial_bal*amfactors
  int_pymts = balance[0:-1]*coupon
  prin_pymts = balance[0:-1] - balance[1:]
  
  return balance, int_pymts, prin_pymts


def forecast_cashflows(mbs_data, cpr):
  forecast = pd.DataFrame({"t": range(0,max(mbs_data['wam'].astype(int)+1)),
                          "Balance": 0, "int_pymts": 0, "prin_pymts": 0})

  for i in range(len(mbs_data)):
    loan = mbs_data.iloc[i]
    term = loan['term'].astype(int)
    age = loan['age'].astype(int)
    
    balance, int_pymts, prin_pymts = forecast_pandi(loan, cpr)
    
    forecast['Balance'] += np.pad(balance,(0,360-term+age),constant_values=0)
    forecast['int_pymts'] += np.pad(int_pymts,(1,360-term+age),constant_values=0)
    forecast['prin_pymts'] += np.pad(prin_pymts,(1,360-term+age),constant_values=0)

  forecast['bal_frac'] = forecast['Balance']/forecast.iloc[0,1]
  return forecast


def months_to_runoff(cf, pct):
  return min(cf[cf['bal_frac']<pct]['t'])
  

class TreasuryParCurve():
  def __init__(self, maturities, rates, asof_date):

    convention = ql.Unadjusted
    day_count = ql.ActualActual(ql.ActualActual.Bond)
    ql.Settings.instance().evaluationDate = asof_date
    self.asof_date = asof_date

    bonds = []
    for r, m in zip(rates, maturities):
      # ql.Schedule(effectiveDate, terminationDate, tenor, calendar, convention, terminationDateConvention, rule, endOfMonth)
      schedule = ql.Schedule(asof_date, asof_date + m, ql.Period(ql.Semiannual), ql.UnitedStates(), convention, convention, ql.DateGeneration.Backward, True)
      # ql.FixedRateBondHelper(price, settlementDays, faceAmount, schedule, coupons, dayCounter, paymentConv=Following)
      helper_base = ql.FixedRateBondHelper(ql.QuoteHandle(ql.SimpleQuote(100)), 0, 100.0, schedule, [r/100.0], day_count, convention,)
      bonds.append(helper_base)

    self.treasury_curve = ql.PiecewiseLogCubicDiscount(asof_date, bonds, day_count)

  def calculate_df(self, period_range):
    return [self.treasury_curve.discount(self.asof_date+ql.Period(m, ql.Months)) for m in period_range]

  def fwd_rates(self, period_range):
    asof_date = self.asof_date
    fwd_dates = [asof_date+ql.Period(m, ql.Months) for m in period_range]
    fwd_rates = [self.treasury_curve.forwardRate(d, d+ql.Period(1,ql.Months), ql.ActualActual(ql.ActualActual.Bond), ql.Simple).rate() for d in fwd_dates]
    return fwd_rates


def static_pricing(mbs_data, cpr, discount_factors):

  prices = []

  for i in range(len(mbs_data)):
    loan = mbs_data.iloc[i]
    term = loan['term'].astype(int)
    age = loan['age'].astype(int)
    initial_bal = loan['curr_bal']

    balance, int_pymts, prin_pymts = forecast_pandi(loan, cpr)

    price = 100*np.sum(discount_factors[0:term-age]*(int_pymts + prin_pymts))/initial_bal
    prices.append(price)

  price_frame = pd.DataFrame({"price": prices, "balance": mbs_data['curr_bal']})
  return price_frame


def static_gap(mbs_data, cpr, funding_rates):

  gap_forecast = pd.DataFrame({"t": range(0,max(mbs_data['wam'].astype(int)+1)),
                          "int_received": 0, "funding_paid": 0})

  for i in range(len(mbs_data)):
    loan = mbs_data.iloc[i]
    term = loan['term'].astype(int)
    age = loan['age'].astype(int)
    
    balance, int_pymts, prin_pymts = forecast_pandi(loan, cpr)
    funding_paid = -balance[0:-1]*funding_rates[0:term-age]/12

    gap_forecast['int_received'] += np.pad(int_pymts,(1,360-term+age),constant_values=0)
    gap_forecast['funding_paid'] += np.pad(funding_paid,(1,360-term+age),constant_values=0)

  gap_forecast['gap'] = gap_forecast['int_received'] + gap_forecast['funding_paid']
  return gap_forecast[1:]

def exploratory_plots(mbs_data):
  for v in ['note_rate', 'wam']:
    mbs_data[v].hist(weights=mbs_data['curr_bal'],figsize=(8,5))
    if v=="note_rate":
      plt.xlabel("Note Rate (%)")
    elif v=="wam":
      plt.xlabel("WAM (months)")
    plt.ylabel("Balance ($)")
    plt.show()


def data_summary(mbs_data):
    data_summary = pd.DataFrame([str(len(mbs_data)),
                              str("${:,.2f}".format(sum(mbs_data['curr_bal'])/10**12)) + "T",
                              str(round(100*sum(mbs_data['curr_bal'])/(2.69*10**12),2))], 
                            index=['# of CUSIPs in sample','Sample balance','Percent of total'])
    
    print("\n")
    print(data_summary.to_string(header=False))
    print("\n")
    print(mbs_data[['note_rate','coupon','age','term','curr_bal']].describe())

def plot_par_rates(term_points, rates, asof_date):
  yield_curve = pd.DataFrame({"Term": term_points, "Rate": rates})
  yield_curve.plot(x="Term", y="Rate",figsize=(8,5))
  plt.title("Treasury par yield curve as-of " + str(asof_date))
  plt.ylabel("Rate (%)")
  plt.show()
  
def plot_fwd_rates(fwd_rates, asof_date):
  fwd_frame = 100*pd.DataFrame({"Forward rates": fwd_rates})
  fwd_frame.plot(figsize=(8,5))
  plt.title("1-month forward rates as-of " + str(asof_date))
  plt.xlabel("Period")
  plt.ylabel("Rate (%)")
  plt.show()

def plot_balances(cf, title_label, asof_date):
  runoff_50pct = months_to_runoff(cf, 0.5)
  runoff_75pct = months_to_runoff(cf, 0.25)
  runoff_95pct = months_to_runoff(cf, 0.05)

  print("\n")
  print("Months to 50% runoff: ", runoff_50pct)
  print("Months to 95% runoff: ", runoff_95pct)
  print("Date of 50% runoff: ", asof_date + ql.Period(runoff_50pct,ql.Months))
  print("Date of 95% runoff: ", asof_date + ql.Period(runoff_95pct,ql.Months))
  print("\n")
  
  date_label = np.datetime64(str(asof_date.year()) + "-" + str(asof_date.month()))

  fig, ax = plt.subplots()
  bal_forecast = cf['Balance']/0.95
  bal_forecast.plot(figsize=(13,8),kind='bar',xticks=np.arange(0,361,60),rot=0,width=1)
  ax.set_xticklabels([date_label + np.timedelta64(m,'M') for m in np.arange(0,361,60)])
  plt.title(title_label)
  plt.xlabel("Forecast date")
  plt.ylabel("Balance remaining ($)")
  plt.axvline(x=runoff_50pct,color='C1',label='50% runoff',linestyle='dashed')
  plt.axvline(x=runoff_95pct,color='C3',label='95% runoff',linestyle='dashed')
  plt.legend()
  plt.show()


def plot_runoff(cf, title_label, asof_date):

  runoff_forecast = cf.loc[1:,'prin_pymts']/0.95

  print("\n")
  print("1-year average runoff: ", np.round(np.average(runoff_forecast[0:12]/1000000000),2), "billion")
  print("5-year average runoff: ", np.round(np.average(runoff_forecast[0:60]/1000000000),2), "billion")
  print("\n")
  
  date_label = np.datetime64(str(asof_date.year()) + "-" + str(asof_date.month()))

  fig, ax = plt.subplots()
  runoff_forecast.plot(figsize=(13,8),kind='line',xticks=np.arange(0,361,60))
  ax.set_xticklabels([date_label + np.timedelta64(m,'M') for m in np.arange(0,361,60)])
  plt.title(title_label)
  plt.xlabel("Forecast date")
  plt.ylabel("Monthly runoff ($)")
  plt.legend(["Principal repayment"])
  plt.show()
  
def analysis_summary(mbs_prices):
  wavg_price = np.sum(mbs_prices['price']*mbs_prices['balance'])/np.sum(mbs_prices['balance'])
  total_bal = sum(mbs_prices['balance'])/0.95

  outframe = pd.DataFrame([str(np.round(wavg_price,2)), 
                         "$" + str(np.round(total_bal/10**9,1))+"B",
                         "$" + str(np.round(wavg_price/100*total_bal/10**9,1))+"B",
                         "$" + str(np.round((wavg_price/100-1)*total_bal/10**9,1))+"B"],
                        index=['Average price', 'Balance outstanding', 'Estimated value', 'Capital shortfall'])

  print("\n")
  print(outframe.to_string(header=False))
  print("\n")

  mbs_prices['price'].hist(weights=mbs_prices['balance'],figsize=(8,5))
  plt.xlabel("Price")
  plt.ylabel("Balance")
  plt.title("Distribution of RMBS prices")
  plt.show()
  
def plot_durations(mbs_prices, duration):
  wavg_duration = np.sum(mbs_prices['balance']*duration)/np.sum(mbs_prices['balance'])
  portfolio_duration = wavg_duration*np.sum(mbs_prices['balance']*mbs_prices['price']/100)/10000

  print("\n")
  print("Average duration: \t", np.round(wavg_duration,2))
  print('Portfolio DV01: \t', "$" + str(np.round(portfolio_duration/10**9,2)) + "B")
  print("\n")

  duration.hist(weights=mbs_prices['balance'], figsize=(8,5))
  plt.xlabel("Duration")
  plt.ylabel("Balance")
  plt.title("Distribution of portfolio duration")
  plt.show()
  
def plot_gap(interest_gap, asof_date):
  print("\n")
  print("Cumulative net income: ", round(interest_gap['gap'].cumsum().iloc[-1]/10**9,1), "billion")
  print("\n")

  annual_gap = interest_gap.rolling(12).sum()[11::12]
  annual_period = [y for y in range(1,31)]
  date_label = np.datetime64(str(asof_date.year()) + "-" + str(asof_date.month()))

  fig, ax = plt.subplots(figsize=(13,8))
  ax.bar(annual_period,annual_gap['int_received'])
  ax.bar(annual_period,annual_gap['funding_paid'])
  ax.plot(annual_period,annual_gap['gap'], color='C3')
  ax.axhline(lw=1, color='black')
  ax.set_xticklabels([date_label + np.timedelta64(y*12,'M') for y in range(1,31)])
  ax.set_title("Annual interest rate gap")
  ax.set_xlabel("Period")
  ax.set_ylabel("Interest gap")
  plt.show()
  