# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 09:59:42 2022

@author: paulw
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import QuantLib as ql

def exploratory_plots(mbs_data):
  for v in ['note_rate', 'wam']:
    mbs_data[v].hist(weights=mbs_data['curr_bal'],figsize=(8,5))
    if v=="note_rate":
      plt.xlabel("Note Rate (%)")
    elif v=="wam":
      plt.xlabel("WAM (months)")
    plt.ylabel("Balance ($)")
    plt.show()

def forecast_cashflows(mbs_data, cpr):
  forecast = pd.DataFrame({"t": range(0,max(mbs_data['wam'].astype(int)+1)),
                          "Balance": 0, "int_pymts": 0, "prin_pymts": 0})

  for i in range(len(mbs_data)):
    loan = mbs_data.iloc[i]
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
    
    forecast['Balance'] += np.pad(balance,(0,360-term+age),constant_values=0)
    forecast['int_pymts'] += np.pad(int_pymts,(1,360-term+age),constant_values=0)
    forecast['prin_pymts'] += np.pad(prin_pymts,(1,360-term+age),constant_values=0)

  forecast['bal_frac'] = forecast['Balance']/forecast.iloc[0,1]
  return forecast

def months_to_runoff(cf, pct):
  return min(cf[cf['bal_frac']<pct]['t'])

def plot_balances(cf, title_label):
  runoff_50pct = months_to_runoff(cf, 0.5)
  runoff_75pct = months_to_runoff(cf, 0.25)
  runoff_95pct = months_to_runoff(cf, 0.05)

  asof_date = np.datetime64('2022-10')

  print("\n")
  print("Months to 50% runoff: ", runoff_50pct)
  print("Months to 95% runoff: ", runoff_95pct)
  print("Date of 50% runoff: ", asof_date + np.timedelta64(runoff_50pct,'M'))
  print("Date of 95% runoff: ", asof_date + np.timedelta64(runoff_95pct,'M'))
  print("\n")

  fig, ax = plt.subplots()
  bal_forecast = cf['Balance']/0.95
  bal_forecast.plot(figsize=(13,8),kind='bar',xticks=np.arange(0,361,60),rot=0,width=1)
  ax.set_xticklabels([asof_date+np.timedelta64(m,'M') for m in np.arange(0,361,60)])
  plt.title(title_label)
  plt.xlabel("Forecast date")
  plt.ylabel("Balance remaining ($)")
  plt.axvline(x=runoff_50pct,color='C1',label='50% runoff',linestyle='dashed')
  plt.axvline(x=runoff_95pct,color='C3',label='95% runoff',linestyle='dashed')
  plt.legend()
  plt.show()

def plot_runoff(cf, title_label):

  asof_date = np.datetime64('2022-10')
  runoff_forecast = cf.loc[1:,'prin_pymts']/0.95

  print("\n")
  print("1-year average runoff: ", np.round(np.average(runoff_forecast[0:12]/1000000000),2), "billion")
  print("5-year average runoff: ", np.round(np.average(runoff_forecast[0:60]/1000000000),2), "billion")
  print("\n")

  fig, ax = plt.subplots()
  runoff_forecast.plot(figsize=(13,8),kind='line',xticks=np.arange(0,361,60))
  ax.set_xticklabels([asof_date+np.timedelta64(m,'M') for m in np.arange(0,361,60)])
  plt.title(title_label)
  plt.xlabel("Forecast date")
  plt.ylabel("Monthly runoff ($)")
  plt.legend(["Principal repayment"])
  plt.show()
  

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
