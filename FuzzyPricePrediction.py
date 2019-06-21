# -*- coding: utf-8 -*-
"""
Created on Mon May 13 18:24:29 2019

@author: Matthew Pryse
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Remove outliers of the output variable from the datasets (both training and test), 
#Load the data files
energy_prices_testing_file = 'Testing_Data.csv'
energy_prices_training_file = 'Training_Data.csv'
energy_prices_testing_data = pd.read_csv(energy_prices_testing_file)
energy_prices_training_data = pd.read_csv(energy_prices_training_file)
#ccm_threshold value is used to evaluate signficant items from correlation matrix
ccm_threshold = .55
#temperature at time instant t T(t)
#total demand at time instant t D(t)
#time instant t is composed of a subset of the set M={T(t-2),T(t-1), T(t), D(t-2), D(t-1), D(t)}. 
#denoted by P(t+1) forecasting value of the Recommended Retail Price (RRP) of electricity at the next time instant t+1

#Function to return position of outlier values below 25% and above 75% percentiles
def identify_outliers(series_data):
    #Define 25% quartile
    Q1=np.percentile(series_data, 25) ; # the value 25 is fixed for every problem;
    print ('Q1:{}'.format(Q1))
    #Define 75% quartile
    Q3=np.percentile(series_data, 75); # the value 75 is fixed for every problem; 
    print ('Q3:{}'.format(Q3))
    range=[Q1-1.5*(Q3-Q1),Q3+1.5*(Q3-Q1)];    
    position=np.concatenate((np.where(series_data>range[1]),np.where(series_data<range[0])),axis=1) 
    return position

plt.plot(energy_prices_testing_data['P(t+1)'], 'o')
plt.show()
outlier_positions = identify_outliers(energy_prices_testing_data['P(t+1)'])
print ('Test values to remove:{}'.format(energy_prices_testing_data.iloc[outlier_positions[0]]))
energy_prices_testing_data = energy_prices_testing_data.drop(outlier_positions[0], axis=0)
energy_prices_testing_data.reset_index()

outlier_positions = identify_outliers(energy_prices_training_data['P(t+1)'])
print ('Training values to remove:{}'.format(energy_prices_training_data.iloc[outlier_positions[0]]))
energy_prices_training_data = energy_prices_training_data.drop(outlier_positions[0], axis=0)
energy_prices_training_data.reset_index()

plt.plot(energy_prices_testing_data['P(t+1)'], 'o')
plt.show()

# Variable selection by using Correlation Coefficient Matrix
energy_prices_testing_ccm=np.corrcoef(energy_prices_testing_data.values, rowvar=False)
energy_prices_training_ccm=np.corrcoef(energy_prices_training_data.values, rowvar=False)
testing_index = np.where(abs(energy_prices_testing_ccm[6,0:7]) > ccm_threshold, True, False)
training_index = np.where(abs(energy_prices_training_ccm[6,0:7]) > ccm_threshold, True, False)
energy_prices_testing_data
result = np.where(np.logical_and(testing_index, training_index))[0]
print('Result{}'.format(result))
energy_prices_testing_data = energy_prices_testing_data.iloc[:, result]
energy_prices_training_data = energy_prices_training_data.iloc[:, result]

print(energy_prices_training_data.columns)

demand_minus_1_training = energy_prices_training_data['D(t-1)']
demand_training = energy_prices_training_data['D(t)']
price_training = energy_prices_training_data['P(t+1)']

demand_minus_1_testing = energy_prices_testing_data['D(t-1)']
demand_testing = energy_prices_testing_data['D(t)']
price_testing = energy_prices_testing_data['P(t+1)']

demand_minus_1_training.hist(label='D{t-1}')
plt.show()
demand_training.hist(label='D{t}')
plt.show()
price_training.hist(label='P{t+1}')
plt.show()

demand_minus_1_testing.hist(label='D{t-1}')
plt.show()
demand_testing.hist(label='D{t}')
plt.show()
price_testing.hist(label='P{t+1}')
plt.show()
    
#Design fuzzy subsets for linguistic variables and the membership functions of input variables and output variable
#demand {Very Low, Low, Medium, High, Very High}
demand = ctrl.Antecedent(np.arange(1000, 8000, 1), 'demand')
demand_minus_1 = ctrl.Antecedent(np.arange(1000, 8000, 1), 'demand_minus_1')
#price {low, medium, high, very high}
price = ctrl.Consequent(np.arange(5, 65, 1), 'price')

demand['Very Low'] = fuzz.trimf (demand.universe, [1000, 1200, 2500])
demand['Low'] = fuzz.trimf(demand.universe, [1200, 2500, 4500])
demand['Medium'] = fuzz.trimf(demand.universe, [2500, 4500, 5500])
demand['High'] = fuzz.trimf(demand.universe, [4500, 5500, 6500])
demand['Very High'] = fuzz.trimf(demand.universe, [5500, 6500, 8000])

demand_minus_1['Very Low'] = fuzz.trimf(demand_minus_1.universe, [1000, 1200, 2500])
demand_minus_1['Low'] = fuzz.trimf(demand_minus_1.universe, [1200, 2500, 4500])
demand_minus_1['Medium'] = fuzz.trimf(demand_minus_1.universe, [2500, 4500, 5500])
demand_minus_1['High'] = fuzz.trimf(demand_minus_1.universe, [4500, 5500, 6500])
demand_minus_1['Very High'] = fuzz.trimf(demand_minus_1.universe, [5500, 6500, 8000])

price['Low'] = fuzz.trimf(price.universe, [5, 10, 15])
price['Medium'] = fuzz.trimf(price.universe, [10, 15, 30])
price['High'] = fuzz.trimf(price.universe, [15, 30, 40])
price['Very High'] = fuzz.trimf(price.universe, [30, 40, 60])

demand_minus_1.view() # show all the three membership functions
demand_minus_1['Medium'].view() #Highlight the 'average' membership function

demand.view() # show all the three membership functions
demand['Medium'].view() #Highlight the 'average' membership function

price.view() # show all the three membership functions
price['Medium'].view() #Highlight the 'average' membership function

rule0 = ctrl.Rule(demand['Low'] | demand_minus_1['Low'], price['Low'])
rule1 = ctrl.Rule(demand['Low'] | demand_minus_1['Very Low'], price['Low'])
rule2 = ctrl.Rule(demand['Low'] | demand_minus_1['Medium'], price['Medium'])
rule3 = ctrl.Rule(demand['Low'] | demand_minus_1['Very High'], price['Medium'])
rule2 = ctrl.Rule(demand['Very Low'] | demand_minus_1['Medium'], price['Low'])
#rule3 = ctrl.Rule(demand['Very Low'] | demand_minus_1['High'], price['Medium'])
#rule4 = ctrl.Rule(demand['Very Low'] | demand_minus_1['Very High'], price['High'])

#rule5 = ctrl.Rule(demand['Low'] | demand_minus_1['Very Low'], price['Low'])
#rule6 = ctrl.Rule(demand['Low'] | demand_minus_1['Low'], price['Low'])
#rule7 = ctrl.Rule(demand['Low'] | demand_minus_1['Medium'], price['Low'])
#rule8 = ctrl.Rule(demand['Low'] | demand_minus_1['High'], price['Medium'])
#rule9 = ctrl.Rule(demand['Low'] | demand_minus_1['Very High'], price['High'])
                  
rule10 = ctrl.Rule(demand['Medium'] | demand_minus_1['Medium'], price['Medium'])
#rule11 = ctrl.Rule(demand['Medium'] | demand_minus_1['Medium'], price['Medium'])
#rule12 = ctrl.Rule(demand['Medium'] | demand_minus_1['High'], price['Medium'])
rule13 = ctrl.Rule(demand['Medium'] | demand_minus_1['Very High'], price['High'])

rule14 = ctrl.Rule(demand['High'] | demand_minus_1['Low'], price['Medium'])
#rule15 = ctrl.Rule(demand['High'] | demand_minus_1['Medium'], price['Medium'])
rule16 = ctrl.Rule(demand['High'] | demand_minus_1['High'], price['Medium'])
#rule17 = ctrl.Rule(demand['High'] | demand_minus_1['Very High'], price['Medium'])

rule18 = ctrl.Rule(demand['Very High'] | demand_minus_1['Medium'], price['Medium'])
rule19 = ctrl.Rule(demand['Very High'] | demand_minus_1['High'], price['High'])
#rule20 = ctrl.Rule(demand['Very High'] | demand_minus_1['Very High'], price['Very High'])

price_ctrl = ctrl.ControlSystem([rule1, rule2, rule0, rule10, rule13,
                                 rule14, rule16,rule18,rule19])
pricing = ctrl.ControlSystemSimulation(price_ctrl)
rule6.view()
                
# Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
# Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
def run_fuzzy(demand_series_data, demand_minus_1_series_data):
    System_outputs=np.zeros(len(demand_series_data.values), dtype=np.float64)
    for i in range (len(demand_minus_1_series_data.values)):
        pricing.input['demand'] = demand_training.values[i]
        pricing.input['demand_minus_1'] = demand_minus_1_training.values[i]
        # Crunch the numbers
        pricing.compute()
        price = pricing.output['price']
        System_outputs[i]=print(price)
    return System_outputs
        

#Do a fuzzy run with training dataset
SystemOutput = run_fuzzy(demand_training, demand_minus_1_training)
N=len(price_training.values)
input=np.arange(1,N+1, dtype=float)/5   #np.arange(1,11,dtype=Float) can generate a sequence [1,2,3,...,100];
TargetOutput=2+np.sin(input) # Expected outputs
SystemOutput= TargetOutput+0.15*np.random.random(input.shape)  #Real outputs (as supposed) act as the system outputs
RErr=np.sum(np.absolute(TargetOutput-SystemOutput)/np.absolute(TargetOutput))/N # Calculate RErr                 
print('The Average Relative Error Value is', RErr)   # Return and print the error value

#Do a fuzzy run with test dataset
SystemOutput = run_fuzzy(demand_testing, demand_minus_1_testing)
N=len(price_testing.values)
input=np.arange(1,N+1, dtype=float)/5   #np.arange(1,11,dtype=Float) can generate a sequence [1,2,3,...,100];
TargetOutput=2+np.sin(input) # Expected outputs
SystemOutput= TargetOutput+0.15*np.random.random(input.shape)  #Real outputs (as supposed) act as the system outputs
RErr=np.sum(np.absolute(TargetOutput-SystemOutput)/np.absolute(TargetOutput))/N # Calculate RErr                 
print('The Average Relative Error Value is', RErr)   # Return and print the error value

print (pricing.output['price'])
price.view()
plt.show()
plt.plot(input, TargetOutput)
plt.plot(input, SystemOutput)
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('')
plt.legend(['Target Outputs', 'System Outputs'])
plt.show()





