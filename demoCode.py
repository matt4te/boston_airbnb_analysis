
"""
Analysis of Airbnb Listings in Boston
"""



### SET UP SHOP ###


# Import required modules

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Locate project directories

projdir = 'C:\\Users\\mattf\\desktop\\job_stuff\\insight\\analysis_demo'
datadir = projdir + '\\boston'

os.chdir(projdir)



# Read in data files

df = pd.DataFrame()

datafiles = os.listdir(datadir)

for filename in datafiles:
    if filename.endswith('.csv'):
        file = pd.read_csv(os.path.join(datadir,filename))
        df = df.append(file)



# Inspect the data frame 
        
df.head()

print('Numbers of bathrooms: ', df.bathrooms.unique(), '\n')
print('Borough IDs: ', df.borough.unique(), '\n')
print('City IDs: ', df.city.unique(), '\n')
print('Country IDs: ', df.country.unique(), '\n')
print('Location IDs: ', df.location.unique(), '\n')
print('Listing Names: ', df.name.unique(), '\n')
print('Survey IDs: ', df.survey_id.unique())



# Filter the data frame columns

drop_columns = ['bathrooms', 'borough', 'city', 'country', 'location', 'name', 'survey_id']
data = df.drop(drop_columns,1)



# Set appropriate data types

print('\nData types: \n')
data.dtypes

data.host_id = data.host_id.astype('category')
data.last_modified = pd.to_datetime(data.last_modified)
data.neighborhood = data.neighborhood.astype('category')
data.room_id = data.room_id.astype('category')
data.overall_satisfaction = data.overall_satisfaction.astype('category')



# Check for Completeness

print('\nPercentange of Missing Values: \n')
data.isnull().sum() / len(data) * 100



# Create dictionaries for room IDs and common missing fields

room_av = data.groupby('room_id').mean()

room_accom_dict = {index: row['accommodates'] for (index, row) in room_av.iterrows()}
room_bed_dict = {index: row['bedrooms'] for (index, row) in room_av.iterrows()}
room_min_dict = {index: row['minstay'] for (index, row) in room_av.iterrows()}

dict(list(room_accom_dict.items())[0:2])



# Replace nan values with the mean value for that room_id

data.loc[data['accommodates'].isnull(),'accommodates'] = data.loc[data['accommodates'].isnull(),'room_id'].map(room_accom_dict)
data.loc[data['bedrooms'].isnull(),'bedrooms'] = data.loc[data['bedrooms'].isnull(),'room_id'].map(room_bed_dict)
data.loc[data['minstay'].isnull(),'minstay'] = data.loc[data['minstay'].isnull(),'room_id'].map(room_min_dict)


print('\nPercentange of Missing Values Still Remaining: \n')
data.isnull().sum() / len(data) * 100



# Final Cleanup 

data.loc[data['minstay'].isnull(),'minstay'] = 0

print('\nIncomplete Data Had',len(data),'data points')

dc = data.dropna(subset=['accommodates','bedrooms','host_id','room_type'])

print('\nThere are now',len(dc),'data points with complete data\n')

print('Check for any remaining nan values in dataset: \n')
dc.isnull().sum() / len(dc) * 100



# Add a season variable to data

dc['season'] = (dc.last_modified.dt.month%12 + 3)//3

dc.season = dc.season.astype('category')



### DATA SUMMARY ###


# What are the common accommodation options?

fig, (ax1,ax2,ax3) = plt.subplots(1,3)
fig.tight_layout()

ax1.hist(dc.accommodates,bins=int(max(dc.accommodates)))
ax1.set_title('Accommodation')
ax1.set_xlabel('# Accommodated')
ax1.set_ylabel('Count')

ax2.hist(dc.bedrooms,bins=int(max(dc.bedrooms))-1)
ax2.set_title('Bedrooms')
ax2.set_xlabel('# of Bedrooms')
ax2.set_ylabel('Count')

ax3.hist(dc.room_type)
ax3.set_title('Room Type')
ax3.set_xlabel('Room Type')
ax3.set_ylabel('Count')
xtickNames = plt.setp(ax3, xticklabels=['Private', 'Room', 'Shared'])
plt.setp(xtickNames)



# How expensive are the listings?

fig, (ax1,ax2) = plt.subplots(1,2)
fig.tight_layout()

ax1.hist(dc.price)
ax1.set_title('Price Frequency')
ax1.set_xlabel('Price (USD)')
ax1.set_ylabel('Count')

ax2.hist(dc.price,range=(dc.price.min(),1000),bins=10)
ax2.set_title('Truncated Price Frequency')
ax2.set_xlabel('Price (USD)')
ax2.set_ylabel('Count')



# Where are the listings located?

fig, ax1 = plt.subplots()

plt.hist(dc.neighborhood, bins=len(dc.neighborhood.unique()))
ax1.set_title('Neighborhood Frequency')
ax1.set_xlabel('Neighborhood')
ax1.set_ylabel('Count')
xtickNames = plt.setp(ax1,xticklabels=list(dc.neighborhood.unique()))
plt.setp(xtickNames, rotation=45, fontsize=6)



# Are people satisfied?

fig, ax1 = plt.subplots()

plt.hist(dc.overall_satisfaction, range=(0,5),bins=11)
ax1.set_title('Customer Satisfaction')
ax1.set_xlabel('Customer Rating')
ax1.set_ylabel('Count')





### VARIABLE RELATIONSHIPS ####


# Average price over time (2014-2017)

dc['date'] = dc.last_modified.dt.round('D')
dc = dc.drop('last_modified', axis=1)
dc['satis'] = dc.overall_satisfaction.astype('float')

ad = dc.groupby('date').mean()

ad.insert(ad.shape[1],
          'row_count',
          ad.index.value_counts().sort_index().cumsum())

ad = ad[ad.price < 300]
(m,b) = np.polyfit(ad.row_count,ad.price,1)

plt.plot(ad.row_count,ad.price,'r+')
plt.plot(ad.row_count,m*ad.row_count + b, 'b-')
axes = plt.gca()
axes.set_ylim([0,300])
plt.title('Average Prices over Time')
plt.xlabel('Date Index - not equidistant in time!')
plt.ylabel('Price (USD)')



# Customer satisfaction over time (2014-2017)

plt.plot(ad.row_count,ad.satis,'r+')
axes = plt.gca()
axes.set_ylim([0,5])
plt.title('Customer Satisfaction over Time')
plt.xlabel('Date Index - not equidistant in time!')
plt.ylabel('Customer Satisfaction')



# Price by accommodation capacity

ap = dc[dc.price < 1000]

(m2,b2) = np.polyfit(dc.accommodates,dc.price,1)

(m3,b3) = np.polyfit(dc.bedrooms,dc.price,1)

fig, (ax1,ax2) = plt.subplots(1,2,sharey=True)
fig.tight_layout()

ax1.plot(dc.accommodates,dc.price,'r+')
ax1.plot(dc.accommodates,m2*dc.accommodates + b2, 'b-')
ax1.set_title('Accommodation v Price')
ax1.set_xlabel('Accommodation Capacity')
ax1.set_ylabel('Price (USD)')

ax2.plot(dc.bedrooms,dc.price,'r+')
ax2.plot(dc.bedrooms,m3*dc.bedrooms + b3, 'b-')
ax2.set_title('Bedrooms v Price')
ax2.set_xlabel('# Bedrooms')
ax2.set_ylabel('Price (USD)')
axes = plt.gca()
axes.set_ylim([0,1000])
axes.set_xlim([0,8])
  
             

# Price by Room Type

fig, (ax1,ax2,ax3) = plt.subplots(1,3,sharey=True)

fig.suptitle('Prices by Room Type')

ax1.boxplot(dc.loc[dc.room_type=='Entire home/apt'].price)
ax1.set_ylabel('Price (USD)')
ax1.set_xlabel('Entire Home/Apt')

ax2.boxplot(dc.loc[dc.room_type=='Private room'].price)
ax2.set_xlabel('Private Room')

ax3.boxplot(dc.loc[dc.room_type=='Shared room'].price)
ax3.set_xlabel('Shared Room')
axes = plt.gca()
axes.set_ylim([0,1000])



# Satisfaction by Room Type

dcn = dc.dropna()

fig, (ax1,ax2,ax3) = plt.subplots(1,3,sharey=True)

fig.suptitle('Satisfaction by Room Type')

ax1.boxplot(dcn.loc[dcn.room_type=='Entire home/apt'].overall_satisfaction)
ax1.set_ylabel('Customer Satisfaction')
ax1.set_xlabel('Entire Home/Apt')

ax2.boxplot(dcn.loc[dcn.room_type=='Private room'].overall_satisfaction)
ax2.set_xlabel('Private Room')

ax3.boxplot(dcn.loc[dcn.room_type=='Shared room'].overall_satisfaction)
ax3.set_xlabel('Shared Room')



# Could do the same thing by neighborhood...

dc.isnull().sum() / len(dc) * 100





### Predicting Customer Satisfaction - Logistic Regression ###


# Set up a logistic regression, predicting customer satisfaction, using only data where we have satisfaction

dcC = dc.dropna()
dcC['overall_satisfaction'] = (dcC.overall_satisfaction * 2).astype(int)

y = dcC.overall_satisfaction

X = dcC.drop(['overall_satisfaction','satis'], axis=1)



# Recode categorical variables to integers

cat_cols = ['host_id', 'neighborhood', 'room_id', 'room_type','date']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in X.columns.values:
    for col in cat_cols:
        dle = X[col]
        le.fit(dle.values)
        X[col] = le.transform(X[col])



# Split data into train/test groups

from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2)



# Construct logistic model for customer satisfaction

from sklearn import linear_model
sat_reg = linear_model.LogisticRegression(C=1)

sat_reg.fit(X_train,y_train,sample_weight=None)



# Make predictions using the testing set

sat_pred = sat_reg.predict(X_test)



# Print the model accuracy score and classification

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print('\nModel Accuracy Score:', accuracy_score(y_test,sat_pred))

print('\nModel Classifcation Report\n\n',classification_report(y_test, sat_pred))



# Visualize model effectiveness with confusion matrix

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tickmarks = np.arange(len(dcC.overall_satisfaction.unique()))
    plt.xticks(tickmarks, [0, 2, 4, 5, 6, 7, 8, 9, 10]) 
    plt.yticks(tickmarks, [0, 2, 4, 5, 6, 7, 8, 9, 10])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm = confusion_matrix(y_test,sat_pred)

plt.figure()
plot_confusion_matrix(cm)

print(cm)





# Fill in original dataset by predicting satisfaction values for listings

dcN = dc[pd.isnull(dc['overall_satisfaction'])]
dcN = dcN.drop(['overall_satisfaction','satis'], axis=1)

dcN2 = dcN.copy()
cat_cols = ['host_id', 'neighborhood', 'room_id', 'room_type','date']

le2 = LabelEncoder()

for col in dcN2.columns.values:
    for col in cat_cols:
        dle2 = dcN2[col]
        le2.fit(dle2.values)
        dcN2[col] = le2.transform(dcN2[col])

sat_fill = sat_reg.predict(dcN2)

dcN['overall_satisfaction'] = sat_fill

dcC = dcC.drop('satis', axis=1)
dataC = pd.concat([dcC,dcN],axis=0)

dataC.host_id = dataC.host_id.astype('category')
dataC.neighborhood = dataC.neighborhood.astype('category')
dataC.overall_satisfaction = dataC.overall_satisfaction.astype('category')
dataC.room_id = dataC.room_id.astype('category')
dataC.room_type = dataC.room_type.astype('category')
dataC.date = pd.to_datetime(dataC.date)

print('\nRemaining Null Values:', dataC.isnull().sum() / len(dc) * 100)

print('\nDataset now has', len(dataC), 'values')



### Price Prediction - Linear Regression ###

# Inspect the price variable, remove outliers

plt.boxplot(dataC.price)
plt.title('Airbnb Prices')
plt.ylabel('Price (USD)')

print('\n Investigating the quantiles of the data:', dataC.price.describe(),'\n')

dataCp = dataC[dataC.price < 250]

print('\n Filtering out the top %.2f' % ((len(dataC) - len(dataCp)) / len(dataC) * 100), '% of the price data leaves', len(dataCp),'data points')



# Set up a supervised regression, predicting customer satisfcation, for data where we have satisfaction

yl = dataCp.price

Xp = dataCp.drop('price', axis=1)



# Recode categorical variables to integers

cat_cols = ['host_id', 'neighborhood', 'room_id', 'room_type','date','overall_satisfaction']

le3 = LabelEncoder()

for col in Xp.columns.values:
    for col in cat_cols:
        dle3 = Xp[col]
        le3.fit(dle3.values)
        Xp[col] = le3.transform(Xp[col])
        

print('\nThere are',len(Xp.host_id.unique()),'unique host IDS and',len(Xp.room_id.unique()),'room IDs')
        
    
       
# Recode integer categorical variables to binary variables, when the order of integers is meaningless 
        
cat_cols2 = ['neighborhood','room_type', 'season']
        
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse=False)

for col in cat_cols2:
    denc = Xp[[col]]
    enc.fit(denc)
    temp = enc.transform(Xp[[col]])
    temp = pd.DataFrame(temp,columns=[(col+'_'+str(i)) for i in denc[col].value_counts().index])
    temp = temp.set_index(Xp.index.values)
    Xp = pd.concat([Xp,temp],axis=1)
    
print('\nSize of the new data frame is:',Xp.shape)



# Split data into train/test groups

drop_cols = ['host_id', 'room_id','room_type','neighborhood','season','latitude','longitude']
Xl = Xp.drop(drop_cols,axis=1)

(Xl_train, Xl_test, yl_train, yl_test) = train_test_split(Xl, yl, test_size=0.2)



# Construct linear model for customer satisfaction

p_reg = linear_model.LinearRegression(normalize=True)

p_reg.fit(Xl_train,yl_train,sample_weight=None)



# Make predictions using the testing set

p_pred = p_reg.predict(Xl_test)



# Print the coefficients, mean squared error, and explained variance score

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

print('\nMean absolute error: %.2f' % mean_absolute_error(yl_test, p_pred),'dollars')

print('\nVariance score: %.2f' % r2_score(yl_test, p_pred))



# Visualize the model effectiveness

(mp,bp) = np.polyfit(yl_test,pd.Series(p_pred),1)

plt.scatter(yl_test,pd.Series(p_pred), c='r', marker='+')
plt.title('Test Data Prediction Error')
plt.xlabel('Actual Listed Price')
plt.ylabel('Predicted Price')
plt.plot(yl_test, mp*yl_test + bp, 'b-')
plt.plot(yl_test, yl_test, 'k-')





### So where should I look to live? - Feature Selection ###


# Well, what am I interested in?

def desirable(row):
    if row['price'] < 50 and row['overall_satisfaction'] >= 4:
        val = 1
    elif row['price'] < 100 and row['overall_satisfaction'] >= 3:
        val = 2
    else:
        val = 3
    return val

dataC['desired'] = dataC.apply(desirable, axis=1)

dataC.desired.value_counts()



# Set up a model for important feature selection

yf = dataC.desired.astype('category')

drop_feats = ['overall_satisfaction', 'price', 'host_id', 'room_id', 'latitude', 'longitude', 'date', 'desired']
Xf = dataC.drop(drop_feats, axis=1)



# Set up recoded train/test datasets for decision trees

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

norm_cols = ['accommodates', 'bedrooms', 'minstay', 'reviews']
Xf[norm_cols] = scaler.fit_transform(Xf[norm_cols])

cat_cols = ['neighborhood', 'room_type']

lef = LabelEncoder()

for col in Xf.columns.values:
    for col in cat_cols:
        dlef = Xf[col]
        lef.fit(dlef.values)
        Xf[col] = lef.transform(Xf[col])
        

cat_cols2 = ['neighborhood','room_type', 'season']
        
encf = OneHotEncoder(sparse=False)

for col in cat_cols2:
    dencf = Xf[[col]]
    encf.fit(dencf)
    temp = encf.transform(Xf[[col]])
    temp = pd.DataFrame(temp,columns=[(col+'_'+str(i)) for i in dencf[col].value_counts().index])
    temp = temp.set_index(Xf.index.values)
    Xf = pd.concat([Xf,temp],axis=1)

Xf = Xf.drop(['neighborhood', 'room_type', 'season'], axis=1)


(Xf_train, Xf_test, yf_train, yf_test) = train_test_split(Xf, yf, test_size=0.2)



# Fit an extra trees classifier

from sklearn.ensemble import ExtraTreesClassifier

forest = ExtraTreesClassifier(n_estimators=50)

forest.fit(Xf_train, yf_train)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
indices = np.argsort(importances)[::-1]



# Plot the feature importance ranks

feats = {} 
for feature, importance in zip(Xf.columns, importances):
    feats[feature] = importance 

impp = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})

impp.sort_values(by='Gini-importance', ascending=False).plot(kind='bar', color='r', rot=90, yerr=std[indices], align='center')
plt.title('Feature importances')
plt.xlabel('Feature Name')
plt.ylabel('Relative Importance')
plt.xticks(fontsize=8)



### What is the influence of the important features?

import seaborn as sns



# What does the number of reviews say about desirability? 

#... Having reviews is good, but the most reviewed place won't be the best for me 

fig, (ax2) = plt.subplots()

rd = sns.regplot(x='reviews',y='desired', data=dataC, size=7)
rd.set(xlabel='# of Reviews',ylabel='Desirability',title='Desirabilty by # of Reviews')

ax2 = sns.factorplot(x='desired', y='reviews', data=dataC, kind='bar', size=7.5, aspect=1.5, ci=95)
ax2.set(xlabel='Desirability Class', ylabel='# of Reviews (95% confidence)', title='Number of Reviews by Desirability Class')

       
       
# What does the minimum stay say about desirability?

#... nothing useful 

fig, (ax2) = plt.subplots()

md = sns.regplot(x='minstay',y='desired', data=dataC)
md.set(xlabel='# of Reviews',ylabel='Desirability',title='Desirability by Minimum Stay Requirement')

md = sns.factorplot(x='desired', y='minstay', data=dataC, kind='bar', size=7.5, aspect=1.5)
md.set(xlabel='Desirability Class', ylabel='Minimum Stay Required (95% confidence)',title='Minimum Stay Requirement by Desirability Class')
       


# What types of rooms are most desirable?

#... It is unlikely I will end up in an entire home/apartment 

fig, (ax2) = plt.subplots()

td = sns.boxplot(x='room_type',y='desired', data=dataC)
td.set(xlabel='Desirability Class', ylabel='Minimum Stay Required', title='Desirability by Room Type')

ax2 = sns.factorplot(x='room_type',y='desired', kind='bar', data=dataC,size=7.5,aspect=1.5)
ax2.set(xlabel='Room Type', ylabel='Desirability (95% confidence)',title='Desirability by Room Type')
     



# How is desirability related to accommodation capacity?

#... Rather, I'll end up sharing a place with a couple of people

fig, (ax2) = plt.subplots()

ad = sns.regplot(x='desired',y='accommodates',data=dataC)
ad.set(xlabel='Desirability Class',ylabel='Accommodation Capacity',title='Accommodation Capacity by Desirability Class')

ax2 = sns.factorplot(x='desired',y='accommodates',data=dataC,kind='bar',size=7.5,aspect=1.5)
ax2.set(xlabel='Desirability Class',ylabel='Accommodation Capacity',title='Accommodation Capacity by Desirability Class')


# set up an HTML search for jupyter
