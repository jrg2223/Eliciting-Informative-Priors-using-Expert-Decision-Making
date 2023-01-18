import pymc3 as pm
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import arviz as az
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, roc_curve
from IPython.core.pylabtools import figsize
import itertools

from scipy import stats
from scipy.stats import entropy

from sklearn.calibration import calibration_curve

#Data Loading and Cleaning

#Parole Board CSV found @ https://github.com/rcackerman/parole-hearing-data
data_upload = pd.read_csv('paroleboard.csv')

#data_upload.describe()
#data_upload.head()
data_upload = data_upload[~data_upload["interview decision"].str.startswith('*')]
count_nan_in_df = data_upload.isnull().sum().sum()
#print ('Count of NaN: ' + str(count_nan_in_df))
count_nan_in_df = data_upload.isnull().sum()
#print (count_nan_in_df)

def crime_count(data):
  nan_count = data.isnull().sum()
  crime_count = len(data) - nan_count
  return crime_count

crimes_class = data_upload[['crime 1 - class','crime 2 - class','crime 3 - class','crime 4 - class','crime 5 - class','crime 6 - class','crime 7 - class','crime 8 - class']]
crime_count = crimes_class.apply(crime_count, axis = 1)
data_upload['crime_count'] = crime_count

data_main = data_upload.dropna(axis = 1, thresh = 20000)
data_main.dtypes

#Get rid of postponed meetings and meetings where the outcome is just to change or keep the original release date.
data_main = data_main[~data_main["interview decision"].str.startswith('R')]
data_main = data_main[~data_main["interview decision"].str.startswith('OR')]

#Replace reponses with accepted and rejected parole (even with probabtion)
data_main['interview decision'] = data_main['interview decision'].str.replace('OPEN DATE','ACCEPT')
data_main['interview decision'] = data_main['interview decision'].str.replace('GRANTED','ACCEPT')
data_main['interview decision'] = data_main['interview decision'].str.replace('PAROLED','ACCEPT')
data_main['interview decision'] = data_main['interview decision'].str.replace('DENIED','REJECT')
data_main['interview decision'] = data_main['interview decision'].str.replace('NOT GRANTD','REJECT')

#Replace Ethnicity - Hispanic, Black, White, Other
data_main['race / ethnicity'] = data_main['race / ethnicity'].str.replace('AMERINDALSK','OTHER')
data_main['race / ethnicity'] = data_main['race / ethnicity'].str.replace('UNKNOWN','OTHER')
data_main['race / ethnicity'] = data_main['race / ethnicity'].str.replace('AMER IND/ALSK','OTHER')
data_main['race / ethnicity'] = data_main['race / ethnicity'].str.replace('ASIAN/PACIFIC','OTHER')



data_main['interview decision'].value_counts()

#Take information where it is only their initial appearance
data_main = data_main[data_main['parole board interview type']=="INITIAL"]
#Assume that FE means life sentence - forever so change to numeric value
data_main['aggregated maximum sentence'][data_main["aggregated maximum sentence"]=='FE'] = 1000

#only take the first line of sentence
data_main['agg_max_sent'] = data_main['aggregated maximum sentence'].str.split('-').str[0].astype('float')
data_main['agg_min_sent'] = data_main['aggregated minimum sentence'].str.split('-').str[0].astype('float')

#Create Categorical Variables
for col in ['sex', 'race / ethnicity', 'housing or interview facility', 'parole board interview type', 'interview decision', 'release type','housing/release facility', 'crime 1 - crime of conviction', 'crime 1 - class', 'crime 1 - county of commitment']:
    data_main[col] = data_main[col].str.replace(" ","")
    data_main[col] = data_main[col].str.replace("-","")
    data_main[col] = data_main[col].str.replace(">","")
    data_main[col] = data_main[col].str.replace("/","")
    data_main[col] = data_main[col].str.replace("<","")
    data_main[col] = data_main[col].str.replace("'","")
    data_main[col] = data_main[col].str.replace(":","")
    data_main[col] = data_main[col].astype('category')
    #data_main[col] = label_encoder.fit_transform(data_main[col])



data_main['birth date'].iloc[data_main['birth date']=='UNKNOWN'] = None
data_main['conditional release date'].iloc[data_main['conditional release date']=='NONE'] = None
data_main['maximum expiration date'].iloc[data_main['maximum expiration date']=='LIFE SENTENCE'] = None # only 1 life sentence value
data_main['parole me date'].iloc[data_main['parole me date']=='LIFE SENTENCE'] = None

#Create Date-Time variables
for col in ['parole board interview date','scrape date', 'birth date','release date','parole eligibility date','conditional release date','maximum expiration date','parole me date']:
    print(col)
    data_main[col] = pd.to_datetime(data_main[col])
    
from datetime import date

def calculateYears(birthDate, currentDate):
    years = currentDate.year - birthDate.year - ((currentDate.month, currentDate.day) < (birthDate.month, birthDate.day))
    return years

age = np.empty(len(data_main))
year_to_release_date = np.empty(len(data_main))
year_to_parole_date = np.empty(len(data_main))

#Create meaningful variables from date-time variables
for i in range(len(data_main)):
    age[i]= calculateYears(data_main['birth date'].iloc[i], data_main['parole board interview date'].iloc[i])
    year_to_release_date[i]= calculateYears(data_main['parole board interview date'].iloc[i], data_main['release date'].iloc[i])
    year_to_parole_date[i]= calculateYears(data_main['parole board interview date'].iloc[i], data_main['parole eligibility date'].iloc[i])
    
data_main['age'] = np.absolute(age)
data_main['num_years_to_release_date'] = year_to_release_date

#Check there are no null values/
data_main['age'].isnull().sum()
data_main['birth date'].isnull().sum() #124 before changing to date time
data_main['parole board interview date'].isnull().sum()

#Create Data set of wanted variables
data = data_main[['sex','race / ethnicity', 'age','interview decision','crime 1 - crime of conviction','crime 1 - class','num_years_to_release_date','num_years_to_parole_date','agg_max_sent','agg_min_sent','crime_count']]
#data = data_main[['sex', 'age','interview decision','crime 1 - crime of conviction','crime 1 - class','num_years_to_release_date','num_years_to_parole_date','agg_max_sent','agg_min_sent','crime_count']]
#data = data_main[['sex','race / ethnicity', 'age','interview decision','crime 1 - class','num_years_to_release_date','num_years_to_parole_date','agg_max_sent','agg_min_sent','crime_count']]

data.describe()
data.head()
data.dtypes

count_nan_in_df = data.isnull().sum().sum()
print ('Count of NaN: ' + str(count_nan_in_df))

count_nan_in_df = data.isnull().sum()
print (count_nan_in_df)
data = data.dropna(axis =0)



#Standardize numeric variables
means_sd =  pd.DataFrame(columns =["age",'num_years_to_release_date','num_years_to_parole_date','agg_max_sent','agg_min_sent','crime_count'] , index = ["means", "sd"])
for key in data[["age",'num_years_to_release_date','num_years_to_parole_date','agg_max_sent','agg_min_sent','crime_count']].keys():
    try:
        print("Standardizing "+key+".")
        means_sd[key]["means"] = np.mean(data[key])
        means_sd[key]["sd"] = np.std(data[key])
        data[key] = data[key] - np.mean(data[key])
        data[key] = data[key] / np.std(data[key])
        
    except:
        print("Predictor "+key+" cannot be standardized (probably a categorical variable).")
data_main['num_years_to_parole_date'] = year_to_parole_date

data.columns = ['sex', 'race', 'age','decision','crime1_conviction','crime1_class','num_years_to_release_date','num_years_to_parole_date','agg_max_sent','agg_min_sent','crime_count']

#Create crime 1 conviction variable with concatinated values
data['c1_con'] = data['crime1_conviction'].astype('str')

data.loc[data['c1_con'].str.contains('BURGLARY'), 'c1_con'] = 'BURGLARY'
data.loc[data['c1_con'].str.contains('ASSAULT'), 'c1_con'] = 'ASSAULT'
data.loc[data['c1_con'].str.contains('ASSULT'), 'c1_con'] = 'ASSAULT'
data.loc[data['c1_con'].str.contains('ASLT'), 'c1_con'] = 'ASSAULT'
data.loc[data['c1_con'].str.contains('STRANGULATION'), 'c1_con'] = 'ASSAULT'

data.loc[data['c1_con'].str.contains('GRANDLARCENY'), 'c1_con'] = 'GRANDLARCENY'
data.loc[data['c1_con'].str.contains('GRDLRCNY'), 'c1_con'] = 'GRANDLARCENY' #Any type of Grand larceny including crime
data.loc[data['c1_con'].str.contains('ROBBERY'), 'c1_con'] = 'ROBBERY'
data.loc[data['c1_con'].str.contains('DWI'), 'c1_con'] = 'DWI' #
data.loc[data['c1_con'].str.contains('DWAI'), 'c1_con'] = 'DWI' #DWI under drugs
data.loc[data['c1_con'].str.contains('MURDER'), 'c1_con'] = 'MURDER'
data.loc[data['c1_con'].str.contains('STALKING'), 'c1_con'] = 'STALKING'
data.loc[data['c1_con'].str.contains('FRAUD'), 'c1_con'] = 'FRAUD'
data.loc[data['c1_con'].str.contains('SEX'), 'c1_con'] = 'SEX' #Sexual related crimes
data.loc[data['c1_con'].str.contains('RAPE'), 'c1_con'] = 'SEX'
data.loc[data['c1_con'].str.contains('INCEST'), 'c1_con'] = 'SEX'
data.loc[data['c1_con'].str.contains('SX'), 'c1_con'] = 'SEX'
data.loc[data['c1_con'].str.contains('SODOMY'), 'c1_con'] = 'SEX'
data.loc[data['c1_con'].str.contains('MANSLAUGHTER'), 'c1_con'] = 'DEATH' #Death related 
data.loc[data['c1_con'].str.contains('HOMICIDE'), 'c1_con'] = 'DEATH' 
data.loc[data['c1_con'].str.contains('DEATH'), 'c1_con'] = 'DEATH' 
data.loc[data['c1_con'].str.contains('MANSLTER'), 'c1_con'] = 'DEATH' 
data.loc[data['c1_con'].str.contains('HOM'), 'c1_con'] = 'DEATH'
data.loc[data['c1_con'].str.contains('STALKING'), 'c1_con'] = 'STALKING' 
data.loc[data['c1_con'].str.contains('EAVESDROPPING'), 'c1_con'] = 'STALKING' 
data.loc[data['c1_con'].str.contains('SURVEIL'), 'c1_con'] = 'STALKING'
data.loc[data['c1_con'].str.contains('HARAS'), 'c1_con'] = 'STALKING'

data.loc[data['c1_con'].str.contains('KIDNAPP'), 'c1_con'] = 'KIDNAPPING'
data.loc[data['c1_con'].str.contains('ATTLURECHILDAFEL'), 'c1_con'] = 'KIDNAPPING'

data.loc[data['c1_con'].str.contains('CP'), 'c1_con'] = 'POSSESSION'
data.loc[data['c1_con'].str.contains('POS'), 'c1_con'] = 'POSSESSION'
data.loc[data['c1_con'].str.contains('CSC'), 'c1_con'] = 'SALE'
data.loc[data['c1_con'].str.contains('CS'), 'c1_con'] = 'SALE'
data.loc[data['c1_con'].str.contains('IDENTITY'), 'c1_con'] = 'FALSE'
data.loc[data['c1_con'].str.contains('FORGERY'), 'c1_con'] = 'FALSE'
data.loc[data['c1_con'].str.contains('IDENT'), 'c1_con'] = 'FALSE'
data.loc[data['c1_con'].str.contains('IDTHEFT'), 'c1_con'] = 'FALSE'

data.loc[data['c1_con'].str.contains('CONSPIRACY'), 'c1_con'] = 'CONSPIRACY'
data.loc[data['c1_con'].str.contains('BAILJUMPING'), 'c1_con'] = 'COURT'
data.loc[data['c1_con'].str.contains('TAMP'), 'c1_con'] = 'COURT'
data.loc[data['c1_con'].str.contains('PERJURY'), 'c1_con'] = 'COURT'
data.loc[data['c1_con'].str.contains('CONTEMPT'), 'c1_con'] = 'COURT'
data.loc[data['c1_con'].str.contains('TPT'), 'c1_con'] = 'COURT'
data.loc[data['c1_con'].str.contains('FALSEREPORT'), 'c1_con'] = 'COURT'

data.loc[data['c1_con'].str.contains('ARSON'), 'c1_con'] = 'ARSON'

data.loc[data['c1_con'].str.contains('[0-9]', regex = True), 'c1_con'] = 'OTHER'
data.loc[data['c1_con'].str.contains('ENTRPSECORRUPT'), 'c1_con'] = 'OTHER'
data.loc[data['c1_con'].str.contains('UNAUTPRACTPRO'), 'c1_con'] = 'OTHER'
data.loc[data['c1_con'].str.contains('LVSCNACCDNT'), 'c1_con'] = 'OTHER'
data.loc[data['c1_con'].str.contains('CRINJNARDRG'), 'c1_con'] = 'OTHER'
data.loc[data['c1_con'].str.contains('ATTMKTRRORTHRET'), 'c1_con'] = 'OTHER'
data.loc[data['c1_con'].str.contains('ATTENTRPSECORRUPT'), 'c1_con'] = 'OTHER'
data.loc[data['c1_con'].str.contains('ANIMALFIGHTING'), 'c1_con'] = 'OTHER'
data.loc[data['c1_con'].str.contains('LABORTRAFFICK'), 'c1_con'] = 'OTHER'
data.loc[data['c1_con'].str.contains('AGGFAMILYOFF'), 'c1_con'] = 'OTHER'
data.loc[data['c1_con'].str.contains('MISUSEFOODSTP'), 'c1_con'] = 'OTHER'
data.loc[data['c1_con'].str.contains('PROOBSPFMCHD'), 'c1_con'] = 'OTHER'
data.loc[data['c1_con'].str.contains('COMPTRESPASS'), 'c1_con'] = 'OTHER'

#Create all appropriate datasets
data_norace = data[['sex',  'age','decision','crime1_class','num_years_to_release_date','num_years_to_parole_date','agg_max_sent','agg_min_sent','crime_count','c1_con']]
data = data[['sex', 'race', 'age','decision','crime1_class','num_years_to_release_date','num_years_to_parole_date','agg_max_sent','agg_min_sent','crime_count','c1_con']]
data_nosex = data[[ 'race', 'age','decision','crime1_class','num_years_to_release_date','num_years_to_parole_date','agg_max_sent','agg_min_sent','crime_count','c1_con']]
data_noage = data[['sex', 'race','decision','crime1_class','num_years_to_release_date','num_years_to_parole_date','agg_max_sent','agg_min_sent','crime_count','c1_con']]
data_noage_numyearparole = data[['sex', 'race','decision','crime1_class','num_years_to_release_date','agg_max_sent','agg_min_sent','crime_count','c1_con']]

#Observe counts from crime class and ethnicity
black_DF = data.loc[data['race']== "BLACK"]
HISPANIC_DF = data.loc[data['race']== "HISPANIC"]
WHITE_DF = data.loc[data['race']== "WHITE"]
OTHER_DF = data.loc[data['race']== "OTHER"]

print(black_DF['crime1_class'].value_counts())
print(HISPANIC_DF['crime1_class'].value_counts())
print(WHITE_DF['crime1_class'].value_counts())
print(OTHER_DF['crime1_class'].value_counts())

print(black_DF.loc[data['decision']== "REJECT"]['crime1_class'].value_counts())
print(HISPANIC_DF.loc[data['decision']== "REJECT"]['crime1_class'].value_counts())
print(WHITE_DF.loc[data['decision']== "REJECT"]['crime1_class'].value_counts())
print(OTHER_DF.loc[data['decision']== "REJECT"]['crime1_class'].value_counts())

#Create dummy variables dataset
dummies = pd.get_dummies(data,drop_first=True)
dummies_norace = pd.get_dummies(data_norace,drop_first=True)
dummies_nosex= pd.get_dummies(data_nosex,drop_first=True)
dummies_noage= pd.get_dummies(data_noage,drop_first=True)
dummies_noage_numyearparole = pd.get_dummies(data_noage_numyearparole,drop_first=True)

#Observe Correlation Plots
g = sns.pairplot(data)
# Compute the correlation matrix
corr = data.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(
    corr,
    mask=mask,
    cmap=cmap,
    vmax=0.3,
    linewidths=0.5,
    cbar_kws={"shrink": 0.5},
    ax=ax,
);
