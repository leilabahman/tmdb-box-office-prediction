# -*- coding: utf-8 -*-
"""
Created on Apr 15, 2019
"""

# importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from wordcloud import WordCloud
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import ast
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

# reading the train data
train = pd.read_csv('C:/Users/Lavesh/.PyCharm2018.2/config/scratches/IMDB_Data/train.csv')

# json columns in data set
dict_columns = ['belongs_to_collection', 'genres', 'production_companies',
                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']

# converting into dictionary
def text_to_dict(df):
    for column in dict_columns:
        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )
    return df
train = text_to_dict(train)

# dropped features
train.drop(['id','imdb_id','poster_path','overview','status'],axis=1,inplace=True)

# belongs_to_collection feature
train['collection_name'] = train['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)
train['collection_name'], uniques = pd.factorize(train['collection_name'])
#Data Visualization
train['has_collection'] = train['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)
sns.catplot(x='has_collection', y='revenue', data=train);
plt.title('Compare Revenue for movie with and without collection');
plt.ylabel('Revenue (100 million dollars)')
train=train.drop(['has_collection'],axis =1)
#Drop original column
train=train.drop(['belongs_to_collection'],axis =1)

#homepage feature
train['has_homepage'] = 0
train.loc[train['homepage'].isnull() == False, 'has_homepage'] = 1
train=train.drop(['homepage'],axis =1)
#Data Visualization
sns.catplot(x='has_homepage', y='revenue', data=train);
plt.title('Compare Revenue for movie with and without homepage');
plt.ylabel('Revenue (100 million dollars)')

# Data Visualization
fig = plt.figure(figsize=(30, 25))
plt.subplot(321)
train['revenue'].plot(kind='hist',bins=100)
plt.title('Distribution of Revenue')
plt.xlabel('Revenue')

plt.subplot(322)
np.log1p(train['revenue']).plot(kind='hist',bins=100)
plt.title(' Log Revenue Distribution')
plt.xlabel('Log Revenue')

plt.subplot(323)
train['budget'].plot(kind='hist',bins=100)
plt.title('Distribution of Budget')
plt.xlabel('Budget')

plt.subplot(324)
np.log1p(train['budget']).plot(kind='hist',bins=100)
plt.title('Log Budget Distribution')
plt.xlabel('Log Budget')

plt.subplot(325)
train['popularity'].plot(kind='hist',bins=100)
plt.title('Distribution of Revenue')
plt.xlabel('Popularity')

plt.subplot(326)
np.log1p(train['popularity']).plot(kind='hist',bins=100)
plt.title('Log popularity Distribution')
plt.xlabel('Log Popularity')

# log money values
train['revenue']=np.log1p(train['revenue'])
train['budget']=np.log1p(train['budget'])
train['popularity']=np.log1p(train['popularity'])

#Genre feature
train['num_genres'] = train['genres'].apply(lambda x: len(x) if x != {} else 0)
list_of_genre_names = list(train['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
#Data Visualization
sns.barplot(x='num_genres', y='revenue', data=train,ci=None)
plt.ylabel('Revenue (100 million dollars)')
words = ' '.join([i for j in list_of_genre_names for i in j])
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1000, height=800).generate(words)
plt.imshow(wordcloud)
plt.title('Top movie genres')
plt.axis("off")
plt.show()
top_genre_names = [m[0] for m in Counter([i for j in list_of_genre_names for i in j]).most_common(10)]
res=[]
for i, value in enumerate(list_of_genre_names):
    res.append(any(elem in value for elem in top_genre_names))

train['genre_common']=res
train['genre_common'] = train['genre_common'].astype(int)
train= train.drop(['genres'],axis=1)

# original_language feature
labels, uniques = pd.factorize(train['original_language'])
train['original_language_factorize']=labels
train= train.drop(['original_language'],axis=1)
train["title_different"]=1
train.loc[train["title"]==train["original_title"],"title_different"]=0
train=train.drop(['original_title'],axis=1)

# production_companies
train['num_production_companies'] = train['production_companies'].apply(lambda x: len(x) if x != {} else 0)
list_of_production_companies= list(train['production_companies'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
top_production_companies = [m[0] for m in Counter([i for j in list_of_production_companies for i in j]).most_common(15)]
res3=[]
for i, value in enumerate(list_of_production_companies):
    res3.append(any(elem in value for elem in top_production_companies))

train['production_companies_common']=res3
train['production_companies_common'] = train['production_companies_common'].astype(int)
train=train.drop(['production_companies'], axis=1)
#Data Visualization
sns.barplot(x='num_production_companies', y='revenue', data=train,ci=None)
plt.ylabel('Revenue (100 million dollars)')
plt.title('Number of production_companies in each movie')

# production countries feature
train['num_production_countries'] = train['production_countries'].apply(lambda x: len(x) if x != {} else 0)
list_of_production_countries = list(train['production_countries'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
top_production_countries = [m[0] for m in Counter([i for j in list_of_production_countries for i in j]).most_common(1)]
res3=[]
for i, value in enumerate(list_of_production_countries):
    res3.append(any(elem in value for elem in top_production_countries))
train['production_country_us']=res3
train['production_country_us'] = train['production_country_us'].astype(int)
train = train.drop(['production_countries'], axis=1)

#Data Visualization
most_common_countries_plot=Counter([i for j in list_of_production_countries for i in j]).most_common(15)
fig = plt.figure()
data=dict(most_common_countries_plot)
names = list(data.keys())
values = list(data.values())
plt.barh(range(len(data)),values,tick_label=names,color='green')
plt.xlabel('Count')
plt.title('Leading countries in movie production')
plt.show()

#Release date feature
train['release_date']=pd.to_datetime(train['release_date'])
train['release_weekday'] = train['release_date'].dt.dayofweek
train['release_month'] = train['release_date'].dt.month
train['release_quarter'] = train['release_date'].dt.quarter
train['release_year'] = train['release_date'].dt.year
train.loc[train['release_year']>2019 , 'release_year'] = train['release_year']-100
train = train.drop(['release_date'], axis=1)
#Data Visualization
fig = plt.figure()
train.groupby('release_year').agg('mean')['revenue'].plot()
plt.ylabel('Revenue (100 million dollars)')
fig = plt.figure()
sns.barplot(x='release_weekday', y='revenue', data=train,ci=None)
loc, labels = plt.xticks()
plt.ylabel('Revenue (100 million dollars)')
loc, labels = loc, ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
plt.xticks(loc, labels)
fig = plt.figure()
sns.barplot(x='release_month', y='revenue', data=train,ci=None)
plt.ylabel('Revenue (100 million dollars)')
plt.title('Released quarter vs Revenue')
fig = plt.figure()
sns.barplot(x='release_quarter', y='revenue', data=train,ci=None)
plt.ylabel('Revenue (100 million dollars)')
plt.title('Released quarter vs Revenue')

# spoken languages feature
train['num_spoken_languages'] = train['spoken_languages'].apply(lambda x: len(x) if x != {} else 0)
train = train.drop(['spoken_languages'], axis=1)
#Data Visualization
fig = plt.figure()
sns.barplot(x='num_spoken_languages', y='revenue', data=train,ci=None)
plt.ylabel('Revenue (100 million dollars)')
plt.title('Number of spoken languages in each movie')

# tagline feature
train['isTaglineNA'] = 0
train.loc[pd.isnull(train['tagline']) ,"isTaglineNA"] = 1
train= train.drop(['tagline'],axis=1)
#Data Visualization
fig = plt.figure()
sns.catplot(x='isTaglineNA', y='revenue', data=train);
plt.title('Compare Revenue for movie with and without tagline');
plt.ylabel('Revenue (100 million dollars)')

# cast feature
train['num_cast'] = train['cast'].apply(lambda x: len(x) if x != {} else 0)
list_of_cast_names = list(train['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
top_cast_names = [m[0] for m in Counter([i for j in list_of_cast_names for i in j]).most_common(15)]
res2=[]
for i, value in enumerate(list_of_cast_names):
    res2.append(any(elem in value for elem in top_cast_names))
train['cast_common']=res2
train['cast_common'] = train['cast_common'].astype(int)
train = train.drop(['cast'], axis=1)
#Data Visualization
fig = plt.figure()
sns.barplot(x='num_cast', y='revenue', data=train,ci=None)
plt.ylabel('Revenue (100 million dollars)')
plt.title('Number of casts in each movie')

# crew feature
train['num_crew'] = train['crew'].apply(lambda x: len(x) if x != {} else 0)
list_of_crew_names = list(train['crew'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
top_crew_names = [m[0] for m in Counter([i for j in list_of_crew_names for i in j]).most_common(15)]
res1=[]
for i, value in enumerate(list_of_crew_names):
    res1.append(any(elem in value for elem in top_crew_names))
train['crew_common']=res1
train['crew_common'] = train['crew_common'].astype(int)
train = train.drop(['crew'], axis=1)
#Data Visualization
fig = plt.figure()
sns.barplot(x='num_crew', y='revenue', data=train,ci=None)
plt.ylabel('Revenue (100 million dollars)')
plt.title('Number of crews in each movie')

# title feature
train['title_count'] = train['title'].str.split().str.len()
train = train.drop(['title'], axis=1)
#Data Visualization
fig = plt.figure()
sns.barplot(x='title_count', y='revenue', data=train,ci=None)
plt.ylabel('Revenue (100 million dollars)')
plt.title('Number of words in each movie title')

# keywords
train['num_Keywords'] = train['Keywords'].apply(lambda x: len(x) if x != {} else 0)
list_of_keywords = list(train['Keywords'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
top_keywords = [m[0] for m in Counter([i for j in list_of_keywords for i in j]).most_common(30)]
res1=[]
for i, value in enumerate(list_of_keywords):
    res1.append(any(elem in value for elem in top_keywords))
train['keyword_common']=res1
train['keyword_common'] = train['keyword_common'].astype(int)
train = train.drop(['Keywords'], axis=1)
#Data Visualization
fig = plt.figure()
sns.barplot(x='num_Keywords', y='revenue', data=train,ci=None)
plt.ylabel('Revenue (100 million dollars)')
plt.title('Number of keywords in each movie')

print(train.head())

train['runtime'].fillna(0,inplace=True)

# correlation of features
data_features = train.select_dtypes(include=['int64', 'float64', 'int32', 'int8']).columns.tolist()
plt.figure(figsize=(18,18))
correlations = train[data_features].corr()
sns.heatmap(correlations, annot=True, fmt='.2', center=0.0, cmap='RdBu_r')
plt.show()

final = 'revenue'
data_features.remove(final)

# implementation of models
def select_model(X, Y):

    best_models = {}
    models = [
        {   'name': 'LinearRegression',
            'estimator': LinearRegression()
        },
        {   'name': 'KNeighborsRegressor',
            'estimator': KNeighborsRegressor(),
        },
        {'name': 'XGBoost',
         'estimator': XGBRegressor(),
         }
    ]
    for model in tqdm(models):
        grid = GridSearchCV(model['estimator'], param_grid={}, cv=5, scoring="neg_mean_squared_error", verbose=False,
                            n_jobs=-1)
        grid.fit(X, Y)
        best_models[model['name']] = {'score': grid.best_score_, 'params': grid.best_params_,
                                      'model': model['estimator']}
    return best_models

models = select_model(train[data_features], train[final])
print(models)

best_model = XGBRegressor()

# Feature Selection - SFS
X, y = train[data_features], train[final]
sfs = SFS(estimator=best_model,
           k_features=(3,26),
           forward=True,
           floating=False,
           scoring='neg_mean_squared_error',
           cv=5)

sfs.fit(X, y, custom_feature_names=data_features)
print('best combination (ACC: %.3f): %s\n' % (sfs.k_score_, sfs.k_feature_names_))

fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')
plt.title('Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()
sfs_features = list(sfs.k_feature_names_)

# optimization of hyper parameters of XGBoost
hyperparameters = {
    'max_depth': range(1, 12, 2),
    'n_estimators': range(90, 201, 10),
    'min_child_weight': range(1, 8, 2),
    'learning_rate': [.05, .1, .15],
}

# using GridSearchCV
grid = GridSearchCV(best_model, param_grid=hyperparameters, cv=3, scoring = "neg_mean_squared_error", verbose=False, n_jobs=-1)
grid.fit(train[sfs_features], train[final])
print('score = {}\nparams={}'.format(grid.best_score_, grid.best_params_))

# final model with tuned parameters and best features
final_model = XGBRegressor(learning_rate=0.15, max_depth=3, min_child_weight=5, n_estimators=100)
final_model.fit(train[sfs_features], train[final], eval_metric='rmse')
predict = final_model.predict(train[sfs_features])
