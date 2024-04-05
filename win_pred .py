#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import warnings


# In[2]:


warnings.filterwarnings('ignore')


# In[3]:


match = pd.read_csv('C://Users//Lenovo//Downloads//ipl data//matches.csv')
delivery = pd.read_csv('C://Users//Lenovo//Downloads//ipl data//deliveries.csv')


# In[4]:


match.head()


# In[5]:


delivery.head()


# In[6]:


match.shape


# In[7]:


match.isnull().sum()


# In[8]:


match.tail()


# In[9]:


delivery.tail()


# In[10]:


delivery.columns


# In[11]:


total_score_df = delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()
total_score_df = total_score_df[total_score_df['inning'] == 1]
total_score_df.head()


# In[12]:


total_score_df.shape


# In[13]:


match_df = match.merge(total_score_df[['match_id','total_runs']], left_on = 'id', right_on = 'match_id')


# In[14]:


match_df['team1'].unique()


# In[15]:


len(match_df['team1'].unique())


# In[16]:


team = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Gujarat Lions',
    'Rising Pune Supergiant', 
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 
    'Delhi Daredevils', 
    'Kings XI Punjab',
    'Chennai Super Kings', 'Rajasthan Royals', 'Deccan Chargers',
       'Kochi Tuskers Kerala', 'Pune Warriors', 'Rising Pune Supergiants',
       'Delhi Capitals'
]


# In[17]:


match_df['team1'] = match_df['team1'].apply(lambda x: x.replace('Delhi Daredevils', 'Delhi Capitals'))


# In[18]:


match_df['team1'] = match_df['team1'].apply(lambda x: x.replace('Delhi Daredevils', 'Delhi Capitals'))
match_df['team1'] = match_df['team1'].apply(lambda x: x.replace('Deccan Chargers', 'Sunrisers hyderabad'))
match_df['team2'] = match_df['team2'].apply(lambda x: x.replace('Deccan Chargers', 'Sunrisers hyderabad'))

len(match_df['team1'].unique())


# In[19]:


match_df = match_df[match_df['team1'].isin(team)]
match_df = match_df[match_df['team2'].isin(team)]

len(match_df['team1'].unique())


# In[20]:


match_df.shape


# In[21]:


len(match_df['team1'].unique())


# In[22]:


match_df


# In[23]:


match_df = match_df[match_df['dl_applied'] == 0]


# In[24]:


match_df = match_df[['match_id','city','winner','total_runs']]
match_df


# In[25]:


delivery_df = match_df.merge(delivery, on = 'match_id')
delivery_df = delivery_df[delivery_df['inning']==2]


# In[26]:


delivery_df.shape


# In[27]:


delivery_df


# In[28]:


delivery_df["current_score"] = delivery_df.groupby('match_id').cumsum()['total_runs_y']


# In[29]:


delivery_df


# In[30]:


delivery_df["runs_left"] = delivery_df['total_runs_x'] - delivery_df['current_score']


# In[31]:


delivery_df["runs_left"] = delivery_df['runs_left'].apply(lambda x: x+1)


# In[32]:


delivery_df['balls_left'] = 126 - (delivery_df['over']*6 + delivery_df['ball'])


# In[33]:


delivery_df


# In[34]:


delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")


# In[35]:


delivery_df


# In[36]:


delivery_df['player_dismissed']= delivery_df['player_dismissed'].apply(lambda x: x if x == "0" else 1)


# In[37]:


delivery_df


# In[38]:


delivery_df['player_dismissed'] = delivery_df['player_dismissed'].astype('int')
wickets = delivery_df.groupby('match_id').cumsum()['player_dismissed'].values
delivery_df['wickets'] = 10 - wickets


# In[39]:


delivery_df


# In[40]:


# current run rate  = runs/overs
delivery_df['crr'] = (delivery_df['current_score']*6)/(120 - delivery_df['balls_left'])


# In[41]:


#rrr 
delivery_df['rrr'] = (delivery_df['runs_left']*6)/delivery_df['balls_left']


# In[42]:


delivery_df


# In[43]:


delivery_df['result'] = delivery_df.apply(lambda x: 1 if x['winner'] == x['batting_team'] else 0, axis = 1)


# In[44]:


delivery_df


# In[45]:


delivery_df['result'].value_counts()


# In[46]:


final_df = delivery_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr','result']]


# In[47]:


final_df


# In[48]:


final_df = final_df.sample(final_df.shape[0])
final_df = final_df[final_df['balls_left'] != 0]
final_df.sample()


# In[49]:


final_df.dropna(inplace = True)
final_df


# In[50]:


X = final_df.iloc[:, : -1]
y = final_df.iloc[: , -1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state= 52)
X_train


# In[51]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

trf = ColumnTransformer([
    ('trf',OneHotEncoder(sparse=False,drop='first'),['batting_team','bowling_team','city'])
]
,remainder='passthrough')
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
pipe = Pipeline(steps = [
    ('step1', trf),
    ('step2', LogisticRegression(solver = 'liblinear'))
])
pipe.fit(X_train, y_train)


# In[52]:


y_pred = pipe.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[53]:


pipe.predict_proba(X_test)[10]


# In[54]:


team


# In[55]:


delivery_df['city'].unique()


# In[56]:


import pickle
pickle.dump(pipe, open('win.pkl', 'wb'))


# In[ ]:




