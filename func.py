import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import logistic_regression_path
# pd.set_option('display.max_row',None)
# pd.set_option('display.max_columns',None)
pd.set_option('display.width',None)
pd.set_option('max_colwidth',1000)

df = pd.read_csv('new.csv')
new_df = df.sort_values(by='funded_amnt_inv',ascending=True)
new_df.to_csv('new.csv')
new = pd.read_csv('new.csv')


df =df.drop("pub_rec_bankruptcies",axis=1)
df = df.dropna(axis=0)
null_counts = df.isnull().sum()
print(null_counts)
print(df.dtypes.value_counts())
object_columns_df = df.select_dtypes(include=["object"])
print(object_columns_df.loc[0])
cols = ['home_ownership','verification_status','emp_length','term','addr_state']
for c in cols:
    print(df[c].value_counts())
mapping_dict = {
    "10+ years":10,
    "9 years":9,
    "8 years":8,
    "7 years":7,
    "6 years":6,
    "5 years": 5,
    "4 years": 4,
    "3 years": 3,
    "2 years": 2,
    "1 year": 1,
    "< 1 year": 0
}
print(df)
# df = df.drop(["last_credit_pull_d","earliest_cr_line","addr_state","title"],axis=1)
df['int_rate'] = df['int_rate'].str.rstrip("%").astype("float")
df['revol_util'] = df['revol_util'].str.rstrip("%").astype("float")
df = df.replace(mapping_dict)

cat_columns=['home_ownership','verification_status','purpose','term']
dummy_df = pd. get_dummies(df[cat_columns]).astype('int64')
df = pd.concat([df,dummy_df],axis=1)
df = df.drop(cat_columns,axis=1)
df = df.drop('pymnt_plan',axis=1)
# df =df.sort_values(by='funded_amnt_inv',ascending=True)
df.to_csv('new.csv')
# print(new)
plt.scatter(range(len(df['funded_amnt_inv'])),df['funded_amnt_inv'],s=1)
plt.show()