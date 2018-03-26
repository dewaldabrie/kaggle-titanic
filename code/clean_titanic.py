
from sklearn import preprocessing
import pandas as  pd
import numpy as np

def clean(df):
    """Clean the data and return the inputs and labels"""
    df = df.replace(["male", "female"], [0,1])
    df = df.replace(["S", "C", "Q"], [0,1,2])
    df = df.fillna(0)
    # Split ticket code into prefix and number
    df2 = pd.concat([df, df['Ticket'].str.extract('(?P<ticket_prefix>[\d\w\W\.\/]*) (?P<ticket_number>\d*)')],
              axis=1)
    # Extract last name from full name and encode
    df2['Name'] = df['Name'].str.extract('(?P<Name>[\d\w\W\.\/]*),')
    le = preprocessing.LabelEncoder()
    le.fit(df2.Name.values[:, np.newaxis])
    df2.Name = le.transform(df2['Name'])
    # ... ended up with some isolated cases of a certain text string in the ticket_number column
    df2['ticket_number'] = np.where(df2['ticket_number'] == 'LINE', df2['ticket_number'], 0)
    df2.ticket_number.fillna(df.Ticket, inplace=True)
    df2.ticket_prefix.fillna('', inplace=True)
    # df3 = df.merge(df2, how='left', left_index=True, right_index=True)
    le = preprocessing.LabelEncoder()
    le.fit(df2.ticket_prefix.values[:, np.newaxis])
    df2.ticket_prefix = le.transform(df2['ticket_prefix'])

    if 'Survived' in df.columns:
        y = df[["Survived"]]
    else:
        y = None
    X = df2[["PassengerId","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","ticket_prefix", "ticket_number", "Name"]]
    return (X,y)
