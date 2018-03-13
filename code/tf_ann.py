import pandas as  pd
import numpy as np

def clean(df):
    """Clean the data and return the inputs and labels"""
    df = pd.read_csv("data/train.csv")
    df = df.replace(["male", "female"], [0,1])
    df = df.replace(["S", "C", "Q"], [0,1,2])
    df = df.fillna(0)
    # split ticket code into prefix and number
    df2 = pd.concat([df['Ticket'], df['Ticket'].str.extract('(?P<ticket_prefix>[\d\w\W\.\/]*) (?P<ticket_number>\d*)')],
              axis=1)
    df2.ticket_number.fillna(df.Ticket, inplace=True)
    df2.ticket_prefix.fillna('', inplace=True)
    import pdb
    pdb.set_trace()
    print("Size of df before merge: ", df.shape)
    print("Size of df2 before merge: ", df2.shape)
    print("All values in merge columns the same?: %s" % all(df.Ticket.values == df2.Ticket.values))
    df3 = pd.merge(df, df2, how='inner', left_on=['Ticket'], right_on=['Ticket'], copy=False)
    print("Size of df3 after merge: ", df3.shape)
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(df3.ticket_prefix.values[:, np.newaxis])
    df3.ticket_prefix = le.transform(df3['ticket_prefix'])
    #extract last name from full name and encode

    if 'Survived' in df.columns:
        y = df[["Survived"]]
    else:
        y = None
    X = df3[["PassengerId","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","ticket_prefix","ticket_number"]]
    return (X,y)

def create_ann():
    # Keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.models import Sequential, load_model
    from keras.layers import Dense, Dropout, BatchNormalization, Activation
    from keras.wrappers.scikit_learn import KerasRegressor

    seed = 42
    np.random.seed(seed)

    # Model
    model = Sequential()
    # input layer
    model.add(Dense(8, input_shape=(8,)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.4))

    # hidden layers
    model.add(Dense(8))
    model.add(BatchNormalization())
    model.add(Activation("sigmoid"))
    model.add(Dropout(0.4))

    model.add(Dense(4))
    model.add(BatchNormalization())
    model.add(Activation("sigmoid"))
    model.add(Dropout(0.4))

    model.add(Dense(2, activation="sigmoid"))

    # output layer
    model.add(Dense(1, activation='linear'))

    # model compile for binary classification
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def main():
    # Training data
    df1 = pd.read_csv("data/train.csv")
    X,y = clean(df1)
    print(df1.head())

    # Test data
    df2 = pd.read_csv("data/test.csv")
    X_Test, _ = clean(df2)
    print(df2.head())

    # Data split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2)

    # Model Architecture
    model = create_ann()

    # Learning
    model.fit(X, y, nb_epoch=300, batch_size=30)

    # Scoring
    # float to [0,1]
    predictions = np.round(model.predict(X_Test))
    predictions = pd.DataFrame(predictions)

    # Result
    result = pd.concat([df1[["PassengerId"]], predictions], axis=1)
    predictions.to_csv("result.csv", index=False)

if __name__ == '__main__':
    main()
