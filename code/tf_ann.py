import os
from sklearn import preprocessing
import pandas as pd
import numpy as np
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from clean_titanic import clean


def create_ann(input_size=9):
    # Keras

    seed = 42
    np.random.seed(seed)

    # Model
    model = Sequential()
    # input layer
    model.add(Dense(100, input_shape=(input_size,)))
    model.add(BatchNormalization())
    model.add(Activation("selu"))
    model.add(Dropout(0.8))

    # hidden layers
    model.add(Dense(200))
    model.add(BatchNormalization())
    model.add(Activation("sigmoid"))
    model.add(Dropout(0.4))
    
    model.add(Dense(200))
    model.add(BatchNormalization())
    model.add(Activation("sigmoid"))
    model.add(Dropout(0.4))

    model.add(Dense(50, activation="sigmoid"))

    # output layer
    model.add(Dense(1, activation='sigmoid'))

    # model compile for binary classification
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def save(model, model_filename='model.json', weights_filename='weights.h5'):
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_filename, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weights_filename)
    print("Saved model to disk")
 
def load(model_filename='model.json', weights_filename='weights.h5'):
    if not os.path.isfile(model_filename):
        raise ValueError("Model file %s does not exists" % model_filename)

    if not os.path.isfile(weights_filename):
        raise ValueError("Weights file %s does not exists" % weights_filename)

    # load json and create model
    json_file = open(model_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_filename)
    print("Loaded model from disk")
    return loaded_model

def evaluate(model, X_test):
    # evaluate loaded model on test data
    predictions = np.round(model.predict(X_test))
    predictions = pd.DataFrame(predictions, dtype=int)

    # Result
    result = pd.concat([X_test[["PassengerId"]], predictions], axis=1)
    result.columns.values[1] = 'Survived'
    result.rename({'0': 'Survived'}, axis='columns')
    result.Survived = result.Survived.astype('int')
    return result

def main():
    # Training data
    df1 = pd.read_csv("data/train.csv")
    X, y = clean(df1)
    print(df1.head())

    # Test data
    df2 = pd.read_csv("data/test.csv")
    X_unknown, _ = clean(df2)
    print(df2.head())

    # Data split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.99)

    # Model Architecture
    model = create_ann(input_size=11)

    # Load model if exists:
    try:
        print("Loading existing model and fine tuning")
        load()
        epochs = 10
    except ValueError:
        print("Training new model from scratch")
        epochs = 900
    # Learning
    model.fit(X, y, epochs=epochs, batch_size=20)
    
    # Save to file
    save(model)

    # Scoring
    # float to [0,1]
    result = evaluate(model, X_unknown)
    result.to_csv("result.csv", index=False)

if __name__ == '__main__':
    main()
