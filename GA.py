from numpy import random
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn import preprocessing
from keras.callbacks import ModelCheckpoint
from pyeasyga.pyeasyga import GeneticAlgorithm
from sklearn.utils import shuffle
import numpy as np
from urllib.parse import quote
from sqlalchemy import create_engine
import urllib
import pandas as pd
import config
import EWMA


def neural_network(dataframe):
    dataset = dataframe.values
    numpy.random.shuffle(dataset)
    # split into input (X) and output (Y) variables
    X = dataset[:, 0:18]
    Y = dataset[:, 18]
    # scale all the data
    scaleX = preprocessing.StandardScaler().fit(X)
    X = scaleX.transform(X)
    Y = Y.reshape(-1, 1)
    scaleY = preprocessing.StandardScaler().fit(Y)
    Y = scaleY.transform(Y)

    # define base model
    def baseline_model():
        # create model
        model = Sequential()
        model.add(Dense(18, input_dim=18, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        return model

    # define wider model
    def wider_model():
        model = Sequential()
        model.add(Dense(18, input_dim=18, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(48, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, kernel_initializer='normal'))
        return model
        # define the model

    def larger_model():
        # create model
        model = Sequential()
        model.add(Dense(18, input_dim=18, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(24, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(24, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, kernel_initializer='normal'))
        return model
    print("ANN Training Started")
    model = larger_model()
    checkpoint = ModelCheckpoint('model/regression.h5', monitor='val_mean_absolute_error', verbose=1,
                                 save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    # Compile model
    model.compile(loss='mse', optimizer='adam', metrics=['mean_absolute_error'])
    # Fit the model
    history = model.fit(X, Y, validation_split=0.2, epochs=100, batch_size=5000, callbacks=callbacks_list,
                        verbose=False)
    #####################################################
    model.load_weights('model/regression.h5')
    validation_start = ((int)(len(Y) * (1.0 - 0.2)))
    validationdata = X[validation_start:, :]
    y = Y[validation_start:]
    # calculate predictions
    predictions = model.predict(validationdata)
    print("training set:")
    trainingdata = X[0:validation_start - 1, :]
    yt = Y[0:validation_start - 1]
    predictionst = model.predict([trainingdata])
    trainingdata = numpy.concatenate(
        (scaleY.inverse_transform(yt).reshape(len(yt), 1), scaleY.inverse_transform(predictionst)), axis=1)
    trainingdata = numpy.concatenate((scaleX.inverse_transform(X[0:validation_start - 1, :]), trainingdata), axis=1)

    validationdata = numpy.concatenate(
        (scaleY.inverse_transform(y).reshape(len(y), 1), scaleY.inverse_transform(predictions)), axis=1)
    validationdata = numpy.concatenate((scaleX.inverse_transform(X[validation_start:, :]), validationdata), axis=1)
    numpy.savetxt("model/training_regression.csv", trainingdata)
    numpy.savetxt("model/validation_regression.csv", validationdata)
    numpy.savetxt('model/trainingchart_regression.csv',
                  scaleY.inverse_transform(history.history['mean_absolute_error']))
    numpy.savetxt('model/validationchart_regression.csv',
                  scaleY.inverse_transform(history.history['val_mean_absolute_error']))
    # calculate predictions
    predictions = model.predict(X)
    # round predictions
    print(scaleY.inverse_transform(predictions))
    return model
    ########################################################################################################################


def Genetic_al(model, data, i, stud1):
    ga = GeneticAlgorithm(data, population_size=100, generations=100, crossover_probability=0.01,
                          mutation_probability=0.01, elitism=False, maximise_fitness=False)

    def create_individual(data):
        return [float(1.0 + 5.0 * random.uniform(0, 1)),  # Drop time actual
                float(1.0 + 15.0 * random.uniform(0, 1)),  # Drop time actual_EWMA
                float(1.0 + 15.0 * random.uniform(0, 1)),  # Drop time difference
                float(1.0 + 15.0 * random.uniform(0, 1)),  # Drop time difference_EWMA
                # float((random.randint(300, 4000) + random.uniform(0, 1))),# Energy
                # float((random.randint(300, 4000) + random.uniform(0, 1))),  # Energy_EWMA
                float(1.0 + 2.0 * random.uniform(0, 1)),  # Lift Height actual
                float(1.0 + 2.0 * random.uniform(0, 1)),  # Lift Height actual_EWMA
                # float((random.randint(0, 2) + random.uniform(0, 1))), # lift height ref
                # float((random.randint(0, 2) + random.uniform(0, 1))),  # lift height ref_EWMA
                float(15.0 + 40.0 * random.uniform(0, 1)),  # Main Weldcurrent voltage actual
                float(15.0 + 4.0 * random.uniform(0, 1)),  # Main Weldcurrent voltage actual_EWMA
                # ((random.randint(25, 50) + random.uniform(0, 1))),# Main Weldcurrent voltage Max
                # ((random.randint(25, 50) + random.uniform(0, 1))),  # Main Weldcurrent voltage Max_EWMA
                # ((random.randint(8, 28) + random.uniform(0, 1))), # Main Weldcurrent voltage Min
                # ((random.randint(8, 28) + random.uniform(0, 1))),  # Main Weldcurrent voltage Min_EWMA
                # (-(random.randint(1, 3) + random.uniform(0, 1))), # Penetration Max
                # (-(random.randint(1, 3) + random.uniform(0, 1))),  # Penetration Max_EWMA
                # (-(random.randint(0, 3) + random.uniform(0, 1))), # Penetration Min
                # (-(random.randint(0, 3) + random.uniform(0, 1))),  # Penetration Min_EWMA
                # (-(random.randint(0, 3) + random.uniform(0, 1))),# Penetratiom Ref
                # (-(random.randint(0, 3) + random.uniform(0, 1))),  # Penetratiom Ref_EWMA
                float(8.0 + 35.0 * random.uniform(0, 1)),  # Pilot Weldcurrent Arc Voltage Act
                float(8.0 + 35.0 * random.uniform(0, 1)),  # Pilot Weldcurrent Arc Voltage Act_EWMA
                # ((random.randint(20, 50) + random.uniform(0, 1))),# Pilot Weldcurrent Arc Voltage Max
                # ((random.randint(20, 50) + random.uniform(0, 1))),  # Pilot Weldcurrent Arc Voltage Max_EWMA
                # ((random.randint(5, 20) + random.uniform(0, 1))),# Pilot Weldcurrent Arc Voltage Min
                # ((random.randint(5, 20) + random.uniform(0, 1))),  # Pilot Weldcurrent Arc Voltage Min_EWMA
                float(1.5 + 8.0 * random.uniform(0, 1)),  # Stickout
                float(1.5 + 8.0 * random.uniform(0, 1)),  # Stickout_EWMA
                float(500.0 + 1000.0 * random.uniform(0, 1)),  # Weldcurrent actual Positive
                float(500.0 + 1000.0 * random.uniform(0, 1)),  # Weldcurrent actual Positive_EWMA
                float(-1500.0 + 1500.0 * random.uniform(0, 1)),  # Weldcurrent actual Negative
                float(-1500.0 + 1500.0 * random.uniform(0, 1)),  # Weldcurrent actual Negative_EWMA
                float(10.0 + 100.0 * random.uniform(0, 1)),  # Weld time actual
                float(10.0 + 100.0 * random.uniform(0, 1))]  # Weld time actual_EWMA
        # ((random.randint(10, 100) + random.uniform(0, 1))), # Weldtime ref
        # ((random.randint(10, 100) + random.uniform(0, 1)))]  # Weldtime ref_EWMA

    ga.create_individual = create_individual

    def eval_fitness(individual, data):
        array = np.array(individual)[np.newaxis]
        error_array = []
        error = (model.predict(array, batch_size=1) + 2) ** 2
        error_array.append(individual)
        print('Evaluating... error: ' + str(error))
        return error

    ga.fitness_function = eval_fitness
    ga.run()
    Gen1 = pd.DataFrame(ga.last_generation())
    filepath = "C:\\Users\PHLEGRA\Desktop\MASTER\Data_intersection\Prescribed_parameters\\new"
    filename = str(stud1) + "_" + str(i) + '_predictions.csv'
    Gen1.to_csv(filepath + '\\' + filename, index=False)
    print('Please see file. Process Complete')


def getEWMATable(stud1):
    df = pd.read_csv(config.config['EWMA_files_dump'] + "\\" + str(stud1) + "_EWMA_Values.csv")
    return df


def BuildTestFrame(stud1):
    # build a new dataframe
    parameter_list = config.config['Parameter_Fields_list']
    column_labels = ['StudID', 'Type', 'Date', 'Time']
    dataFrame = pd.DataFrame(columns=column_labels)
    ewma = getEWMATable(stud1)  # This is to only get the date, type, StudId and Time
    StudSeries = pd.Series(ewma['StudID'])
    dataFrame['StudID'] = StudSeries.values
    StudSeries = pd.Series(ewma['Type'])
    dataFrame['Type'] = StudSeries.values
    DateSeries = pd.Series(ewma['Date'])
    dataFrame['Date'] = pd.Series(DateSeries)
    TimeSeries = pd.Series(ewma['Time'])
    dataFrame['Time'] = pd.Series(TimeSeries)
    # TypeEncodedSeries = pd.Series(ewma['Type Encoded'])
    # dataFrame['Type Encoded'] = pd.Series(TypeEncodedSeries)
    df = ewma
    for parameter in parameter_list:
        # with engine.connect() as con:
        #     rs = con.execute("SELECT StudID,[Type],[Date],[Time],[EWMA],[Value] FROM dbo.EWMA_Stud WHERE (StudID ='"+stud1+"') AND Parameter='"+parameter+"'")
        #     df = pd.DataFrame(rs.fetchall())
        #     df.columns = rs.keys()
        ParameterSeries = pd.Series(df['Value'])
        # ParameterSeries = pd.Series(ewma['Value'])
        dataFrame[str(parameter)] = pd.Series(ParameterSeries)
        EWMASeries = pd.Series(df['EWMA'])
        # EWMASeries = pd.Series(ewma['EWMA'])
        dataFrame[str(parameter) + "_EWMA"] = pd.Series(EWMASeries)
    # dataFrame['Penetration Encoded']=dataFrame['Type Encoded']*dataFrame['Penetration act']
    return dataFrame


def ANN_GA_main():
    if EWMA.EWMA_main():
        params = urllib.parse.quote_plus(
            "Trusted_Connection=yes;DRIVER= {ODBC Driver 13 for SQL Server};SERVER=s175BSQLQ103.zarsa.corpintra.net;DATABASE=Welding")
        link = "mssql+pyodbc:///?odbc_connect=%s" % params
        engine = create_engine(link)
        stud_array = config.config['Stud_list']
        for stud1 in stud_array:
            EWMA_params = BuildTestFrame(stud1)
            WoPframe = EWMA_params[EWMA_params['Type'] == 'WOP']
            WoPframe = shuffle(WoPframe)
            print(len(WoPframe))
            WiPframe = EWMA_params[EWMA_params['Type'] == 'WIP']
            WiPframe = shuffle(WiPframe)
            # WiPframe = WiPframe.head(n=len(WoPframe))
            print("Welds out of parameter : WOP " + str(len(WoPframe)))
            print("Welds in parameter : WIP " + str(len(WiPframe)))
            print("merging frames")
            frames = [WiPframe, WoPframe]
            frame = pd.concat(frames)
            print("Shuffle merged frame")
            frame = shuffle(frame)
            print("Process ANN and Stud_welding_Parameter_Optimisation_Master..........")
            print("output:")
            Dataset_for_GA = frame[config.config['ANN_GA_Frame']]
            print("Columns being measured")
            print(Dataset_for_GA.columns.values)
            print("shape")
            print(Dataset_for_GA.shape)
            count = 20
            for i in range(0, count):
                Genetic_al(neural_network(Dataset_for_GA), Dataset_for_GA, i, stud1)
        print("GA Complete")
        return True
