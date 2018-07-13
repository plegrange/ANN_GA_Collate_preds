import pandas as pd
import urllib
from numpy import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab as pl
from collections import OrderedDict
import matplotlib
from collections import defaultdict
from datetime import datetime
from csv import reader
from math import sqrt
#import pymssql as pm
import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn import preprocessing
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from deap import algorithms, base, creator, tools
from pyeasyga.pyeasyga import GeneticAlgorithm
import time
from functools import wraps
from bs4 import BeautifulSoup
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import CategoricalColorMapper
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool
from bokeh.charts import Histogram
from bokeh.core.properties import Any, Dict, Instance, String
from bokeh.models import ColumnDataSource, LayoutDOM
import requests
from sklearn.utils import shuffle
import numpy as np
from sqlalchemy import create_engine
import pyodbc
import urllib
from urllib.parse import quote
import pandas as pd
from math import sqrt
import requests
from sqlalchemy import create_engine
import pyodbc
import urllib
from urllib.parse import quote
import pandas as pd
import datetime
def neural_network(dataframe):
    dataset = dataframe.values
    numpy.random.shuffle(dataset)
    # split into input (X) and output (Y) variables
    X = dataset[:, 0:17]
    Y = dataset[:, 17]
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
    model = baseline_model()
    checkpoint = ModelCheckpoint('model/regression.h5', monitor='val_mean_absolute_error', verbose=1,
                                 save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    # Compile model
    model.compile(loss='mse', optimizer='adam', metrics=['mean_absolute_error'])
    # Fit the model
    history = model.fit(X, Y, validation_split=0.2, epochs=1000, batch_size=5000, callbacks=callbacks_list,
                        verbose=2)
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
    #data = pd.DataFrame(data.reset_index())
    best = []
    best_error = 10
    ga = GeneticAlgorithm(data, population_size=2500, generations=5, crossover_probability=0.8,
                          mutation_probability=0.1, elitism=False, maximise_fitness=False)
    def create_individual(data):
        return [(random.randint(1,15)+random.uniform(0,1)), # Drop time actual
                (random.randint(1, 15) + random.uniform(0, 1)),  # Drop time actual_EWMA
                ((random.randint(1,15)+random.uniform(0,1))), # Drop time difference
                ((random.randint(1, 15) + random.uniform(0, 1))),  # Drop time difference_EWMA
                #((random.randint(300, 4000) + random.uniform(0, 1))),# Energy
                #((random.randint(300, 4000) + random.uniform(0, 1))),  # Energy_EWMA
                ((random.randint(0, 2) + random.uniform(0, 1))),# Lift Height actual
                ((random.randint(0, 2) + random.uniform(0, 1))),  # Lift Height actual_EWMA
                # ((random.randint(0, 2) + random.uniform(0, 1))), # lift height ref
                # ((random.randint(0, 2) + random.uniform(0, 1))),  # lift height ref_EWMA
                ((random.randint(15, 40) + random.uniform(0, 1))),# Main Weldcurrent voltage actual
                ((random.randint(15, 40) + random.uniform(0, 1))),  # Main Weldcurrent voltage actual_EWMA
                #((random.randint(25, 50) + random.uniform(0, 1))),# Main Weldcurrent voltage Max
                #((random.randint(25, 50) + random.uniform(0, 1))),  # Main Weldcurrent voltage Max_EWMA
                #((random.randint(8, 28) + random.uniform(0, 1))), # Main Weldcurrent voltage Min
                #((random.randint(8, 28) + random.uniform(0, 1))),  # Main Weldcurrent voltage Min_EWMA
                #(-(random.randint(1, 3) + random.uniform(0, 1))), # Penetration Max
                #(-(random.randint(1, 3) + random.uniform(0, 1))),  # Penetration Max_EWMA
                #(-(random.randint(0, 3) + random.uniform(0, 1))), # Penetration Min
                #(-(random.randint(0, 3) + random.uniform(0, 1))),  # Penetration Min_EWMA
                #(-(random.randint(0, 3) + random.uniform(0, 1))),# Penetratiom Ref
                #(-(random.randint(0, 3) + random.uniform(0, 1))),  # Penetratiom Ref_EWMA
                ((random.randint(8, 35) + random.uniform(0, 1))),# Pilot Weldcurrent Arc Voltage Act
                ((random.randint(8, 35) + random.uniform(0, 1))),  # Pilot Weldcurrent Arc Voltage Act_EWMA
                #((random.randint(20, 50) + random.uniform(0, 1))),# Pilot Weldcurrent Arc Voltage Max
                #((random.randint(20, 50) + random.uniform(0, 1))),  # Pilot Weldcurrent Arc Voltage Max_EWMA
                #((random.randint(5, 20) + random.uniform(0, 1))),# Pilot Weldcurrent Arc Voltage Min
                #((random.randint(5, 20) + random.uniform(0, 1))),  # Pilot Weldcurrent Arc Voltage Min_EWMA
                ((random.randint(1.5, 8) + random.uniform(0, 1))), # Stickout
                ((random.randint(1.5, 8) + random.uniform(0, 1))),  # Stickout_EWMA
                ((random.randint(0, 1500) + random.uniform(0, 1))),  # Weldcurrent actual Positive
                ((random.randint(0, 1500) + random.uniform(0, 1))),  # Weldcurrent actual Positive_EWMA
                (-(random.randint(0, 1500) + random.uniform(0, 1))),  # Weldcurrent actual Negative
                (-(random.randint(0, 1500) + random.uniform(0, 1))),  # Weldcurrent actual Negative_EWMA
                ((random.randint(10, 100) + random.uniform(0, 1))), # Weld time actual
                ((random.randint(10, 100) + random.uniform(0, 1)))]  # Weld time actual_EWMA
                #((random.randint(10, 100) + random.uniform(0, 1))), # Weldtime ref
                #((random.randint(10, 100) + random.uniform(0, 1)))]  # Weldtime ref_EWMA
    ga.create_individual = create_individual
    def eval_fitness(individual, data):
        array = np.array(individual)[np.newaxis]
        error_array = []
        error = (model.predict(array, batch_size=1) + 1.5) ** 2
        error_array.append(individual)
        print('Evaluating... error: ' + str(error))
        return error
    ga.fitness_function = eval_fitness
    ga.run()
    Gen1 = pd.DataFrame(ga.last_generation())
    filepath = "\\emea.corpdir.net\e175\Public\MBSA-EL\X-Functional\Big Data\Big Data\Shuld\Alu Stud preds\\'"+str(stud1)+"\\"+"'new'"
    filename = str(stud1) +"_"+ str(i)+ '_predictions.csv'
    Gen1.to_csv(filepath + '\\' + filename, index = False)
    print('Please see file. Process Complete')
def getStuds():
    with engine.connect() as con:
        rs = con.execute("SELECT DISTINCT StudID From dbo.EWMA_Stud")
        df = pd.DataFrame(rs.fetchall())
        df.columns = rs.keys()
        return df
def getEWMATable(stud1):
    # para = 'DropTime act'
    # with engine.connect() as con:
    #     rs = con.execute("SELECT StudID,[Type],[Date],[Time],[EWMA],[Value] FROM dbo.EWMA_Stud WHERE (StudID ='"+stud1+"') AND Parameter='"+para+"'")
    #     df = pd.DataFrame(rs.fetchall())
    #     df.columns = rs.keys()
    df =pd.read_csv("V:\Big Data\Shuld\Alu Stud preds\\"+str(stud1)+"\\"+str(stud1)+ "_EWMA_Values.csv")
    return df
def BuildTestFrame(stud1):
    # build a new dataframe
    parameter_list = ['DropTime act', 'DropTime difference', 'Lift Height act', 'Main WeldCurrent Arc Voltage act', 'Pilot WeldCurrent Arc Voltage Act','Stickout', 'WeldCurrent act Positive'
        ,'WeldCurrent act Negative', 'WeldTime Act','Penetration act']
    column_labels = ['StudID','Type','Date','Time']
    dataFrame = pd.DataFrame(columns=column_labels)
    ewma=getEWMATable(stud1) #This is to only get the date, type, StudId and Time
    StudSeries=pd.Series(ewma['StudID'])
    dataFrame['StudID'] = StudSeries.values
    StudSeries=pd.Series(ewma['Type'])
    dataFrame['Type'] = StudSeries.values
    DateSeries=pd.Series(ewma['Date'])
    dataFrame['Date']=pd.Series(DateSeries)
    TimeSeries=pd.Series(ewma['Time'])
    dataFrame['Time']=pd.Series(TimeSeries)
    # TypeEncodedSeries = pd.Series(ewma['Type Encoded'])
    # dataFrame['Type Encoded'] = pd.Series(TypeEncodedSeries)
    df = ewma
    for parameter in parameter_list:
        # with engine.connect() as con:
        #     rs = con.execute("SELECT StudID,[Type],[Date],[Time],[EWMA],[Value] FROM dbo.EWMA_Stud WHERE (StudID ='"+stud1+"') AND Parameter='"+parameter+"'")
        #     df = pd.DataFrame(rs.fetchall())
        #     df.columns = rs.keys()
        ParameterSeries = pd.Series(df['Value'])
        #ParameterSeries = pd.Series(ewma['Value'])
        dataFrame[str(parameter)] = pd.Series(ParameterSeries)
        EWMASeries = pd.Series(df['EWMA'])
        #EWMASeries = pd.Series(ewma['EWMA'])
        dataFrame[str(parameter) + "_EWMA"] = pd.Series(EWMASeries)
#    dataFrame['Penetration Encoded']=dataFrame['Type Encoded']*dataFrame['Penetration act']
    return dataFrame
params = urllib.parse.quote_plus("Trusted_Connection=yes;DRIVER= {ODBC Driver 13 for SQL Server};SERVER=s175BSQLQ103.zarsa.corpintra.net;DATABASE=Welding")
link = "mssql+pyodbc:///?odbc_connect=%s" % params
engine = create_engine(link)
# parameter_val =['DropTime Actual', 'DropTime Ref', 'Energy','Lift Height act', 'Lift Height ref',
#                  'Main WeldCurrent Voltage act', 'Main Weldcurrent Voltage Maximum', 'Main Weldcurrent Voltage Min',
#                  'Penetration Max', 'Penetration Min', 'Penetration Ref','Pilot WeldCurrent Arc Voltage Act',
#                  'Pilot Weldcurrent Arc Voltage Maximum', 'Pilot Weldcurrent Arc Voltage Minimum', 'Stickout',
#                  'WeldCurrent act', 'WeldTime Act', 'Weldtime Reference', 'Penetration act']

#['6200188','6200229','6200153','6100192','6200085','6200086','6100008','6100009','6100155']
stud_array = ['6100192','6200085','6200086','6100008','6100009','6100155','6200153','6200188','6200229']
for stud1 in stud_array:
    EWMA_params=BuildTestFrame(stud1)
    WoPframe=EWMA_params[EWMA_params['Type'] == 'WOP']
    WoPframe = shuffle(WoPframe)
    print(len(WoPframe))
    WiPframe=EWMA_params[EWMA_params['Type'] == 'WIP']
    WiPframe=shuffle(WiPframe)
    #WiPframe = WiPframe.head(n=len(WoPframe))
    print("Welds out of parameter : WOP "+str(len(WoPframe)))
    print("Welds in parameter : WIP "+str(len(WiPframe)))
    print("merging frames")
    frames=[WiPframe,WoPframe]
    frame=pd.concat(frames)
    print("Shuffle merged frame")
    frame=shuffle(frame)
    print("Process ANN and Stud_welding_Parameter_Optimisation_Master..........")
    print("output:")
    Dataset_for_GA = frame[['DropTime act','DropTime act_EWMA','DropTime difference','DropTime difference_EWMA',
                            'Lift Height act','Lift Height act_EWMA', 'Main WeldCurrent Arc Voltage act','Main WeldCurrent Arc Voltage act_EWMA',
                           'Pilot WeldCurrent Arc Voltage Act','Pilot WeldCurrent Arc Voltage Act_EWMA',
                            'Stickout','Stickout_EWMA','WeldCurrent act Positive','WeldCurrent act Positive_EWMA',
                            'WeldCurrent act Negative','WeldCurrent act Negative_EWMA', 'WeldTime Act','WeldTime Act_EWMA','Penetration act']]
    # ['DropTime Actual', 'DropTime Actual_EWMA', 'DropTime Ref', 'DropTime Ref_EWMA', 'Energy', 'Energy_EWMA',
    #  'Lift Height act', 'Lift Height act_EWMA', 'Lift Height ref', 'Lift Height ref_EWMA',
    #  'Main WeldCurrent Voltage act', 'Main WeldCurrent Voltage act_EWMA',
    #  'Main Weldcurrent Voltage Maximum', 'Main Weldcurrent Voltage Maximum_EWMA', 'Main Weldcurrent Voltage Min',
    #  'Main Weldcurrent Voltage Min_EWMA',
    #  'Penetration Max', 'Penetration Max_EWMA', 'Penetration Min', 'Penetration Min_EWMA', 'Penetration Ref',
    #  'Penetration Ref_EWMA', 'Pilot WeldCurrent Arc Voltage Act', 'Pilot WeldCurrent Arc Voltage Act_EWMA',
    #  'Pilot Weldcurrent Arc Voltage Maximum', 'Pilot Weldcurrent Arc Voltage Maximum_EWMA',
    #  'Pilot Weldcurrent Arc Voltage Minimum', 'Pilot Weldcurrent Arc Voltage Minimum_EWMA',
    #  'Stickout', 'Stickout_EWMA', 'WeldCurrent act', 'WeldCurrent act_EWMA', 'WeldTime Act', 'WeldTime Act_EWMA',
    #  'Weldtime Reference', 'Weldtime Reference_EWMA', 'Penetration act']]
    print("Columns being measured")
    print(Dataset_for_GA.columns.values)
    print("shape")
    print(Dataset_for_GA.shape)
    count = 50
    for i in range(0, count):
        Genetic_al(neural_network(Dataset_for_GA),Dataset_for_GA, i, stud1)