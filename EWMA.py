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
import config
import gc
import _thread
from ServerClient import send_data_to_server
from ServerClient import connect_to_server
def readInValues(carbodyIDs,Robots,Types,StudID,times,dates,list_vals, lambda_val):
    """Reads in a list of values and builds a dataframe that contains columns for vals"""
    # data = pd.read_csv(filepath)

    column_labels = ['CarbodyID','Robot','StudID', 'Date','Time','Shift','Hour','Value', 'Mean', 'lambda', 'LCL_EWMA', 'UCL_EWMA', 'EWMA', 'Problems',
                     'Red band','UpperSigma1','UpperSigma2','UpperSigma3','LowerSigma1','LowerSigma2','LowerSigma3']
    dataFrame = pd.DataFrame(columns=column_labels)

    index = 0
    for ent in list_vals:
        dataFrame.set_value(index, 'CarbodyID', carbodyIDs) #There are no carbodies in the data
        dataFrame.set_value(index, 'Robot', Robots[index])
        dataFrame.set_value(index, 'Type', Types[index])
        dataFrame.set_value(index, 'StudID', StudID)
        date = str(dates[index]).split(" ")[0]
        dataFrame.set_value(index, 'Date', date)
        dataFrame.set_value(index, 'Time', times[index])
        date=str(date)+" "+str(times[index]).strip()
        date=datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        shift = calculateShift(date)
        dataFrame.set_value(index, 'Shift', shift)
        hour=str(times[index]).split(":")[0]
        dataFrame.set_value(index, 'Hour', int(hour))
        dataFrame.set_value(index, 'lambda', lambda_val)
        dataFrame.set_value(index, 'Value', float(ent))
        index += 1
        #print(str(index))

    return len(dataFrame['Value']), dataFrame


def calculateShift(date):
    dt = datetime.datetime.strptime(str(date), "%Y-%m-%d %H:%M:%S").time()
    A_lower = datetime.time(6, 0,0)
    A_upper=datetime.time(14, 14, 0)
    B_lower=datetime.time(14, 15, 0)
    B_upper=datetime.time(22, 14, 0)

    if dt >=A_lower and dt<=A_upper:
        return "A"
    elif dt>=B_lower and dt<=B_upper:
        return "B"
    else:
        return "C"


def caluclateDescriptors(dataframe):
    """[mean,standard deviation,variance,pop variance,sample variance]
    Takes as input a dataframe and calculates values and inserts them
    in specified row and column"""

    # calculate mean
    sum = dataframe['Value'].sum()
    mean = dataframe['Value'].mean()
    # calculate ssd
    differences = [x - mean for x in dataframe['Value']]
    sq_differences = [d ** 2 for d in differences]
    std = dataframe['Value'].std()
    var = std ** 2
    ssd = np.sum(sq_differences)
    pop_var = ssd / len(dataframe['Value'])
    sample_var = ssd / len(dataframe['Value']) - 1
    dataframe['Mean'] = str(mean)
    #print("Statistical descriptors")
    #print(dataframe['Value'].describe())

    return [mean, std, ssd, var, pop_var, sample_var]


def calcEWMA(dataframe, mean, std, ssd, var, pop_var, sample_var, lambda_val,parameter_val):
    """Use the data to calculate the EWMA values
    retruns upper and lower limit values for sigma, list of unclassified EWMA values and an updated dataframe"""

    EWMA_list = []
    items = dataframe['Value']
    dataframe2 = pd.DataFrame()
    mean_EWMA = 0
    index = 0
    EWMA = pd.ewma(items, span=10)
    dataframe['EWMA'] =EWMA
    for row in items:
        LCL = EWMA + std
        dataframe.set_value(index, 'LCL_EWMA', LCL)
        UCL = EWMA - std
        dataframe.set_value(index, 'UCL_EWMA', UCL)
        dataframe.set_value(index,'Parameter',parameter_val)
        index += 1
        # mean_EWMA += row
    index=0
    upper = [mean + std, mean + 2 * std, mean + 3 * std]
    lower = [mean - std, mean - 2 * std, mean - 3 * std]


    #for row in upper:

    return upper, lower, dataframe


def ApplyWesternElectricRules(dataframe2, Mean, sd,upperlimits,lowerlimits):
    """Applies western electric rules to the EWMA, UCL and LCL columns"""

    values = np.array(dataframe2['Value'])
    problems = list()
    red_band = list()
    # dataframe2 = pd.DataFrame()
    for i in range(0, len(dataframe2['Value'])):
        dataframe2.set_value(i, 'Red band', 0)

    for index in range(len(values)):

        # Apply sigma values
        dataframe2.set_value(index, 'UpperSigma1', float(upperlimits[0]))
        dataframe2.set_value(index, 'UpperSigma2', float(upperlimits[1]))
        dataframe2.set_value(index, 'UpperSigma3', float(upperlimits[2]))
        dataframe2.set_value(index, 'LowerSigma1', float(lowerlimits[0]))
        dataframe2.set_value(index, 'LowerSigma2', float(lowerlimits[1]))
        dataframe2.set_value(index, 'LowerSigma3', float(lowerlimits[2]))

        # Apply Redband and Problems
        if values[index] > (Mean + 3 * sd) or values[index] < (Mean - 3 * sd):
            dataframe2.set_value(index, 'Problems', 4)
            dataframe2.set_value(index, 'Red band', 1)
            # Western Electric rule 1
        elif values[index] > (Mean + 2 * sd) and values[index - 1] > (Mean + 2 * sd) and index >= 2:
            dataframe2.set_value(index, 'Problems', 3)
            dataframe2.set_value(index, 'Red band', 1)
            dataframe2.set_value(index - 1, 'Red band', 1)
        elif values[index] < (Mean - 2 * sd) and values[index - 1] < (Mean - 2 * sd) and index >= 2:
            dataframe2.set_value(index, 'Problems', 3)
            dataframe2.set_value(index, 'Red band', 1)
            dataframe2.set_value(index - 1, 'Red band', 1)
            # Western Electric rule 2
        elif values[index] >= (Mean + sd) and values[index - 1] >= (Mean + sd) and values[index - 2] >= (Mean + sd) and \
                        values[index - 3] >= (Mean + sd) and values[index - 4] >= (Mean + sd) and index >= 5:
            dataframe2.set_value(index, 'Problems', 2)
            for i in range(0, 5):
                dataframe2.set_value(index - i, 'Red band', 1)
        elif values[index] <= (Mean - sd) and values[index - 1] <= (Mean - sd) and values[index - 2] <= (Mean - sd) and \
                        values[index - 3] <= (Mean - sd) and values[index - 4] <= (Mean - sd) and index >= 5:
            dataframe2.set_value(index, 'Problems', 2)
            for i in range(0, 5):
                dataframe2.set_value(index - i, 'Red band', 1)
                # Western electric rule 3
        elif values[index] <= (Mean - sd) and values[index - 1] <= (Mean - sd) and values[index - 2] <= (Mean - sd) and \
                        values[index - 3] <= (Mean - sd) and values[index - 4] <= (Mean - sd) and \
                        values[index - 5] <= (Mean - sd) and values[index - 6] <= (Mean - sd) and \
                        values[index - 7] <= (Mean - sd) and index >= 8:
            dataframe2.set_value(index, 'Problems', 1)
            for i in range(0, 8):
                dataframe2.set_value(index - i, 'Red band', 1)
                # Western Electric Rule 4
        else:
            dataframe2.set_value(index, 'Problems', 0)

    return dataframe2


def CalculateEWMA(StudID,Dataframe, lambda_val, parameter_value):
    """Takes as input a dataframe and calulates the EWMA"""
    # calculate EWMA
    end_files = []
    # Build dataframe with specified columns for each parameter
    for par in parameter_value:
        print("Stud : " +str(StudID) +" Parameter : "+par)
        measured_parameter=Dataframe[par]
        carbodyID=-1000#Dataframe['CarbodyID']
        dates= pd.to_datetime(Dataframe['Date'], format='%Y-%m-%d %H:%M:%S')
        times=Dataframe['Time']
        types = Dataframe['Type']
        robots=Dataframe['Robot']
        print('Number of entries to process: ' + str(len(robots)))
        length, sub_frame = readInValues(carbodyID,robots,types,StudID,times,dates, measured_parameter, lambda_val)
        mean, std, ssd, var, pop_var, sample_var = caluclateDescriptors(sub_frame)
        upperlimts, lowerlimits, sub_frame = calcEWMA(sub_frame, mean, std, ssd, var, pop_var, sample_var, lambda_val,par)
        sub_frame = ApplyWesternElectricRules(sub_frame, mean, std,upperlimts,lowerlimits)
        end_files.append(sub_frame)
        print('EWMA table updated...................')
    return pd.concat(end_files)
        #return sub_frame [returns a sub_frame with the data in it]


def exportToSql(dataframe, table, engine):
    """Takes as input a dataframe and writes the data to sql"""

    dataframe.to_sql(table, engine, if_exists='append', index=False)


def getData(stud):
    params = urllib.parse.quote_plus("Trusted_Connection=yes;DRIVER= {ODBC Driver 17 for SQL Server};SERVER=s175BSQLQ103.zarsa.corpintra.net;DATABASE=Welding")
    link = "mssql+pyodbc:///?odbc_connect=%s" % params
    engine = create_engine(link)
    with engine.connect() as con:
        rs = con.execute("SELECT * From [dbo].[WELDING_Stud] where StudID = '" + stud + "' and [Date] > '" +
                         config.config['Start_date'] + "' and [Date] < '" + config.config['End_date'] + "' and [Time] > '" +
                         config.config['Start_time'] + "' and [Time] > '" + config.config['End_time']+"'")
        df = pd.DataFrame(rs.fetchall())
        df.columns = rs.keys()
        print(df)
        return df


def EWMA_main():
    studs = config.config['Stud_list']
    print("Stud count : "+str(len(studs)))
    print(" Running EWMA For studs..................")
    for stud in studs:
        parameter_val = config.config['Parameter_Fields_list']
        frame = getData(stud)
        frame = CalculateEWMA(stud, frame, 0.75, parameter_val)
        #frame.to_csv(config.config['EWMA_files_dump']+"\\"+str(stud)+ "_EWMA_Values.csv")
        connect_to_server()
        _thread.start_new_thread(send_data_to_server,
                                 (config.config['EWMA_files_dump']+"\\"+str(stud)+ "_EWMA_Values.csv", frame))
        #gc.collect()
    print("EWMA Complete")
    return True
