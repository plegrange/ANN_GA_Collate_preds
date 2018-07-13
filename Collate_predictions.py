import os
import shutil
import xml.etree.ElementTree as ET
import bs4 as bs
import pandas as pd
#import extract as ex
import numpy as np

def filter_out_bad_preds(data):
    for index, entry in data.iterrows():
        print(str(index) + ' : ' + str(entry))
        for par in entry:
            if par in [0,1]:
                data.drop(data.index[index], inplace = True)
                print('Deleted entry')
    return data

def CleanDataFrame(df):
    """Clean the dataframe"""
    try:
        newFrame = pd.DataFrame()
        index = 0
        for item, value in df.values:
            value_string = value.replace("[", "")
            value_array = value_string.split(",")
            zero_count = 0
            for value in value_array:
                if value== 0:
                    zero_count = 1
            if zero_count == 0:
                Item = float((item.split("[[ ")[1]).split("]]")[0])
                newFrame.set_value(index, 'Error', Item)
                newFrame.set_value(index, 'Drop Time actual', value_array[0])
                newFrame.set_value(index, 'Drop Time actual EWMA', value_array[1])
                newFrame.set_value(index, 'Drop Time reference', value_array[2])
                newFrame.set_value(index, 'Drop Time reference EWMA', value_array[3])
                # newFrame.set_value(index, 'Energy', value_array[4])
                # newFrame.set_value(index, 'Energy EWMA', value_array[5])
                newFrame.set_value(index, 'Lift Height actual', value_array[4])
                newFrame.set_value(index, 'Lift Height actual EWMA', value_array[5])
                # newFrame.set_value(index, 'Left Higiht ref', value_array[8])
                # newFrame.set_value(index, 'lift height ref_EWMA', value_array[9])
                newFrame.set_value(index, 'Main Weldcurrent voltage actual', value_array[6])
                newFrame.set_value(index, 'Main Weldcurrent voltage actual_EWMA', value_array[7])
                # newFrame.set_value(index, 'Main Weldcurrent voltage Max', value_array[12])
                # newFrame.set_value(index, 'Main Weldcurrent voltage Max_EWMA', value_array[13])
                # newFrame.set_value(index, 'Main Weldcurrent voltage Min', value_array[14])
                # newFrame.set_value(index, 'Main Weldcurrent voltage Min_EWMA', value_array[15])
                # newFrame.set_value(index, 'Penetration Max', value_array[16])
                # newFrame.set_value(index, 'Penetration Max_EWMA', value_array[17])
                # newFrame.set_value(index, 'Penetration Min', value_array[18])
                # newFrame.set_value(index, 'Penetration Min_EWMA', value_array[19])
                # newFrame.set_value(index, 'Penetratiom Ref', value_array[20])
                # newFrame.set_value(index, 'Penetratiom Ref_EWMA', value_array[21])
                newFrame.set_value(index, 'Pilot Weldcurrent Arc Voltage Act', value_array[8])
                newFrame.set_value(index, 'Pilot Weldcurrent Arc Voltage Act_EWMA', value_array[9])
                # newFrame.set_value(index, 'Pilot Weldcurrent Arc Voltage Max', value_array[24])
                # newFrame.set_value(index, 'Pilot Weldcurrent Arc Voltage Max_EWMA', value_array[25])
                # newFrame.set_value(index, 'Pilot Weldcurrent Arc Voltage Min', value_array[26])
                # newFrame.set_value(index, 'Pilot Weldcurrent Arc Voltage Min_EWMA', value_array[27])
                newFrame.set_value(index, 'Stickout', value_array[10])
                newFrame.set_value(index, 'Stickout_EWMA', value_array[11])
                newFrame.set_value(index, 'Weldcurrent actual Positive', value_array[12])
                newFrame.set_value(index, 'Weldcurrent actual Positive_EWMA', value_array[13])
                newFrame.set_value(index, 'Weldcurrent actual Negative', value_array[14])
                newFrame.set_value(index, 'Weldcurrent actual Negative_EWMA', value_array[15])
                newFrame.set_value(index, 'Weld time actual', value_array[16])
                newFrame.set_value(index, 'Weld time actual_EWMA', value_array[17])
                # newFrame.set_value(index, 'Weldtime ref', value_array[34])
                # newFrame.set_value(index, 'Weldtime ref_EWMA', value_array[35])
                index += 1
                print(index, " entries.....")
        return pd.DataFrame(newFrame)
    except:
        print("Dataframe Filled")
        return pd.DataFrame(newFrame)


def processData(new_path,processed_path,errors_path,server):
    """Takes as input a file path and reads in the files in directory"""
    files = os.listdir(new_path)
    #print(files)
    frames=[]

    try:
        processed_files = 0
        for file in files:
            processed_files += 1

            if str(file).endswith(".xlsx"):
                dataframe = pd.read_excel(new_path + '\\' + file)
                print(file.lower())
                frames.append(dataframe)
                #cleanFrame = CleanDataFrame(dataframe)
                # ex.ExportToSql(cleanFrame, server, 'Welding')
                shutil.move(new_path + '\\' + file, processed_path + '\\' + file)
                # os.remove(new_path + '\\' + file)

                print("Processed files :" + str(processed_files) + ", Estimated row count :" + str(
                    processed_files * 5000))

        finalFrame = pd.concat(frames, ignore_index=True)
        cleanFrame = CleanDataFrame(finalFrame)
        return cleanFrame
    except:
        os.remove(new_path + '\\' + file)
        shutil.move(new_path + '\\' + file, errors_path + '\\' + file)
        print(file+ " removed...........")
        #Restart the process
        #processData(new_files_path)


# read from server


# server to read from
stud_array = ['6300120'] #,'6100144','6100192','6100136','6300135','6300136']
for stud in stud_array:
    # server to write to
    server='s175BSQLQ101.zarsa.corpintra.net'
    # raw data hosting file path
    new_files_path = "C:\\Users\shpeard\Desktop\WORKING ON THIS\Code\models\GA\Parameters\\"+stud+"\\new"
    # processed files path
    processed_files_path = "C:\\Users\shpeard\Desktop\WORKING ON THIS\Code\models\GA\Parameters\\"+stud+"\processed"
    # errors file path
    errors="C:\\Users\shpeard\Desktop\WORKING ON THIS\Code\models\GA\Parameters\\"+stud+"\errors"
    data = pd.DataFrame(processData(new_files_path,processed_files_path,errors,server))
    print(data[0].count())
    data = filter_out_bad_preds(data)
    print('after removing bad predictions :' + data[0].count())
    data.to_csv('All_predictions_'+stud+'.csv')
    print(data.head())
