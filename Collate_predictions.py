import os
import shutil
import pandas as pd
import config
import GA
# if data.loc[index, 'null_zero'] == True:
#    limitations = config.config['Hardware_limitations_upper']
#    for key, value in limitations.items():
#         if float(entry[str(key)]) > float(value):
#             data.loc[index, 'null_zero'] = False
#     limitations = config.config['Hardware_limitations_lower']
#     for key, value in limitations.items():
#         if float(entry[str(key)]) < float(value):
#             data.loc[index, 'null_zero'] = False

def filter_out_bad_preds(data):
    #data['null_zero'] = [True for n in range(0, len(data.index))]
    Survived_Elite = pd.DataFrame()
    frames = []
    upper_limits = config.config['Hardware_limitations_upper']
    lower_limits = config.config['Hardware_limitations_lower']
    general_limits = config.config['General_limitations']
    for index, entry in data.iterrows():
        valid = True
        print(str(index) + ' : ' + str(entry))
        for key_limit, value_limit in upper_limits.items():
            if float(entry[str(key_limit)]) > float(value_limit):
                valid = False
        if valid:
            for key_limit, value_limit in lower_limits.items():
                if float(entry[str(key_limit)]) < float(value_limit):
                    valid = False
        if valid:
            for par in entry:
                if float(par) in general_limits:
                    valid = False
        if valid:
            Survived_Elite_record = pd.DataFrame.from_records([entry], columns=data.columns.values)
            frames.append(Survived_Elite_record)
    try:
        print("-----------" + str(len(data.index()))+ " INDIVIDUALS SURVIVED: MORTALITY RATE: " + str(round(len(data.index())/len(frames)*100),2)+" % -----------")
        Survived_Elite = pd.concat(frames)
        return Survived_Elite
    except:
        print("-----------NONE SURVIVED: MORTALITY RATE: 100%-----------")
        pass


def CleanDataFrame(df):
    """Clean the dataframe"""
    try:
        newFrame = pd.DataFrame()
        index = 0
        for item, value in df.values:
            value_string = value.replace("[", "").replace("]", "")
            Item = item.replace("[[", "").replace("]]", "")
            value_array = value_string.split(",")
            # zero_count = 0
            # for value in value_array:
            #     if value== '0':
            #         zero_count = 1
            # if zero_count == 0:
            newFrame.set_value(index, 'Error', float(Item))
            newFrame.set_value(index, 'Drop Time act', float(value_array[0]))
            newFrame.set_value(index, 'Drop Time act EWMA', float(value_array[1]))
            newFrame.set_value(index, 'Drop Time reference', float(value_array[2]))
            newFrame.set_value(index, 'Drop Time reference EWMA', float(value_array[3]))
            # newFrame.set_value(index, 'Energy', value_array[4])
            # newFrame.set_value(index, 'Energy EWMA', value_array[5])
            newFrame.set_value(index, 'Lift Height act', float(value_array[4]))
            newFrame.set_value(index, 'Lift Height act EWMA', float(value_array[5]))
            # newFrame.set_value(index, 'Left Higiht ref', value_array[8])
            # newFrame.set_value(index, 'lift height ref_EWMA', value_array[9])
            newFrame.set_value(index, 'Main Weldcurrent voltage act',float( value_array[6]))
            newFrame.set_value(index, 'Main Weldcurrent voltage act_EWMA', float(value_array[7]))
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
            newFrame.set_value(index, 'Pilot WeldCurrent Arc Voltage Act', float(value_array[8]))
            newFrame.set_value(index, 'Pilot WeldCurrent Arc Voltage Act_EWMA', float(value_array[9]))
            # newFrame.set_value(index, 'Pilot Weldcurrent Arc Voltage Max', value_array[24])
            # newFrame.set_value(index, 'Pilot Weldcurrent Arc Voltage Max_EWMA', value_array[25])
            # newFrame.set_value(index, 'Pilot Weldcurrent Arc Voltage Min', value_array[26])
            # newFrame.set_value(index, 'Pilot Weldcurrent Arc Voltage Min_EWMA', value_array[27])
            newFrame.set_value(index, 'Stickout', float(value_array[10]))
            newFrame.set_value(index, 'Stickout_EWMA', float(value_array[11]))
            newFrame.set_value(index, 'WeldCurrent act Positive', float(value_array[12]))
            newFrame.set_value(index, 'WeldCurrent act Positive_EWMA', float(value_array[13]))
            newFrame.set_value(index, 'WeldCurrent act Negative', float(value_array[14]))
            newFrame.set_value(index, 'WeldCurrent act Negative_EWMA', float(value_array[15]))
            newFrame.set_value(index, 'Weld time act', float(value_array[16]))
            newFrame.set_value(index, 'Weld time act_EWMA', float(value_array[17]))
            # newFrame.set_value(index, 'Weldtime ref', value_array[34])
            # newFrame.set_value(index, 'Weldtime ref_EWMA', value_array[35])
            index += 1
            print(index, " entries.....")
        return pd.DataFrame(newFrame)
    except:
        print("Dataframe Filled")
        return pd.DataFrame(newFrame)


def processData(stud,new_path,processed_path,errors_path,server):
    """Takes as input a file path and reads in the files in directory"""
    files = os.listdir(new_path)
    #print(files)
    frames=[]

    try:
        processed_files = 0
        for file in files:
            processed_files += 1

            if str(file).endswith(".csv") and str(stud) in str(file):
                dataframe = pd.read_csv(new_path + '\\' + file) #changed this from read_excel to read_csv
                print(file.lower())
                frames.append(dataframe)
                #cleanFrame = CleanDataFrame(dataframe)
                # ex.ExportToSql(cleanFrame, server, 'Welding')
                #shutil.move(new_path + '\\' + file, processed_path + '\\' + file)
                # os.remove(new_path + '\\' + file)

                print("Processed files :" + str(processed_files) + ", Estimated row count :" + str(
                    processed_files * 5000))

        finalFrame = pd.concat(frames, ignore_index=True)
        return CleanDataFrame(finalFrame)
    except:
        #os.remove(new_path + '\\' + file)     commented this out to prevent deletion of unprocessed files
        shutil.move(new_path + '\\' + file, errors_path + '\\' + file)
        print(file+ " removed...........")
        #Restart the process
        #processData(new_files_path)


# read from server

def Collate_Main():
    if GA.ANN_GA_main():
        print("Collating Predictions...")
        stud_array = config.config['Stud_list']
        for stud in stud_array:
            # server to write to
            server = config.config['server']
            # raw data hosting file path
            new_files_path = config.config['Central_data_intersection'] + str("\\new")
            # processed files path
            processed_files_path = config.config['Central_data_intersection'] + str("\\processed")
            # errors file path
            errors= config.config['Central_data_intersection'] + str("\\errors")
            data = pd.DataFrame(processData(stud,new_files_path,processed_files_path,errors,server))
            data = filter_out_bad_preds(data)
            try:
                data.to_csv('All_predictions_'+stud+'.csv')
                print(data.head())
            except:
                print('Please collect more data and try again')


Collate_Main()
print("Collation Complete!")
