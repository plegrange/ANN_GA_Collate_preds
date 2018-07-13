import os
import shutil
import xml.etree.ElementTree as ET
import bs4 as bs
import pandas as pd
#import extract as ex
import numpy as np

def FM_XML2DF(index, file_count, file):
    Combined_data = pd.DataFrame()
    print(file)
    data_array = []
    element_index = 0
    with open(file, "r") as fp:
        row_count = 0
        contents = fp.read()
        headings = []
        soup = bs.BeautifulSoup(contents, "xml")
        titles = soup.find_all('Row')
        for title in titles:
            dfs = []
            df1 = pd.DataFrame()
            td = title.find_all('Data')         #Filling row
            data_array = [i.text for i in td]
            #print(data_array)                   #Inserting row into dataframe
            if row_count == 0:
                Combined_data = pd.DataFrame(columns=[data_array])
                headings = data_array
            elif row_count != 0:
                df = pd.DataFrame.from_records([data_array], columns=[headings])
                # dfs.append(df)
                # df1 = pd.concat(dfs, ignore_index=True)
                Combined_data = pd.concat([Combined_data, df], ignore_index=True)
                #print(Combined_data.shape)
            row_count += 1
    fp.close()
    return Combined_data


def CleanDataFrame(df):
    """Clean the dataframe"""
    try:
        newFrame = pd.DataFrame()
        index = 0
        for item, value in df.iterrows():
            newFrame.set_value(index, 'Device Name', value['Device Name'])
            newFrame.set_value(index,'Date',value['Date / Time'])
            newFrame.set_value(index, 'Type', value['Type'])
            newFrame.set_value(index, 'StudID', value['Stud-ID:'])
            newFrame.set_value(index, 'Application', value['Application'])

            # Application encoded:
            encoding = 0
            if value['Application'] == 'Aluminum':
                encoding = 1
            else:
                encoding = 2

            newFrame.set_value(index, 'Application encoded', int(encoding))
            newFrame.set_value(index, 'System Weld Counter', value['System weld  counter'])

            # stickout
            stickout = 0
            stick = value['Stickout']
            if (stick != 'Nan'):
                stickout = float(stick.split(" mm")[0])
            else:
                stickout = 0
            newFrame.set_value(index, 'Stickout', stickout)

            # DropTime Difference
            DropTimeDifference = 0
            DropTimeActual = value['Droptime (Ref / Actual)']
            if DropTimeActual != 'Nan':
                dropSplit = DropTimeActual.split('/')
                ref = float([dropSplit[0].split(' ')][0][0])
                actual = float([dropSplit[1].split(' ')][0][1])
                DropTimeDifference = ref - actual
            else:
                DropTimeDifference = 0
            newFrame.set_value(index, 'DropTime difference', DropTimeDifference)

            # DropTime Actual
            newFrame.set_value(index, 'DropTime act', actual)

            # Pilot WeldCurrent Arc Voltage Act
            PilotWeldCur = value['Pilot Weldcurrent Arc Voltage  Actual (Up)']
            val = PilotWeldCur.split(" ")[0]
            newFrame.set_value(index, 'Pilot WeldCurrent Arc Voltage Act', val)

            # Main WeldCurrent Voltage Act3
            MainWeldCurrent = value['Main Weldcurrent Voltage  Actual (Us)']
            vals = float(MainWeldCurrent.split(' ')[0])
            newFrame.set_value(index, 'Main WeldCurrent Arc Voltage act', vals)

            # WeldCurrent act
            WeldCurrent = 0
            Weld = value['Weldcurrent Actual (Is)']
            if "/" in Weld:
                WeldCurrent = float(Weld.split("/")[1].split("A")[0])
            else:
                WeldCurrent = float(Weld.split("A")[0])
            newFrame.set_value(index, 'WeldCurrent act', WeldCurrent)

            # WeldTime Act
            weldtime = float(value['Weldtime  Actual (It)'].split(" ms")[0])
            newFrame.set_value(index, 'WeldTime Act', weldtime)

            # Lift Height act
            liftHeight = value['Lift height Actual'].split(" mm")[0]
            if (liftHeight == 'Nan'):
                liftHeight = 0
                newFrame.set_value(index, 'Lift Height act', float(liftHeight))
            else:
                newFrame.set_value(index, 'Lift Height act', float(liftHeight))

            # Penetration act
            PenetrationAct = value['Penetration  Actual (P)'].split(" mm")[0]
            if (PenetrationAct == 'Nan'):
                newFrame.set_value(index, 'Penetration act', 0)
            else:
                newFrame.set_value(index, 'Penetration act', float(PenetrationAct))

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

            if str(file).endswith(".csv"):
                dataframe = pd.read_csv(new_path + '\\' + file)
                print(file.lower())
                cleanFrame = CleanDataFrame(dataframe)
                # ex.ExportToSql(cleanFrame, server, 'Welding')
                shutil.move(new_path + '\\' + file, processed_path + '\\' + file)
                # os.remove(new_path + '\\' + file)

                print("Processed files :" + str(processed_files) + ", Estimated row count :" + str(
                    processed_files * 5000))

            elif str(file).endswith(".xml"):
                path_of_file = new_path + '\\' + file
                dataframe = FM_XML2DF(49,6, path_of_file)
                frames.append(dataframe)
                #print((dataframe.head()))
                print('files processed' + str(processed_files))

                # ex.ExportToSql(cleanFrame, server, 'Welding')
                shutil.move(new_path + '\\' + file, processed_path + '\\' + file)
        # row_count = 0
        # for frame in frames:
        #     dfs = []
        #     for rows in frame.iterrows():
        #         if row_count == 0:
        #             finalFrame = pd.DataFrame(columns=[rows])
        #             headings = rows
        #             row_count += 1
        #         if row_count != 0:
        #             row_count += 1
        #             # df = pd.DataFrame.from([rows], columns=[headings])
        #             # dfs.append(df)
        #             df1 = pd.concat(rows, ignore_index=True)
        #         finalFrame = pd.concat([finalFrame, df1], ignore_index=True)
        finalFrame = pd.concat(frames, ignore_index=True)
        #cleanFrame = CleanDataFrame(finalFrame)
        return finalFrame
    except:
        os.remove(new_path + '\\' + file)
        shutil.move(new_path + '\\' + file, errors_path + '\\' + file)
        print(file+ " removed...........")
        #Restart the process
        #processData(new_files_path)


# read from server


# server to read from

# server to write to
server='s175BSQLQ101.zarsa.corpintra.net'
# raw data hosting file path
new_files_path = "C:\\Users\shpeard\Desktop\PyCharm\Code Repository\Stud welding\Fault History\\new"
# processed files path
processed_files_path = "C:\\Users\shpeard\Desktop\PyCharm\Code Repository\Stud welding\Fault History\processed"
# errors file path
errors="C:\\Users\shpeard\Desktop\PyCharm\Code Repository\Stud welding\Fault History\errors"
data = pd.DataFrame(processData(new_files_path,processed_files_path,errors,server))
print(data.head())

