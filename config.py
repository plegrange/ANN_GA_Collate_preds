config = {'Central_data_intersection':'C:\\Users\PHLEGRA\Desktop\MASTER\Data_intersection\Prescribed_parameters',
        'server': 's175BSQLQ101.zarsa.corpintra.net',
        'Stud_list':['6200011','6300120','6200029','6200026','6200797','6200129','6200808','610070','610007','6200189','6200668','6200341','6200669','6200032'],
        'Parameter_Fields_list':['DropTime act', 'DropTime difference', 'Lift Height act', 'Main WeldCurrent Arc Voltage act',
                                 'Pilot WeldCurrent Arc Voltage Act', 'Stickout', 'WeldCurrent act Positive','WeldCurrent act Negative',
                                 'WeldTime Act','Penetration act'],
        'EWMA_files_dump':'C:\\Users\PHLEGRA\Desktop\MASTER\Data_intersection\EWMA_files',
        'Hardware_limitations_upper':{'WeldCurrent act Positive':1500.0},
        'Hardware_limitations_lower':{'WeldCurrent act Positive':800.0},
        'General_limitations':[0,None,1],
        'Start_date':'2018-05-01',
        'Start_time':'00:00:00','End_date':'2018-08-01',
        'End_time':'00:00:00',
        'ANN_GA_Frame':['DropTime act','DropTime act_EWMA','DropTime difference','DropTime difference_EWMA',
                            'Lift Height act','Lift Height act_EWMA', 'Main WeldCurrent Arc Voltage act','Main WeldCurrent Arc Voltage act_EWMA',
                           'Pilot WeldCurrent Arc Voltage Act','Pilot WeldCurrent Arc Voltage Act_EWMA',
                            'Stickout','Stickout_EWMA','WeldCurrent act Positive','WeldCurrent act Positive_EWMA',
                            'WeldCurrent act Negative','WeldCurrent act Negative_EWMA', 'WeldTime Act','WeldTime Act_EWMA','Penetration act']



          }