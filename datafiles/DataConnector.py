import pandas as pd
import numpy as np
from ExternalDataHTMLtoCSVconverter import CreateExternalData
from TramDataCompressor import TramData
from Fussgaenger.TrafficCreator import CreateTrafficData

years = ['2017']
Create = True

ListofData = []

for year in years:
    if Create:
        print('WeatherData')
        #CreateExternalData(year)
        print('TramData')
        TramData(year)
        print('TrafficData')
        CreateTrafficData(year,CreateImage=False) 
    print('Connect Data')
    Traffic = pd.read_csv("Fussgaenger/MaxData"+year+"final.csv",header=0,index_col=['Datum'],parse_dates=True,infer_datetime_format=True)
    External = pd.read_csv("ExternalData/ExternalData"+year+"final.csv",header=0,index_col=['Datum Uhrzeit'],parse_dates=True,infer_datetime_format=True)
    Tram = pd.read_csv("SollIst/Results/VBZDataLine9_"+year+".csv",header=0,index_col=['departure'],parse_dates=True,infer_datetime_format=True)
    for index,row in Tram.iterrows():
        Time = index-pd.Timedelta('1h')
        ExternalRow = External[External.index<Time]
        if ExternalRow.empty:
            continue
        TrafficRow = Traffic[Traffic.index<Time]
        TramEarly = Tram[np.all(np.vstack([index-pd.Timedelta('1h10m')<Tram.index,Tram.index<index-pd.Timedelta('1h'),row['fahrzeug']==Tram['fahrzeug']]),axis=0)]
        if TramEarly.empty:
            prevDelay = 0
        else:
            prevDelay = TramEarly.iloc[-1]['Delay']
        ListofData.append([index]+row.tolist()+TrafficRow.iloc[-1].tolist()+ExternalRow.iloc[-1].tolist()+[prevDelay])
            
AllData = pd.DataFrame(data = ListofData,columns=['datetime','richtung','Stop','fahrzeug','Distance','Delay','Filename','MaxFuss','MaxVelo','Days',
 'Uhrzeit','Weekday','Specialday','Lufttemperatur','Windgeschwindigkeit','Windrichtung','Luftdruck','Niederschlag','Luftfeuchte','delayprior'])

SortedList = ['datetime','richtung','Distance','Filename','MaxFuss','MaxVelo','Days','Uhrzeit','Weekday','Specialday','Lufttemperatur','Windgeschwindigkeit','Windrichtung','Luftdruck','Niederschlag','Luftfeuchte','delayprior','Delay']

SortedData = AllData[SortedList]

SortedData.to_csv("FinalData/AllData.csv",index=False)
SortedData.drop(['datetime'],axis=1).to_csv("FinalData/TrainingData.csv",index=False)
    
