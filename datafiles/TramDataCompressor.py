import pandas as pd
import numpy as np
import os
imp = False
if not imp:
    year = '2017'
    files = [f for f in os.listdir('SollIst/'+year+'/') if f.startswith('fahrzeit')]
    seriestodrop = ['datum_von','datum_nach','halt_punkt_id_nach','seq_nach','halt_diva_nach','halt_punkt_diva_nach'
                    ,'soll_an_nach','soll_ab_nach','ist_ab_nach','fahrt_id','fw_no','umlauf_von'
                    ,'halt_id_nach','halt_punkt_diva_von', 'soll_an_von' , 'ist_an_von','halt_kurz_nach1','ist_an_nach1'
                    ,'fahrweg_id','halt_punkt_id_von','halt_diva_von','seq_von','fw_lang','kurs']
    Dataframes = []

    print("{: 6.1f}".format(0)+' %')
    for idx,file in enumerate(files):
        df = pd.read_csv('SollIst/'+year+'/'+file)
        df = df.drop(seriestodrop,axis=1)
        df = df[df.linie == 9].drop('linie',axis=1)
        df = df[df['fw_kurz'].isin([1,2,3,4])].drop('fw_kurz',axis=1) 
        df['Delay'] = df.ist_ab_von - df.soll_ab_von
        df.soll_ab_von = pd.to_timedelta(df.soll_ab_von,unit='s')
        df.betriebsdatum = pd.to_datetime(df.betriebsdatum) + df.soll_ab_von
        df = df.drop(['soll_ab_von','ist_ab_von'],axis=1)
        Dataframes.append(df)
        print("{: 6.1f}".format((idx+1)/len(files)*100)+' %')

    print('Concat now')
    print('Concat finished. Saving')

    AllData = pd.concat(Dataframes)
    AllData = AllData.sort_values('betriebsdatum').reset_index(drop=True)
    AllData = AllData.rename(columns={'betriebsdatum' : 'departure', 'halt_kurz_von1' : 'Stop', 'halt_id_von' : 'halt_id' })
else: AllData = pd.read_csv('SollIst/Results/VBZDataLine9_2017.csv',header=0)


    
Line9 = pd.read_csv('SollIst/Line9.csv',header=0).drop(['shape_id','shape_pt_sequence'],axis=1)
Line9['shape_dist_traveled'] = Line9['shape_dist_traveled']/Line9['shape_dist_traveled'].max()
uniquevals = np.unique(AllData['halt_id'])
Stops = pd.read_csv('SollIst/haltestelle.csv',header=0)
Stops = Stops[['halt_id','halt_lang']]
Stops = Stops[Stops['halt_id'].isin(uniquevals)].reset_index(drop=True)
Stoplocations = (pd.read_csv('SollIst/Stops.csv')).drop(['stop_url','location_type','parent_station'],axis=1)
Stoplocations = (Stoplocations[~Stoplocations.stop_id.str.startswith('Parent')]).drop('stop_id',axis=1)
Stops['Distance'] = 0
StopMap = pd.Series(0.0,Stops.halt_id)
for i in Stops.index:
    Stopname = Stops.iloc[i].halt_lang
    Possible = Stoplocations[Stoplocations['stop_name'] == Stopname]
    min = 100
    idx = 0
    for j in Possible.index:
        DistanceList = ((Line9['shape_pt_lat']-Possible.loc[j].stop_lat)**2+(Line9['shape_pt_lon']-Possible.loc[j].stop_lon)**2)
        if DistanceList.min() < min:
            min, idx = DistanceList.min(), DistanceList.idxmin()
    StopMap[Stops.iloc[i].halt_id] = Line9['shape_dist_traveled'].iloc[idx]

AllData['Distance'] = AllData.halt_id.map(StopMap)
 
SaveColumns = ['richtung', 'Stop', 'departure','fahrzeug','Distance','Delay']
SaveData = AllData[SaveColumns]

SaveData.to_csv("SollIst/Results/VBZDataLine9_2017.csv",index=False,encoding='utf-8')
print('Saving successful')