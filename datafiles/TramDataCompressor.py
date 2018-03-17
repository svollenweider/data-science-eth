import pandas as pd
import numpy as np
import os

year = '2017'
files = [f for f in os.listdir('SollIst/'+year+'/') if f.startswith('fahrzeit')]
seriestodrop = ['datum_von','datum_nach','halt_punkt_id_nach','seq_nach','halt_diva_nach','halt_punkt_diva_nach'
                ,'soll_an_nach','soll_ab_nach','ist_ab_nach','fahrt_id','fw_no','fw_kurz','umlauf_von'
                ,'halt_id_nach','halt_punkt_diva_von', 'soll_an_von' , 'ist_an_von','halt_kurz_nach1','ist_an_nach1'
                ,'fahrweg_id','halt_punkt_id_von','halt_diva_von','seq_von','kurs']
Dataframes = []

print("{: 6.1f}".format(0)+' %')
for idx,file in enumerate(files):
    df = pd.read_csv('SollIST/'+file)
    df = df.drop(seriestodrop,axis=1)
    df = df[df.linie == 9].drop('linie',axis=1)
    df['Delay'] = df.ist_ab_von - df.soll_ab_von
    df.soll_ab_von = pd.to_timedelta(df.soll_ab_von,unit='s')
    df.betriebsdatum = pd.to_datetime(df.betriebsdatum) + df.soll_ab_von
    df = df.drop(['soll_ab_von','ist_ab_von'],axis=1)
    Dataframes.append(df)
    print("{: 6.1f}".format((idx+1)/len(files)*100)+' %')

print('Concat now')
AllData = pd.concat(Dataframes)
AllData = AllData.sort_values('betriebsdatum').reset_index()
AllData = AllData.rename(columns={'betriebsdatum' : 'departure', 'halt_kurz_von1' : 'Stop', 'halt_id_von' : 'halt_id' })
print('Concat finished. Saving')
AllData.to_csv("VBZDataLine9_2017_final.csv",index=False)
print('Saving successful')