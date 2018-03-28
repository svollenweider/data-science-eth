import pandas as pd
import numpy as np

Year = input("Enter Year: ")

df = pd.read_csv(Year+"verkehrszaehlungenwertefussgaengervelo.csv",header=0)

df['datum'] = pd.to_datetime(df['datum'])

fussgaenger = df[df.fuss_in.notnull()]

fussgaenger = fussgaenger.sort_values('datum')

fussgaenger['total'] = fussgaenger['fuss_in']+fussgaenger['fuss_out']

fussgaenger = fussgaenger.drop(['velo_in','velo_out','fuss_in','fuss_out','objectid'],axis=1).reset_index(drop=True).fillna(value=0)

velo = df[df.velo_in.notnull()]

velo = velo.sort_values('datum')

velo['total'] = velo['velo_in']+velo['velo_out']

velo = velo.drop(['velo_in','velo_out','fuss_in','fuss_out','objectid'],axis=1).reset_index(drop=True).fillna(value=0)

velo_compressed = pd.DataFrame(columns=['DateTime','velo_average','velo_median','velo_std'])

while len(velo.index)>0:
    idx = velo.index[velo['datum'] == velo['datum'][0]].tolist()
    velo_compressed = velo_compressed.append({ 'DateTime': velo['datum'][0], 'velo_average' : velo['total'][idx].mean(), 'velo_median' : velo['total'][idx].median(),'velo_std' : velo['total'][idx].std()},ignore_index=True)
    velo = velo.drop(idx).reset_index(drop=True)

fuss_compressed = pd.DataFrame(columns=['DateTime','fuss_average','fuss_median','fuss_std'])

while len(fussgaenger.index)>0:
    idx = fussgaenger.index[fussgaenger['datum'] == fussgaenger['datum'][0]].tolist()
    fuss_compressed = fuss_compressed.append({'DateTime' : fussgaenger['datum'][0], 'fuss_average': fussgaenger['total'][idx].mean(), 'fuss_median' : fussgaenger['total'][idx].median(), 'fuss_std' : fussgaenger['total'][idx].std()},ignore_index=True)
    fussgaenger = fussgaenger.drop(idx).reset_index(drop=True)

if (fuss_compressed['DateTime']==velo_compressed['DateTime']).all():
    FullTable = pd.concat([velo_compressed,fuss_compressed.drop('DateTime',axis=1)],axis=1)
    FullTable.to_csv("TrafficData"+Year+"final.csv",index=False)
else:
    fuss_compressed.to_csv("Fussgaenger"+Year+"final.csv",index=False)
    velo_compressed.to_csv("Velo"+Year+"final.csv",index=False)

