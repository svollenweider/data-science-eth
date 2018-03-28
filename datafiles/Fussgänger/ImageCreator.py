import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
import imageio

resolution = 50

#Dimensions of Zurich
x = np.linspace(8.46,8.61,num=resolution)
y = np.linspace(47.35,47.4247399,num=resolution)

sigma = 3

# Create Fussgänger and Velo Array
#year = input("Enter Year: ")
year = '2017'
standort = (pd.read_csv("verkehrszaehlungenstandortevelofussgaenger.csv", header=0))[['fk_zaehler','easting_wgs','northing_wgs','von','bis']]
standort['von'],standort['bis'] = pd.to_datetime(standort['von']),pd.to_datetime(standort['bis'])
df = pd.read_csv(year+"verkehrszaehlungenwertefussgaengervelo.csv",header=0)

df['datum'] = pd.to_datetime(df['datum'])

fussgaenger = df[df.fuss_in.notnull()]

fussgaenger = fussgaenger.sort_values('datum')

fussgaenger['total'] = fussgaenger['fuss_in']+fussgaenger['fuss_out']

fussgaenger = fussgaenger.drop(['velo_in','velo_out','fuss_in','fuss_out','objectid'],axis=1).reset_index(drop=True).fillna(value=0)

velo = df[df.velo_in.notnull()]

velo = velo.sort_values('datum')

velo['total'] = velo['velo_in']+velo['velo_out']

velo = velo.drop(['velo_in','velo_out','fuss_in','fuss_out','objectid'],axis=1).reset_index(drop=True).fillna(value=0)
# Ended up with velo and fussgaenger

Maximum = pd.DataFrame({'Datum':[], 'MaxFuss': [], 'MaxVelo': []})

# process Velo, Fussgaenger and Location into 3 arrays with dimensions resolution x resolution
Test = True

while(fussgaenger.shape[0]>0 and Test):
    Zeitraum = fussgaenger['datum'].loc[0]
    image = np.zeros((resolution,resolution,3))
    tempfuss = fussgaenger[fussgaenger['datum'] == Zeitraum].reset_index(drop=True)
    tempvelo = velo[velo['datum'] == Zeitraum].reset_index(drop=True)
    fussgaenger = fussgaenger[fussgaenger['datum'] != Zeitraum].reset_index(drop=True) #remove the used elements
    velo = velo[velo['datum'] != Zeitraum]
    tempstandort = standort[standort['von']<=Zeitraum]
    tempstandort = tempstandort[tempstandort['bis']>Zeitraum]
    #map longitude and lattitude to the data
    Longitude = pd.Series(tempstandort['easting_wgs'].values,tempstandort['fk_zaehler'])

    Lattitude = pd.Series(tempstandort['northing_wgs'].values,tempstandort['fk_zaehler'])
    
    tempfuss['Longitude'] = tempfuss['fk_zaehler'].map(Longitude)
    tempfuss['Lattitude'] = tempfuss['fk_zaehler'].map(Lattitude)
    tempvelo['Longitude'] = tempvelo['fk_zaehler'].map(Longitude)
    tempvelo['Lattitude'] = tempvelo['fk_zaehler'].map(Lattitude)
    for i in range(0,tempfuss.shape[0]-1):
        if x[0] <= tempfuss.Longitude.loc[i] <= x[-1] and y[0] <= tempfuss.Lattitude.loc[i] <= y[-1]:
            xidx = (np.abs(tempfuss.Longitude.loc[i]-x)).argmin()
            yidx = (np.abs(tempfuss.Lattitude.loc[i]-y)).argmin()
            image[xidx,yidx,0] = image[xidx,yidx,0] + tempfuss['total'].iloc[i]
            image[xidx,yidx,2] = image[xidx,yidx,2] + 1
    for i in range(0,tempvelo.shape[0]-1):
        if x[0] <= tempvelo.Longitude.loc[i] <= x[-1] and y[0] <= tempvelo.Lattitude.loc[i] <= y[-1]:
            xidx = (np.abs(tempvelo.Longitude.loc[i]-x)).argmin()
            yidx = (np.abs(tempvelo.Lattitude.loc[i]-y)).argmin()
            image[xidx,yidx,1] = image[xidx,yidx,1] + tempvelo['total'].iloc[i]
            image[xidx,yidx,2] = image[xidx,yidx,2] + 1
    Filename = str(Zeitraum.date())+'-'+str(Zeitraum.minute+Zeitraum.hour*60).zfill(4)
    Maximum = Maximum.append({'Datum' : Zeitraum, 'MaxFuss' : image[:,:,0].max(), 'MaxVelo' : image[:,:,1].max()},ignore_index=True)
    if image[:,:,0].max() > 0.:
        image[:,:,0] = sigma**2*gaussian_filter(image[:,:,0],sigma)
        image[:,:,0] = (image[:,:,0]/image[:,:,0].max()*255.).round()
    if image[:,:,1].max() > 0.:
        image[:,:,1] = gaussian_filter(image[:,:,1],sigma)
        image[:,:,1] = (image[:,:,1]/image[:,:,1].max()*255.).round()
    newimage = image.astype('uint8')
    imageio.imwrite('Images/'+Filename+'.jpg', newimage)
Maximum.to_csv("MaxData"+year+"final.csv",index=False)  
        
        
    
        
    
    





