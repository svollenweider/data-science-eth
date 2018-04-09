import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize
import imageio


def CreateTrafficData(year,CreateImage=True,path='Fussgaenger/'):
    #Dimensions of Zurich
    resolution= 120
    x = np.linspace(8.48,8.56,num=resolution)
    y = np.linspace(47.354,47.415,num=resolution)
    sigma = resolution / 20.

    standort = (pd.read_csv(path+"verkehrszaehlungenstandortevelofussgaenger.csv", header=0))[['fk_zaehler','easting_wgs','northing_wgs','von','bis']]
    standort['von'],standort['bis'] = pd.to_datetime(standort['von']),pd.to_datetime(standort['bis'])
    df = pd.read_csv(path+year+"verkehrszaehlungenwertefussgaengervelo.csv",header=0)

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

    Maximum = pd.DataFrame({'Datum':[], 'MaxFuss': [], 'MaxVelo': [], 'Filename': []})

    size = np.array([15,15])

    while(fussgaenger.shape[0]>0):
        Zeitraum = fussgaenger['datum'].loc[0]
        image = np.zeros((resolution,resolution,4))
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
                image[xidx,yidx,2] = image[xidx,yidx,2] + 64
        for i in range(0,tempvelo.shape[0]-1):
            if x[0] <= tempvelo.Longitude.loc[i] <= x[-1] and y[0] <= tempvelo.Lattitude.loc[i] <= y[-1]:
                xidx = (np.abs(tempvelo.Longitude.loc[i]-x)).argmin()
                yidx = (np.abs(tempvelo.Lattitude.loc[i]-y)).argmin()
                image[xidx,yidx,1] = image[xidx,yidx,1] + tempvelo['total'].iloc[i]
                image[xidx,yidx,3] = image[xidx,yidx,3] + 64
        Filename = str(Zeitraum.date())+'-'+str(Zeitraum.hour).zfill(2) + str(Zeitraum.minute).zfill(2)
        Maximum = Maximum.append({'Datum' : Zeitraum, 'MaxFuss' : image[:,:,0].max()/100., 'MaxVelo' : image[:,:,1].max()/100., 'Filename' : Filename+'.jpg'},ignore_index=True)
        if CreateImage:
            for i in range(0,4):
                if (image[:,:,i].max() > 0.):
                    image[:,:,i] = sigma**2*gaussian_filter(image[:,:,i],sigma)
                    if i < 2:
                        image[:,:,i] = (image[:,:,i]/image[:,:,i].max()*255.).round()
            imped = resize(image[:,:,0],size*3,preserve_range=True).round().astype('uint8')
            imcycl = resize(image[:,:,1],size*2,preserve_range=True).round().astype('uint8')
            impedc = resize(image[:,:,2],size,preserve_range=True).round().astype('uint8')
            imcyclc = resize(image[:,:,3],size,preserve_range=True).round().astype('uint8') 
            gsimage = np.append(imped,np.append(imcycl,np.append(impedc,imcyclc,axis=0),axis=1),axis=0)  #compactify into single image
            imageio.imwrite(path+'Images/'+Filename+'.jpg', gsimage)
    Maximum.to_csv(path+"MaxData"+year+"final.csv",index=False)  

if __name__ == "__main__":
    CreateTrafficData('2017')
        
        
    
        
    
    





