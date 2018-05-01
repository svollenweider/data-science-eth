import pandas as pd
import numpy as np
from ExternalDataHTMLtoCSVconverter import CreateExternalData
from TramDataCompressor import TramData
from Fussgaenger.TrafficCreator import CreateTrafficData
from multiprocessing import Process,Value,Queue
from datetime import datetime

#only works directly in python prompt, ipython does not work
probability = 0.001
mode = "eval" #eval or training

def labelling(delay):
    if delay<=0: return 0 # "No delay"
    if delay<30: return 1 # "Delay between 0 and 30s"
    if delay<60: return 2 # "Delay between 30 and 60 s"
    if delay<150: return 3 # "Delay between 1 min and 2.5 min"
    if delay<300: return 4 # "Delay between 2.5 min and 5 min"
    if delay<480:  return 5 # "Delay between 5 min and 8 min"
    return 6 # "Delay greater than 8 min"

def worker(counter,q,year,no): #worker that takes an element and puts it into a queue
    print("Worker " + str(no) + " started")
    start = datetime.now()
    Traffic = pd.read_csv("Fussgaenger/MaxData"+year+"final.csv",header=0,index_col=['Datum'],parse_dates=True,infer_datetime_format=True)
    External = pd.read_csv("ExternalData/ExternalData"+year+"final.csv",header=0,index_col=['Datum Uhrzeit'],parse_dates=True,infer_datetime_format=True)
    Tram = pd.read_csv("SollIst/Results/VBZDataLine9_"+year+".csv",header=0,index_col=['departure'],parse_dates=True,infer_datetime_format=True)
    print("Worker " + str(no) + " Data read")
    localit=0
    with counter.get_lock():
        localit = counter.value
        counter.value += 1
    while localit < Tram.shape[0]:
        Time = Tram.index.values[localit]
        row = Tram.iloc[localit]
        ExternalRow = External[External.index<Time-pd.Timedelta('1h')]
        with counter.get_lock():
            oldlocalit = localit
            localit = counter.value
            counter.value += 1
        if localit % int(Tram.shape[0] *0.005) == 0:
            percentage = localit*1./Tram.shape[0]
            print('{:3.1f}'.format(percentage*100) +" %") 
            print("Estimated completion: " + (start+(datetime.now()-start)/percentage).strftime('%H:%M:%S') )
        if np.random.random()>probability or ExternalRow.empty:
            continue
            if localit % 10000 == 0:
                q.put("Save")
        fahrzeug = row['fahrzeug']
        TrafficRow = Traffic[Traffic.index<Time-pd.Timedelta('1h')]
        TramEarly = Tram[np.all(np.vstack([Time-pd.Timedelta('1h10m')<Tram.index,Tram.index<Time-pd.Timedelta('1h'),row['fahrzeug']==Tram['fahrzeug']]),axis=0)]        
        if TramEarly.empty:
            prevDelay = 0
        else:
            prevDelay = TramEarly.iloc[-1]['Delay']
        queput = np.array(row.drop('fahrzeug').tolist()+TrafficRow.iloc[-1].tolist()+ExternalRow.iloc[-1].tolist()+[prevDelay]+[row["Delay"], labelling(row["Delay"])],dtype=object)
        q.put(queput)
        if localit % 50000 == 0:
            q.put("Save")
    q.put("END")
    print("Worker " + str(no) + " ended")
    
def combiner(q,NoProcesses,counter,pickup = False,name="Trainigdata"): #takes elements from queue and puts them in DataFrame
    ended = 0
    saverproc = Process()
    print("Combiner started")
    dfcolumns = ['richtung','Stop','Distance','Delay','Filename','MaxFuss','MaxVelo','Days',
 'Uhrzeit','Weekday','Specialday','Lufttemperatur','Windgeschwindigkeit','Windrichtung','Luftdruck','Niederschlag','Luftfeuchte','delayprior','label','altlabel']
    if pickup:
        AllData = pd.read_csv("FinalData/AllData.csv",index=range(3000000),header=0,parse_dates=True,infer_datetime_format=True)
    else:
        AllData = pd.DataFrame(index=range(3000000),columns=dfcolumns)
    location = 0
    ValueArray = np.empty((3000000,len(dfcolumns)),dtype=object)
    while NoProcesses > ended:
        element = q.get()
        if type(element) is str:
            if element == "END":
                ended += 1
            if element == "Save":
                print("Saving at " + str(counter.value))
                if saverproc.is_alive(): print("Warning, saving to slow, skipping")
                else:
                    saverproc = Process(target=saver, args=(counter.value,ValueArray,dfcolumns))
                    saverproc.start()
        else:          
            ValueArray[location] = element
            location +=1
            
    SortedList = ['Filename','richtung','Distance','MaxFuss','MaxVelo','Days','Uhrzeit','Weekday','Specialday','Lufttemperatur','Windgeschwindigkeit','Windrichtung','Luftdruck','Niederschlag','Luftfeuchte','delayprior','label','altlabel']
    SortedData = pd.DataFrame(data=ValueArray,columns=dfcolumns)[SortedList]
    print("Saving")
    (SortedData.dropna(axis=0)).to_csv("FinalData/"+mode+"Datafinal.csv",index=False)
    print("Saved")
    
def saver(ctr,VA,dfcolumns):
    np.savetxt("FinalData/LastElement.csv",np.array([ctr]))
    (pd.DataFrame(data=VA,columns=dfcolumns)).to_csv("FinalData/AllData.csv",index=False)  

    
if __name__ == '__main__':
    years = ['2017']
    Create = False
    ContinueFromCheckpoint = False
    NoofProcesses = 7
    if ContinueFromCheckpoint:
        counter=Value('i',np.loadtxt("FinalData/LastElement.csv",dtype=int)[0])
    else:
        counter=Value('i',0)
    que = Queue(200)
    for year in years:
        if Create:
            print('WeatherData')
            #CreateExternalData(year)
            print('TramData')
            TramData(year)
            print('TrafficData')
            CreateTrafficData(year,CreateImage=False) 
        print('Connect Data')
        procs = [Process(target=worker, args=(counter,que,year,i)) for i in range(NoofProcesses)]
        Comb = Process(target=combiner, args=(que,NoofProcesses,counter,ContinueFromCheckpoint))
        for p in procs: p.start()
        Comb.start()
        for p in procs: p.join()
        Comb.join()
        
