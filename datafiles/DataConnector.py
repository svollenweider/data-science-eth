import pandas as pd
import numpy as np
from ExternalDataHTMLtoCSVconverter import CreateExternalData
from TramDataCompressor import TramData
from Fussgaenger.TrafficCreator import CreateTrafficData
from multiprocessing import Process,Value,Queue
from datetime import datetime

#only works directly in python prompt, ipython does not work

#Determines probability that an entry is listed in the Training file, separated to "on time" and "late"
probabilityontime = 0.01
probabilitylate = 0.1

#name of output file
mode = "Training" #eval or training

#links delay with a label
def labelling(delay): 
    if delay<-30: return -1 #tram to early
    if delay<=10: return 0 # "No delay"
    if delay<30: return 1 # "Delay between 0 and 30s"
    if delay<60: return 2 # "Delay between 30 and 60 s"
    if delay<150: return 3 # "Delay between 1 min and 2.5 min"
    if delay<300: return 4 # "Delay between 2.5 min and 5 min"
    if delay<480:  return 5 # "Delay between 5 min and 8 min"
    return 6 # "Delay greater than 8 min"

#A thread that reads a line of data determined by counter, then creates a row of features and sends it to the combiner
def worker(counter,q,year,no): #worker that takes an element and puts it into a queue
    print("Worker " + str(no) + " started")
    start = datetime.now() # Used to estimate runtime
    #read all input files to append
    Traffic = pd.read_csv("Fussgaenger/MaxData"+year+"final.csv",header=0,index_col=['Datum'],parse_dates=True,infer_datetime_format=True)
    External = pd.read_csv("ExternalData/ExternalData"+year+"final.csv",header=0,index_col=['Datum Uhrzeit'],parse_dates=True,infer_datetime_format=True)
    Tram = pd.read_csv("SollIst/Results/VBZDataLine9_"+year+".csv",header=0,index_col=['departure'],parse_dates=True,infer_datetime_format=True)
    print("Worker " + str(no) + " Data read")
    
    #get the line to work in from the counter variable and iterate it by 1
    with counter.get_lock():
        localit = counter.value
        counter.value += 1
    #check if row to edit is still in the table and repeat until it isn't
    while localit < Tram.shape[0]:
        #get the time of the row
        Time = Tram.index.values[localit]
        #get the row 
        row = Tram.iloc[localit]
        #Using the time, get the external data (weather eg.) one hour earlier
        ExternalRow = External[External.index<Time-pd.Timedelta('1h')]
        #get new value of the counter
        with counter.get_lock():
            oldlocalit = localit
            localit = counter.value
            counter.value += 1
        #every 0.5% of the way print an output with the estimated completion
        if localit % int(Tram.shape[0] *0.005) == 0:
            percentage = localit*1./Tram.shape[0]
            print('{:3.1f}'.format(percentage*100) +" %") 
            print("Estimated completion: " + (start+(datetime.now()-start)/percentage).strftime('%H:%M:%S') )
        #determine probability to include in the final file, if new use 
        if labelling(row["Delay"]) <= 10:
            probability = probabilityontime
        else: 
            probability  = probabilitylate
        if np.random.random()>probability or ExternalRow.empty:
            continue
        #find tram number
        fahrzeug = row['fahrzeug']
        #get the  Tram data from one hour earlier to get the delay at that point
        TrafficRow = Traffic[Traffic.index<Time-pd.Timedelta('1h')]
        TramEarly = Tram[np.all(np.vstack([Time-pd.Timedelta('1h10m')<Tram.index,Tram.index<Time-pd.Timedelta('1h'),row['fahrzeug']==Tram['fahrzeug']]),axis=0)]        
        if TramEarly.empty:
            prevDelay = 0 #if tramnumber can not be found set delay to 0
        else:
            prevDelay = TramEarly.iloc[-1]['Delay']
        #make an array and put in a queue for the combiner to work with
        queput = np.array(row.drop('fahrzeug').tolist()+TrafficRow.iloc[-1].tolist()+ExternalRow.iloc[-1].tolist()+[prevDelay]+[row["Delay"], labelling(row["Delay"])],dtype=object)
        q.put(queput)
        if localit % 50000 == 0:
            q.put("Save")
    q.put("END")
    print("Worker " + str(no) + " ended")
    
def combiner(q,NoProcesses,counter,pickup = False,name="Trainigdata"): #takes elements from queue and puts them in DataFrame
    #number of workers that are finished
    ended = 0
    saverproc = Process()
    print("Combiner started")
    #columns for the dataframe
    dfcolumns = ['richtung','Stop','Distance','Delay','Filename','MaxFuss','MaxVelo','Days',
 'Uhrzeit','Weekday','Specialday','Lufttemperatur','Windgeschwindigkeit','Windrichtung','Luftdruck','Niederschlag','Luftfeuchte','delayprior','label','altlabel']
    #can reload data
    if pickup:
        AllData = pd.read_csv("FinalData/AllData.csv",index=range(3000000),header=0,parse_dates=True,infer_datetime_format=True)
    else:
        AllData = pd.DataFrame(index=range(3000000),columns=dfcolumns)
    location = 0
    #create empty array to put the data in
    ValueArray = np.empty((3000000,len(dfcolumns)),dtype=object)
    #while not all workers have ended
    while NoProcesses > ended:
        #grab item from queue
        element = q.get()
        #check if item is  a string and perform action to end the process or save
        if type(element) is str:
            if element == "END":
                ended += 1
            if element == "Save":
                #start a process that saves the array temporarely
                print("Saving at " + str(counter.value))
                if saverproc.is_alive(): print("Warning, saving to slow, skipping")
                else:
                    saverproc = Process(target=saver, args=(counter.value,ValueArray,dfcolumns))
                    saverproc.start()
        else:          
            ValueArray[location] = element
            location +=1
    #create dataframe and save it        
    SortedList = ['Filename','richtung','Distance','MaxFuss','MaxVelo','Days','Uhrzeit','Weekday','Specialday','Lufttemperatur','Windgeschwindigkeit','Windrichtung','Luftdruck','Niederschlag','Luftfeuchte','delayprior','label','altlabel']
    SortedData = pd.DataFrame(data=ValueArray,columns=dfcolumns)[SortedList]
    print("Saving")
    (SortedData.dropna(axis=0)).to_csv("FinalData/"+mode+"Datafinal.csv",index=False)
    print("Saved")
    
def saver(ctr,VA,dfcolumns):
    np.savetxt("FinalData/LastElement.csv",np.array([ctr]))
    (pd.DataFrame(data=VA,columns=dfcolumns)).to_csv("FinalData/AllData.csv",index=False)  

    
if __name__ == '__main__':
    #years we want to have the data from (list)=
    years = ['2017']
    #Has the data to be created or can it be loaded from files?
    Create = False
    #should we continue from a checkpoint or create the data from scratch?
    ContinueFromCheckpoint = False
    #number of processes that execute the worker function
    NoofProcesses = 7
    if ContinueFromCheckpoint:
        counter=Value('i',np.loadtxt("FinalData/LastElement.csv",dtype=int)[0])
    else:
        counter=Value('i',0)
    #create a Queue with 200 elements
    que = Queue(200)
    for year in years:
        #if create, create other data
        if Create:
            print('WeatherData')
            CreateExternalData(year)
            print('TramData')
            TramData(year)
            print('TrafficData')
            CreateTrafficData(year,CreateImage=False) 
        print('Connect Data')
        #start processes for worker and combinator
        procs = [Process(target=worker, args=(counter,que,year,i)) for i in range(NoofProcesses)]
        Comb = Process(target=combiner, args=(que,NoofProcesses,counter,ContinueFromCheckpoint))
        for p in procs: p.start()
        Comb.start()
        for p in procs: p.join()
        Comb.join()
        
