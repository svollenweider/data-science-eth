import pandas as pd
import numpy as np

def CreateExternalData(year):
    Dates = {'2017' : {'dates':['2017-01-01','2017-04-14','2017-04-16','2017-04-17','2017-05-01','2017-05-25','2017-06-05','2017-08-01','2017-12-25','2017-12-26']},
             '2016' : {'dates':['2016-01-01','2016-03-25','2016-03-27','2016-03-28','2016-05-01','2016-05-25','2016-05-16','2016-08-01','2016-12-25','2016-12-26']},
             '2015' : {'dates':['2015-01-01','2017-04-14','2017-04-16','2017-04-17','2017-05-01','2017-05-25','2017-06-05','2017-08-01','2017-12-25','2017-12-26']},
             '2014' : {'dates':['2017-01-01','2017-04-14','2017-04-16','2017-04-17','2017-05-01','2017-05-25','2017-06-05','2017-08-01','2017-12-25','2017-12-26']}}

    Specialdays = pd.DataFrame(Dates[year])
    Specialdays['dates'] = pd.to_datetime(Specialdays['dates'])  
    df = pd.read_html("ExternalData/Weather"+year+".html",header=0)[0]
    DataColumnNamesTemp = ['Datum Uhrzeit','Lufttemperatur','Windgeschwindigkeit','Windrichtung','Luftdruck','Niederschlag','Luftfeuchte']
    df.columns = DataColumnNamesTemp

    df['Datum'], df['Uhrzeit'] = df['Datum Uhrzeit'].str.split(' ', 1).str

    df['Uhrzeit'] = pd.to_timedelta(df['Uhrzeit']).dt.seconds/86400.

    df['Datum Uhrzeit'] = pd.to_datetime(df['Datum Uhrzeit'])

    df['Datum'] = pd.to_datetime(df['Datum'])

    df['Weekday'] = df['Datum'].dt.dayofweek/7.

    df['Specialday'] = df['Datum'].isin(Specialdays['dates']).astype(int)

    df['Datum'] = pd.to_timedelta(df['Datum']-df['Datum'][0])

    df['Days'] = (df['Datum']/df['Datum'].iloc[-1]).round(decimals=2)

    df['Luftdruck'] = df['Luftdruck']/1000.

    df['Lufttemperatur'] = df['Lufttemperatur']/20.

    df['Luftfeuchte'] = df['Luftfeuchte']/100.

    df['Windgeschwindigkeit'] = df['Windgeschwindigkeit']/10.

    df['Windrichtung'] = df['Windrichtung']/360.
    
    df['Niederschlag'] = df['Niederschlag'].rolling(window = 4, center=False, min_periods=0).sum()

    ColumnNames = ['Datum Uhrzeit','Days','Uhrzeit', 'Weekday', 'Specialday', 'Lufttemperatur', 'Windgeschwindigkeit','Windrichtung', 'Luftdruck', 'Niederschlag', 'Luftfeuchte']

    df = df[ColumnNames]

    df.to_csv("ExternalData/ExternalData"+year+"final.csv",index=False)
    
if __name__ == "__main__":
    CreateExternalData('2017')
