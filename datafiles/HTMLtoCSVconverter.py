import pandas as pd

df = pd.read_html("Weather2017.html",header=0)[0]

DataColumnNamestemp = ['Datum Uhrzeit','Lufttemperatur','Windgeschwindigkeit','Windrichtung','Luftdruck','Niederschlag','Luftfeuchte']

df.columns = DataColumnNamestemp

df['Datum'], df['Uhrzeit'] = df['Datum Uhrzeit'].str.split(' ', 1).str

ColumnNames = ['Datum','Uhrzeit', 'Lufttemperatur', 'Windgeschwindigkeit','Windrichtung', 'Luftdruck', 'Niederschlag', 'Luftfeuchte']

df = df[ColumnNames]

df.to_csv("Weather2017final.csv",index=False)
