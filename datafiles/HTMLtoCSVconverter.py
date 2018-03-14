import pandas as pd

df = pd.read_html("Weather2017.html",header=0)[0]

df.to_csv("Weather2017parsed.csv")