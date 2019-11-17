import csv
import pandas as pd

changes_dict = {"UP": 1, "DOWN": 0}

new_rows = []

i = 0
with open('data/electricity_normalized.csv', 'rt') as f:
    reader = csv.reader(f)
    for row in reader:
        new_row = row
        for key, value in changes_dict.items():
            new_row = [ x.replace(key, str(value)) for x in new_row ]
        for key, value in changes_dict.items():
            new_row = [x.replace(key, str(value)) for x in new_row]
        new_rows.append(new_row)
        i += 1
        if(i % 100 == 0):
            print(i)

with open('data/electricity_normalized_numerical.csv', 'wt') as f:
    writer = csv.writer(f)
    writer.writerows(new_rows)

'''
df=pd.read_csv("data/airlines_num.csv",converters={"Airline":int, "AirportFrom":int, "AirportTo":int})
df.to_csv("data/airlines_numerical.csv")
'''
