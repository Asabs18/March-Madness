import pandas as pd
import numpy as np

def readData(filename):
    df = pd.read_csv(filename)
    return df

data = readData('../data/MarchMadnessData.csv')

def findGame(index):
    global data
    return f"{data.iloc[index]['Team1']} V.S {data.iloc[index]['Team2']} \nScore: {data.iloc[index]['Score1']} to {data.iloc[index]['Score2']}"

def main():
    global data
    print(findGame(1000))

if __name__ == "__main__":
    main()