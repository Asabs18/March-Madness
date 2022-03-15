import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv("../data/MarchMadnessData.csv")

def createSchoolArray(school, xAxis, yAxis):
    schoolArray = []
    for i in range(len(df.index)):
        if str(df.iloc[i]['Team1']) == school and df.iloc[i]['Round'] == 1:
            schoolArray.append((df.iloc[i][f"{yAxis}1"], df.iloc[i][xAxis]))
        elif str(df.iloc[i]['Team2']) == school and df.iloc[i]['Round'] == 1:
            schoolArray.append((df.iloc[i][f"{yAxis}2"], df.iloc[i][xAxis]))
    return schoolArray

def plotData(schoolName, xAxis, yAxis):
    plt.bar(xAxis, yAxis)
    plt.title(f"Round 1 {schoolName} March Madness Scores 1985-2021:")
    plt.xlabel('Year')
    plt.ylabel('Score')
    plt.show()

def main():
    schoolName = input("School Input > ")
    schoolArray = createSchoolArray(schoolName, "Year", "Seed")
    scoreArray = []
    yearArray = []


    for item in schoolArray:
        scoreArray.append(item[0])
        yearArray.append(item[1])

    plotData(schoolName, yearArray, scoreArray)



if __name__ == "__main__":
    main()