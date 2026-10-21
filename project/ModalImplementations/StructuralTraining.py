import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn as nn

PROJECT_DIR = Path(__file__).resolve().parent.parent # project/ folder
sys.path.insert(0, str(PROJECT_DIR)) # so that "Modals" package can be found

#modal
from Modals.LinearRegressionModal import LinearRegressionModal

def main():
    data = pd.read_csv(PROJECT_DIR / "06-study_hours_grades.csv") # our data

    studyHours = torch.tensor(data["study_hours"].values,dtype= torch.float32).unsqueeze(1) # getting study hours
    grades = torch.tensor(data["grade"].values,dtype= torch.float32).unsqueeze(1)  # getting grades

    trainSplit = int(0.8 * len(studyHours)) # %80 data collected for training
    XTrain, YTrain = studyHours[:trainSplit] , grades[:trainSplit] # train datas collected 
    XTest, YTest = studyHours[trainSplit:], grades[trainSplit:] # test datas collected

    torch.manual_seed(42) # manuel random seed

    modal = LinearRegressionModal()
    lossFn = nn.MSELoss()

    LEARNING_RATE = 0.01
    optimizer = torch.optim.SGD(params = modal.parameters(),lr = LEARNING_RATE)

    epochs = 120
    for epoch in range(epochs):
        modal.train()

        YPred = modal(XTrain)
        loss = lossFn(YPred,YTrain)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        modal.eval()
        with torch.inference_mode():
            testPred = modal(XTest)
            testLoss = lossFn(testPred,YTest)

            if(epoch % 5 == 0):
                print(f"Epoch: {epoch}, Train Loss: {loss}, Test Loss: {testLoss}")




if __name__ == "__main__":
    main()
