import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch import nn
from Modals.SimpleLinearRegressionModal import SimpleLinearRegressionModal

data = pd.read_csv(Path(__file__).parent / "06-study_hours_grades.csv")

Y = torch.tensor(data["grade"].values, dtype=torch.float32) # grades (prediction)
X = torch.tensor(data["study_hours"].values, dtype=torch.float32) # study hours (input)

trainSplit = int(len(X) * 0.8) #train split calculated %80

X_train,Y_train = X[:trainSplit],Y[:trainSplit] # train values collected
X_test,Y_test = X[trainSplit:], Y[trainSplit:] #test values collected

#DATAS ARE SEPERATED - WE HAVE TRAIN DATA AND TEST DATA

torch.manual_seed(42)
simpleModal = SimpleLinearRegressionModal()

with torch.inference_mode():
    Y_pred = simpleModal(X_test)

plt.scatter(X_train,Y_train,c = "b" , s = 10, label = "Training Data")
plt.scatter(X_test,Y_test,c = "g" , s = 10 , label = "Test Data")
plt.scatter(X_test,Y_pred,c = "r", s = 10, label = "Prediction")
plt.show()
