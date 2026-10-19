import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch import nn

PROJECT_DIR = Path(__file__).resolve().parent.parent # project/ folder
sys.path.insert(0, str(PROJECT_DIR)) # so that "Modals" package can be found

from Modals.SimpleLinearRegressionModal import SimpleLinearRegressionModal


def main():
    data = pd.read_csv(PROJECT_DIR / "06-study_hours_grades.csv")

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

    loss_fn = nn.MSELoss() #we select a loss function for our model
    optimizer = torch.optim.SGD(params = simpleModal.parameters(), lr = 0.001) #we select an optimizer for our model
    epochs = 200 # we select how many times we will train our model

    train_loss_values = []
    test_loss_values = []
    epochCount = []

    #we train our model for the number of epochs we selected
    for epoch in range(epochs):
        simpleModal.train() #we should set our model to train mode

        y_pred = simpleModal(X_train) #we make a prediction with our model

        loss = loss_fn(y_pred,Y_train) #we calculate the loss value

        optimizer.zero_grad() #we set the gradients to zero
        loss.backward() #we calculate the gradients with derivative of the loss function with respect to the weights
        optimizer.step() #we update the weights

        simpleModal.eval()

        with torch.inference_mode():
            test_pred = simpleModal(X_test)

            test_loss = loss_fn(test_pred, Y_test)

            if epoch % 5 == 0:
                epochCount.append(epoch)
                train_loss_values.append(loss.item())
                test_loss_values.append(test_loss.item())
                print(f"Epoch: {epoch}, Train Loss: {loss}, Test Loss: {test_loss}")

    simpleModal.eval()
    with torch.inference_mode():
        y_preds = simpleModal(X_test)

    plt.scatter(X_train,Y_train,c = "b" , s= 10, label = "Training Data")
    plt.scatter(X_test,Y_test,c = "g" , s= 10, label = "Test Data")
    plt.scatter(X_test,y_preds,c = "r" , s= 10, label = "Prediction Data")
    plt.show()


if __name__ == "__main__":
    main()
