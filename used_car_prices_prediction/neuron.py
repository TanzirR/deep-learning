import os
import torch 
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./data/used_cars.csv")

#clean price col
price = df["price"].str.replace("$","").str.replace(",","").astype(int)
#clean milage col
mileage = df["milage"].str.replace("mi.","").str.replace(",","").astype(int)
#For the model col, find the age
age = df["model_year"].max() - df["model_year"]

# X = torch.tensor([[age],
#                   [mileage]
#                   ], dtype=torch.float32)
# print(X.shape)
#Cant use this: PyTorch will treat them as two samples instead of two features — 
# so this will end up with shape (2, N) instead of (N, 2)

#convert to tensors: Creating X and y data
X = torch.column_stack([
    torch.tensor(age, dtype=torch.float32),
    torch.tensor(mileage, dtype=torch.float32)
])
#this will create (N samples, 2 features)
print(X.shape)
#Normalize the inputs since they are in different scale. The mileage will
#have significant effect compared to the model_age
X_mean = X.mean(axis=0) #mean of each column
X_std = X.std(axis=0)   #std of each column
X = (X - X_mean) / X_std

#.reshape((-1,1)) same as (4009,1)
y = torch.tensor([[price]], dtype=torch.float32).reshape((-1,1))
print(y.shape)
#prices need to be normalized(z standardization)
# Range of price is too large. Fixes gradient explosion 
y_mean = y.mean()
y_std = y.std()
y = (y - y_mean) / y_std

#Initialize the model
model = nn.Linear(2,1)
#loss function for price
loss_fn = nn.MSELoss()
#optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

#Collect the valus of loss in each iteration for plotting later to determine
#the range of training loop 
losses = []

# Training loop
for i in range(2500):
    optimizer.zero_grad()          # reset gradients
    output = model(X)              # forward pass
    loss = loss_fn(output, y)      # compute loss
    loss.backward()                # backpropagation
    optimizer.step()               # update weights

    # if i % 100 == 0:
    #     print(f"weight: {model.weight}, bias: {model.bias}, loss: {loss}")
    losses.append(loss.item()) 

#plot the losses. Tweak the learning rate and number of iterations
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss over Iterations")
plt.show()
# plt.savefig("training_loss.png")

#saving the model input and output normalization parameters
if not os.path.exists("./model"):
    os.makedirs("./model")
torch.save({'X_mean': X_mean,
            'X_std': X_std,
            'y_mean': y_mean,
            'y_std': y_std}, "./model/used_car_price_params.pt")

#saving model weights and bias
torch.save(model.state_dict(), "./model/used_car_price_model.pt")

