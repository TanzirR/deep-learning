import sys
import torch 
import torch.nn as nn
import pandas as pd

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

# Training loop
for i in range(10000):
    optimizer.zero_grad()          # reset gradients
    output = model(X)              # forward pass
    loss = loss_fn(output, y)      # compute loss
    loss.backward()                # backpropagation
    optimizer.step()               # update weights

    if i % 100 == 0:
        print(f"weight: {model.weight}, bias: {model.bias}, loss: {loss}")


#Inference 
X_data = torch.tensor([[5,10000],
                       [2,10000],
                       [5,20000]
                       ], dtype=torch.float32)

prediction = model((X_data - X_mean) / X_std)
print(f"prediction: {prediction * y_std + y_mean}")  #denormalize

