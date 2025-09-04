import pandas as pd
import torch
from torch import nn

#load the normalization params
X_mean = torch.load("./model/used_car_price_params.pt", weights_only=True)['X_mean']
X_std = torch.load("./model/used_car_price_params.pt", weights_only=True)['X_std']
y_std = torch.load("./model/used_car_price_params.pt", weights_only=True)['y_std']
y_mean = torch.load("./model/used_car_price_params.pt", weights_only=True)['y_mean']

#load the model params
model = nn.Linear(2,1)
model.load_state_dict(torch.load("./model/used_car_price_model.pt", weights_only=True))

#Since model not trained here
model.eval()

#Inference
X_data = torch.tensor([[5,10000],
                       [2,10000],
                       [5,20000]
                       ], dtype=torch.float32)

with torch.no_grad(): #no need to track gradients
    prediction = model((X_data - X_mean) / X_std)
    print(f"prediction: {prediction * y_std + y_mean}")  #denormalize