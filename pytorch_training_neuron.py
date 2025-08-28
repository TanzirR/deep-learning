import torch
from torch import nn


# Input: Temp in C
X = torch.tensor([[50],
                  [37.8]], dtype=torch.float32)

# Output: Temp in F
y = torch.tensor([[122],
                  [100]], dtype=torch.float32)

# Linear model
model = nn.Linear(in_features=1, out_features=1)

# Loss function
loss_fn = nn.MSELoss()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

# Training loop
for i in range(250000):
    optimizer.zero_grad()          # reset gradients
    output = model(X)              # forward pass
    loss = loss_fn(output, y)      # compute loss
    loss.backward()                # backpropagation
    optimizer.step()               # update weights

    if i % 200 == 0:
        print(f"weight: {model.weight.item():.4f}, bias: {model.bias.item():.4f}, loss: {loss.item():.4f}")


# Inference
print('=================')
with torch.no_grad():
    test_input = torch.tensor([[60]], dtype=torch.float32)
    test_output = model(test_input)
    print(f"Input (C): {test_input.item()}, Predicted Output (F): {test_output.item()}")