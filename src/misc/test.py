import torch
import torch.nn as nn

# Fake data
x = torch.randn(100, 10)
y = torch.randn(100, 1)

# Model
model = nn.Sequential(
  nn.Linear(10, 32),
  nn.ReLU(),
  nn.Linear(32, 1)
)

loss_fn = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr = 1e-3)

# Training loop
for epoch in range(100):
  pred = model(x)                 # forward pass

loss = loss_fn(pred, y)         # compute loss

opt.zero_grad()                 # clear old gradients
loss.backward()                 # backpropagation
opt.step()                      # update weights

print(loss.item())
