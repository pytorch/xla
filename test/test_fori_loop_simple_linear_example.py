import os
# import unittest
# from typing import Callable, Dict, List

import torch
import torch_xla
# We need to import the underlying implementation function to register with the dispatcher
import torch_xla.experimental.fori_loop
from torch_xla.experimental.fori_loop import fori_loop
# from torch._higher_order_ops.while_loop import while_loop
import torch_xla.core.xla_model as xm
# import torch_xla.core.xla_builder as xb

device = xm.xla_device()

l_in = torch.randn(10, device=xm.xla_device())
linear = torch.nn.Linear(10, 20).to(xm.xla_device())
l_out = linear(l_in)
print(l_out)

# import numpy as np
# # create dummy data for training
# x_values = [i for i in range(11)]
# x_train = np.array(x_values, dtype=np.float32)
# x_train = x_train.reshape(-1, 1)

# y_values = [2*i + 1 for i in x_values]
# y_train = np.array(y_values, dtype=np.float32)
# y_train = y_train.reshape(-1, 1)

# # import torch
# from torch.autograd import Variable

# class linearRegression(torch.nn.Module):
#     def __init__(self, inputSize, outputSize):
#         super(linearRegression, self).__init__()
#         self.linear = torch.nn.Linear(inputSize, outputSize)

#     def forward(self, x):
#         out = self.linear(x)
#         return out

# # --- training ---
# inputDim = 1        # takes variable 'x' 
# outputDim = 1       # takes variable 'y'
# learningRate = 0.01 * xm.xrt_world_size()
# epochs = 10 # 100

# model = linearRegression(inputDim, outputDim).to(device)
# # model = MNIST().to(device)
# ##### For GPU #######
# if torch.cuda.is_available():
#     model.cuda()

# criterion = torch.nn.MSELoss() 
# optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

# for epoch in range(epochs):
#     # Converting inputs and labels to Variable
#     if torch.cuda.is_available():
#         inputs = Variable(torch.from_numpy(x_train).cuda())
#         labels = Variable(torch.from_numpy(y_train).cuda())
#     else:
#         inputs = Variable(torch.from_numpy(x_train))
#         labels = Variable(torch.from_numpy(y_train))

#     # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
#     optimizer.zero_grad()

#     # get output from the model, given the inputs
#     outputs = model(inputs)

#     # get loss for the predicted output
#     loss = criterion(outputs, labels)
#     print(loss)
#     # get gradients w.r.t to parameters
#     loss.backward()

#     # update parameters
#     # optimizer.step()
#     xm.optimizer_step(optimizer)

#     print('epoch {}, loss {}'.format(epoch, loss.item()))

# --- while simple test case ---

# device = xm.xla_device()

lower = torch.tensor([2], dtype=torch.int32, device=device)
upper = torch.tensor([52], dtype=torch.int32, device=device)
one_value = torch.tensor([1], dtype=torch.int32, device=device)
init_val = torch.tensor([1], dtype=torch.int32, device=device)

def body_fun(a, b):
  return torch.add(a, b) # [0])

lower_, upper_, res_ = fori_loop(upper, lower, body_fun, one_value, init_val)

print("lower_: ", lower_)
print("upper_: ", upper_)
print("res_: ", res_)

# --- test ---
# for epoch in range(epochs):
#     with torch.no_grad(): # we don't need gradients in the testing phase
#         if torch.cuda.is_available():
#             predicted = model(Variable(torch.from_numpy(x_train).cuda())).cpu().data.numpy()
#         else:
#             predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
#         print(epoch, "-th prediction finised") # ed result: ", predicted)

# print("do one more prediction")
# with torch.no_grad(): # we don't need gradients in the testing phase
#     if torch.cuda.is_available():
#         predicted = model(Variable(torch.from_numpy(x_train).cuda())).cpu().data.numpy()
#     else:
#         predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
#     print(predicted)
# print("finished one more prediction")

# --- draw ---
# plt.clf()
# plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
# plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
# plt.legend(loc='best')
# plt.show()