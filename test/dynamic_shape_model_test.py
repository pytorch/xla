from sklearn.datasets import make_blobs
import torch, torch_xla
import torch_xla.core.xla_model as xm
import numpy

pd = torch._C._EnablePythonDispatcher()
dev = xm.xla_device()

# CREATE RANDOM DATA POINTS
def blob_label(y, label, loc): # assign labels
    target = numpy.copy(y)
    for l in loc:
        target[y == l] = label
    return target

# SIMPLE OPS TEST #
def simple_test():
    a1 = torch.tensor([[1,0,0,5,0,6]], device=dev)
    a2 = torch.nonzero(a1)
    a2.shape
    torch_xla._XLAC._get_xla_tensor_dimension_size(a2,0)
    a3 = torch.t(torch.tensor([[1,0,0,5,0,6]], device=dev))
    a3.shape
    torch.Size([6, 1])
    a4 = a3.expand(a2.shape)
    torch_xla._XLAC._get_xla_tensor_dimension_size(a4,0)

# SIMPLE NN MODEL
class Feedforward(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, 1)
            self.sigmoid = torch.nn.Sigmoid()
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            output = self.sigmoid(output)
            return output

# CREATE FAKE TRAIN DATA
x_train, y_train = make_blobs(n_samples=40, n_features=2, cluster_std=1.5, shuffle=True)
x_train = torch.Tensor(x_train)
y_train = torch.Tensor(blob_label(y_train, 0, [0]))
y_train = torch.Tensor(blob_label(y_train, 1, [1,2,3]))

# CREATE FAKE TEST DATA
x_test, y_test = make_blobs(n_samples=10, n_features=2, cluster_std=1.5, shuffle=True)
x_test = torch.Tensor(x_test)
num_non_zero = len(torch.nonzero(x_test.int()))
x_test = x_test.to(dev)
print(x_test.int())
x_test = torch.nonzero(x_test.int()).float()
y_test = torch.Tensor(blob_label(y_test, 0, [0]))
y_test = torch.Tensor(blob_label(y_test, 1, [1,2,3]))
y_test = torch.cat((y_test, y_test))
y_test = y_test[:num_non_zero]
y_test = y_test.to(dev)

# MODEL SETUP
model = Feedforward(2, 10).to(dev)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# DEBUG
print(x_test)
print(y_test)

# RUN THE FWD PASS
model.eval()
y_pred = model(x_test)
before_train = criterion(y_pred.squeeze(), y_test)
xm.mark_step()
print('Test loss before training' , before_train.item())

# DISABLE PYTHON DISPATCHER FLAG
del pd
