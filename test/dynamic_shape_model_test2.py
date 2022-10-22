from sklearn.datasets import make_blobs
import torch
import torch_xla.core.xla_model as xm

pd = torch._C._EnablePythonDispatcher()
dev = xm.xla_device()

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
num_features = 2
x_train, y_train = make_blobs(n_samples=40, n_features=num_features, cluster_std=1.5, shuffle=True)
x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)

# CREATE FAKE TEST DATA
num_test_samples = 5
x_test = torch.ones_like(torch.empty(num_test_samples, num_features))
x_test[0][0] = 0
y_test = torch.ones_like(torch.empty(num_test_samples*2-1))

x_test = torch.nonzero(x_test.int()).float().to(dev)
y_test = y_test.to(dev)
print('xw32 debug line41 x_test=', x_test)
print('xw32 debug line42 y_test=', y_test)

# MODEL SETUP
model = Feedforward(2, 10).to(dev)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# RUN THE FWD PASS
model.eval()
y_pred = model(x_test)
before_train = criterion(y_pred.squeeze(), y_test)
xm.mark_step()
print('Test loss before training' , before_train.item())

# DISABLE PYTHON DISPATCHER FLAG
del pd
