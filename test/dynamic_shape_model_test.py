import torch
import torch_xla.core.xla_model as xm

pd = torch._C._EnablePythonDispatcher()
dev = xm.xla_device()

# SIMPLE NN MODEL
class Feedforward(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.fc1.weight.data.fill_(0.01)
            self.fc1.bias.data.fill_(0.01)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, 1)
            self.fc2.weight.data.fill_(0.01)
            self.fc2.bias.data.fill_(0.01)
            self.sigmoid = torch.nn.Sigmoid()

        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            output = self.sigmoid(output)
            return output

num_features = 2
num_test_samples = 5
x_test = torch.ones(num_test_samples, num_features)
x_test[0][0] = 0
y_test = torch.ones(num_test_samples*2)
y_test[0] = 0

x_test = x_test.to(dev)
x_test = torch.nonzero(x_test.int()).float()
y_test = y_test.to(dev)
y_test = torch.nonzero(y_test.int()).float().squeeze()

# MODEL SETUP
hidden_size = 10
model = Feedforward(num_features, hidden_size).to(dev)
criterion = torch.nn.BCELoss()

# RUN THE FWD PASS
# print(model)
model.eval()
with torch.no_grad():
    y_pred = model(x_test)
    before_train = criterion(y_pred.squeeze(), y_test)
    xm.mark_step()
    print('Test loss before training' , before_train.item())

# DISABLE PYTHON DISPATCHER FLAG
del pd

