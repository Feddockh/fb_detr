import torch
from torchviz import make_dot

# Define a simple model
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNet()
dummy_input = torch.randn(1, 10)
output = model(dummy_input)

# Generate and save the visualization
dot = make_dot(output, params=dict(model.named_parameters()))
dot.format = 'png'
dot.render('simple_net_graph')