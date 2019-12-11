import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, X):
        hid = self.fc1(X)
        hid = F.sigmoid(hid)
        hid = self.fc2(hid)
        out = F.sigmoid(hid)

        return out

def get_grads_hook(self, input_grad, output_grad):
    print('grad_in:', input_grad)
    print('grad_out:', output_grad)

mlp = MLP(2, 1, 10)
X = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y = torch.Tensor([[0], [1], [1], [0]])
criterion = nn.MSELoss()

optimizer = optim.SGD(mlp.parameters(), lr=0.05)

for epoch in range(30000):
    optimizer.zero_grad()

    if (epoch + 1) % 1000 == 0:
        bh1 = mlp.fc1.register_backward_hook(get_grads_hook)
        bh2 = mlp.fc1.register_backward_hook(get_grads_hook)
    if (epoch + 1) % 1000 == 0:
        print('1')
    output = mlp(X)
    loss = criterion(output, y)
    loss.backward()
    if (epoch + 1) % 1000 == 0:
        print('2')
    if (epoch + 1) % 1000 == 0:
        print('3')
        bh1.remove()
        bh2.remove()

    optimizer.step()
    if (epoch + 1) % 1000 == 0:
        print('epoch={}, loss={}'.format(epoch + 1, loss.item()), str(output.detach().numpy().reshape(-1).tolist()))

out = mlp(X)
print(out)