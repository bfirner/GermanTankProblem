import torch

net = torch.randn(6)
# With random initialization the output is nearly always 0 because of the ReLUs, so try to fight
# that by making it more likely that the weights are positive.
net = torch.abs(net)
net.requires_grad_(True)
optimizer = torch.optim.SGD([net], lr=10e-7)

loss_fn = torch.nn.L1Loss()

relu = torch.nn.LeakyReLU()

def forward(net, x):
    return relu(
        relu(net[0] * x[0] + net[1] * x[1]) * net[2] +
        relu(net[3] * x[0] + net[4] * x[1]) * net[5])


x = torch.ones((4, 2))
x[0, 0] = 100
x[0, 1] = 200

x[1, 0] = 200
x[1, 1] = 100

x[2, 0] = 100
x[2, 1] = 101

x[3, 0] = 201
x[3, 1] = 200

for epoch in range(100000):
    # Use one of the four states as input
    netin = x[epoch%4]
    out = forward(net, netin)
    # Optimizer to output the value of x[0] if x[0] > x[1], and 0 otherwise
    if netin[0] > netin[1]:
        expected = torch.max(x)
    else:
        expected = torch.zeros(1)[0]
    loss = loss_fn(out, expected)
    loss.backward()
    if epoch % 1000 == 0:
        print("")
    if epoch % 1000 < 4:
        print(f"Weights are {net}")
        print(f"Gradients are {net.grad}")
        print(f"Output is {out}, with loss {loss}")
    optimizer.step()
    optimizer.zero_grad()
