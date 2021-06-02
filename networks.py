import torch

# Mostly the example network from pytorch docs
class RegularNet(torch.nn.Module):
    """Regular network."""

    def __init__(self, num_inputs, num_outputs):
        """Initialize the demonstration network.
        """
        super(RegularNet, self).__init__()
        layer_sizes = [
            num_inputs,
            num_inputs**2,
            num_inputs,
            num_inputs,
            num_outputs]

        self.net = torch.nn.ModuleList()
        self.net.append(torch.nn.Linear(in_features = layer_sizes[0], out_features = layer_sizes[1], bias = True))
        self.net.append(torch.nn.ReLU())
        self.net.append(torch.nn.Linear(in_features = layer_sizes[1], out_features = layer_sizes[2], bias = True))
        self.net.append(torch.nn.ReLU())
        self.net.append(torch.nn.Linear(in_features = layer_sizes[2], out_features = layer_sizes[3], bias = True))
        self.net.append(torch.nn.Linear(in_features = layer_sizes[3], out_features = layer_sizes[4], bias = True))

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x

class MaxNetwork(RegularNet):
    """Basically the regular network so that they both work with the presolve function, but this
    network will output in the next to last layer.
    """

    def __init__(self, num_inputs):
        """Initialize the network.
        """
        super(MaxNetwork, self).__init__(num_inputs, num_outputs=1)

    def forward(self, x):
        """Override the forward pass to skip the last layer.
        """
        for layer in self.net[0:-1]:
            x = layer(x)
        # Only return the first element of the next to last layer.
        return x[:,0:1]

def presolve(network, num_inputs, freeze_layers = [4, 5]):
    """Set the weights of a RegularNet to the frequentist approach.

    Arguments:
        network (RegularNet) : Instance of RegularNet to presolve
        num_inputs (int)     : The number of inputs to the network
        freeze_layers ([int]): List of layers to set required_grad_ to False.
    """
    # The first layer computes x_i - x_j for all pairs of inputs, i and j where i /= j.
    # We will also need to retrieve the inputs again later, so pass them through.
    net = network.net
    with torch.no_grad():
        net[0].bias.fill_(0)
        net[0].weight.fill_(0)
        # Perform x_i - x_j for all input pairs
        for i in range(num_inputs):
            for j in range(num_inputs):
                offset = node_num = i * num_inputs
                if i == j:
                    # Copy the original value through to the next layer.
                    net[0].weight[offset+j][i] = 1.0
                else:
                    # Subtract the ith value from the jth value
                    net[0].weight[offset+j][j] = 1.0
                    net[0].weight[offset+j][i] = -1.0

        # The values now go through the ReLU. Any negative value will turn to 0 after the ReLU. If the
        # ith element was greater than all other elements, then all of those outputs turn to 0.
        # Perform x_i + -N*ReLU(x_i - x_j), where N is an arbitrary large number.
        net[2].bias.fill_(0)
        net[2].weight.fill_(0)
        for i in range(num_inputs):
            for j in range(num_inputs):
                # This is the offset to the difference section of element i
                src_offset = i * num_inputs
                if i == j:
                    # Copy the original value through to the next layer.
                    net[2].weight[i][src_offset + j] = 1.0
                else:
                    # If this value is positive then the jth element was larger than the ith element.
                    # This means that i is not the maximum element, so we will suppress the output.
                    net[2].weight[i][src_offset + j] = -5000

        # The values go through a ReLU layer, and all non-maximal elements are suppressed to 0 by the
        # large negative weight applied.
        # To create the final output, the desired equation is: m + (m-k)/k, where m is the max and k is
        # the number of inputs. The first node will pass the max through, and the second will perform
        # m/k-1.
        net[4].bias.fill_(0)
        net[4].weight.fill_(0)
        net[4].weight[0].fill_(1.0)
        net[4].weight[1].fill_(1.0 / num_inputs)
        net[4].bias[1] = -1
        ## The final layer adds the two operations of the previous layer. We assume that there is a
        ## single output.
        net[5].bias.fill_(0)
        net[5].weight.fill_(0)
        net[5].weight[0][0:2].fill_(1.0)
    for layer_idx in freeze_layers:
        net[layer_idx].requires_grad_(False)


class LatentNet(torch.nn.Module):
    """Latent network."""

    def __init__(self, num_inputs, num_outputs, latent_size):
        """Initialize the latent network.
        """
        super(LatentNet, self).__init__()
        # The latent size should be enough to capture the latent space, but not enough to just pass
        # the answer around.
        self.latent_size = latent_size
        self.net = RegularNet(num_inputs = num_inputs + latent_size, num_outputs = num_outputs)

        # Now a latent generator
        self.latent_net = torch.nn.ModuleList()
        self.latent_net.append(torch.nn.Linear(in_features = num_inputs, out_features = 16 * latent_size))
        self.latent_net.append(torch.nn.Linear(in_features = 16 * latent_size, out_features = 8 * latent_size))
        #self.latent_net.append(torch.nn.BatchNorm1d(num_features = 4 * latent_size, affine=False))
        self.latent_net.append(torch.nn.ReLU())
        self.latent_net.append(torch.nn.Linear(in_features = 8 * latent_size, out_features = latent_size))
        # The output of the latent network will be sent through torch.where to make it a one hot
        # vector and prevent information leakage.

    def forward(self, x):
        """Forward x and return both the output and the latent vector."""
        # Generate latent inputs, then forward.
        latent = self.forward_get_latent(x)
        x = self.forward_with_latent(x, latent)
        # Return both the prediction output and the latent prediction.
        return x, latent

    def forward_get_latent(self, x):
        """Forward x although through the latent part of the network."""
        latent = x
        for layer in self.latent_net:
            latent = layer(latent)
        # Turn into a one hot vector
        latent = torch.where(
            latent > 0.5, torch.ones(latent.size()).cuda(), torch.zeros(latent.size()).cuda())
        return latent

    def forward_with_latent(self, x, latent):
        """Forward with batch with the given latent input."""
        # Concat the latent vector with the input
        x = torch.cat((x, latent), dim=1)
        x = self.net(x)
        return x

def train_step(net, batch, loss_fn, labels, optimizer, print_grad = False):
    out = net.forward(batch)
    loss = loss_fn(out, labels)
    loss.backward()
    if print_grad:
        print("Inputs:")
        print(f"\t{batch}")
        print("Outputs:")
        print(f"\tDesired output {labels}")
        print(f"\tActual output {out}")
        print("Gradient:")
        print(f"\tLayer 0 weight: {net.net[0].weight.grad}")
        print(f"\tLayer 0 bias: {net.net[0].bias.grad}")
        print(f"\tLayer 2 weight: {net.net[0].weight.grad}")
        print(f"\tLayer 2 bias: {net.net[0].bias.grad}")
    optimizer.step()
    optimizer.zero_grad()
    return loss.detach()

def latent_train_step(net, batch, loss_fn, labels, optimizer, true_prob=0.5):
    # Here the latent vector will just be a prediction of the true value being greater than certain
    # values.
    with torch.no_grad():
        batch_size = labels.size(0)
        desired_latent_1 = torch.where(
            labels > 250.0, torch.ones((batch_size, 1)).cuda(), torch.zeros((batch_size, 1)).cuda())
        desired_latent_2 = torch.where(
            labels > 500.0, torch.ones((batch_size, 1)).cuda(), torch.zeros((batch_size, 1)).cuda())
        desired_latent_3 = torch.where(
            labels > 750.0, torch.ones((batch_size, 1)).cuda(), torch.zeros((batch_size, 1)).cuda())
        desired_latent_4 = torch.where(
            labels > 1000.0, torch.ones((batch_size, 1)).cuda(), torch.zeros((batch_size, 1)).cuda())
        desired_latent_5 = torch.where(
            labels > 1250.0, torch.ones((batch_size, 1)).cuda(), torch.zeros((batch_size, 1)).cuda())

        desired_latent = torch.cat(
            (desired_latent_1, desired_latent_2, desired_latent_3, desired_latent_4, desired_latent_5), dim=1)

    # Train with the latent vector coming from truth sometimes. This gives the network access to
    # "privileged information" and forces the latent prediction to conform move towards the real
    # values.
    latent = net.forward_get_latent(batch)
    # Create the latent input tensor, sampling from the network prediction or true values as
    # indicated by the true_prob input.
    # Notice that we are recreating this tensor each time around, we could just create it once if we
    # were worried about speed.
    latent_input = torch.where(
        torch.zeros(latent.size(0), 1).random_(0, 1000).cuda() < true_prob*1000, latent, desired_latent)
    out = net.forward_with_latent(batch, latent_input)

    # Use the latent vector to compute an additional loss
    # Apply some weights to the latent vector, otherwise its loss won't show up at all.
    loss = loss_fn(out, labels) + loss_fn(latent, desired_latent) * labels.max()/latent.size(1)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.detach()
