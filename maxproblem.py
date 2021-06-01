import math
import torch
import random
import sys

from networks import (
    MaxNetwork,
    presolve,
    train_step
)

def makeMaxBatches(num_samples, batch_size, min_serial=1, max_serial=1000, noise_magnitude=0):
    """Make batch_size examples with num_samples positive integers from min_serial to max_serial."""
    # The actual maximum serial number for each element of the batch.
    ranges = torch.zeros(batch_size, 1).random_(min_serial, max_serial+1)
    # The observed serial numbers, drawn from the multinomial distribution.
    # Create the weights through a few steps. First, create zero weights with an extra column to
    # prevent some indexing from being out of bounds. Set 1 at the highest value and beyond, then
    # subtract from 1 to flip the 0s and 1s and remove the extra column. Final pass to the
    # multinomial.
    weights = torch.zeros(batch_size, max_serial+1)
    weights[torch.arange(ranges.size(0)).unsqueeze(1), ranges.long()] = 1
    weights = 1 - weights.cumsum(dim=1)[:,:-1]

    observations = torch.multinomial(
        input=weights, num_samples=num_samples, replacement=False).float()
    answers, _ = torch.max(observations, dim=1, keepdim=True)
    if 0 < noise_magnitude:
        noise = torch.zeros(answers.size()).random_(to=noise_magnitude)
        answers = answers + noise
    # Return the random values and the maximum in each example.
    return observations.cuda(), answers.cuda()


if __name__ == "__main__":

    # Number of "samples" observed from which the maximum number will be predicted.
    number_samples = 10
    if 1 < len(sys.argv):
        number_samples = int(sys.argv[1])

    # Number of tests to use when computing final performance statistics.
    final_evaluation_size = 100
    if 2 < len(sys.argv):
        final_evaluation_size = int(sys.argv[2])

    # Total number of batches to train
    total_batches = 3000
    if 3 < len(sys.argv):
        total_batches = int(sys.argv[3])

    # Size of the batches. The networks are tiny, so use a big batch size.
    batch_size = 1024
    if 4 < len(sys.argv):
        batch_size = int(sys.argv[4])

    # Noise +/- this value will be added to the correct answers.
    noise_magnitude = 0
    if 5 < len(sys.argv):
        noise_magnitude = int(sys.argv[5])


    maxnet = MaxNetwork(num_inputs=number_samples).cuda()
    maxnet.train()
    max_optimizer = torch.optim.SGD(maxnet.parameters(), lr=10e-7)

    solvednet = MaxNetwork(num_inputs=number_samples).cuda()
    solvednet.train()
    presolve(solvednet, number_samples)
    solved_optimizer = torch.optim.SGD(solvednet.parameters(), lr=10e-7)

    loss_fn = torch.nn.L1Loss()

    # Now train each of the networks. This will be the maximum N.
    max_number = 1500

    for batch_num in range(1, total_batches + 1):

        observations, answers = makeMaxBatches(
            num_samples=number_samples, batch_size=batch_size, min_serial=number_samples,
            max_serial=max_number, noise_magnitude=noise_magnitude)

        # Train each of the networks
        reg_loss = train_step(maxnet, observations, loss_fn, answers, max_optimizer, print_grad=False)
        solved_loss = train_step(solvednet, observations, loss_fn, answers, solved_optimizer, print_grad=False)

        # Only useful for debugging.
        if 0 == batch_num%500:
            print(f"Batch {batch_num} loss - maxnet:{reg_loss}, solved:{solved_loss}")


    # Now evaluate
    batch_size = final_evaluation_size
    maxnet.eval()
    solvednet.eval()
    observations, answers = makeMaxBatches(
        num_samples=number_samples, min_serial=number_samples, batch_size=batch_size,
        noise_magnitude=noise_magnitude)

    reg_out = maxnet.forward(observations)
    solved_out = solvednet.forward(observations)

    # Make two empty arrays to store some error statistics
    errors = [[] for _ in range(2)]
    for i in range(observations.size(0)):
        samples = observations.long().tolist()

        # Track errors
        errors[0].append(abs(answers[i].item() - reg_out[i,0].item()))
        errors[1].append(abs(answers[i].item() - solved_out[i,0].item()))

    print("Average absolute errors are:")
    print(f"\tRegular Network: {sum(errors[0])/batch_size}")
    print(f"\tSolved Network: {sum(errors[1])/batch_size}")
