import argparse
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

    parser = argparse.ArgumentParser(description="Demonstrate some NN stuff.")
    parser.add_argument(
        "--number_samples",
        "-n",
        help="The number of samples observed from which the maximum is predicted.",
        type=int,
        default=10)
    parser.add_argument(
        "--eval_size",
        help="The number of tests to use when computer final statistics.",
        type=int,
        default=100)
    parser.add_argument(
        "--total_batches",
        help="The total number of batches during training.",
        type=int,
        default=3000)
    parser.add_argument(
        "--batch_size",
        help="The size of each batch. Tiny networks, so use a large batch size.",
        type=int,
        default=1024)
    parser.add_argument(
        "--noise_magnitude",
        help="Add +/- this value to all correct answers.",
        type=int,
        default=0)

    args = parser.parse_args()

    maxnet = MaxNetwork(num_inputs=args.number_samples).cuda()
    maxnet.train()
    max_optimizer = torch.optim.SGD(maxnet.parameters(), lr=10e-7)

    solvednet = MaxNetwork(num_inputs=args.number_samples).cuda()
    solvednet.train()
    presolve(solvednet, args.number_samples)
    solved_optimizer = torch.optim.SGD(solvednet.parameters(), lr=10e-7)

    loss_fn = torch.nn.L1Loss()

    # Now train each of the networks. This will be the maximum N.
    max_number = 1500

    for batch_num in range(1, args.total_batches + 1):

        observations, answers = makeMaxBatches(
            num_samples=args.number_samples, batch_size=args.batch_size, min_serial=args.number_samples,
            max_serial=max_number, noise_magnitude=args.noise_magnitude)

        # Train each of the networks
        reg_loss = train_step(maxnet, observations, loss_fn, answers, max_optimizer, print_grad=False)
        solved_loss = train_step(solvednet, observations, loss_fn, answers, solved_optimizer, print_grad=False)

        # Only useful for debugging.
        if 0 == batch_num%500:
            print(f"Batch {batch_num} loss - maxnet:{reg_loss}, solved:{solved_loss}")


    # Now evaluate
    maxnet.eval()
    solvednet.eval()
    observations, answers = makeMaxBatches(
        num_samples=args.number_samples, min_serial=args.number_samples, batch_size=args.eval_size,
        noise_magnitude=args.noise_magnitude)

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
    print(f"\tRegular Network: {sum(errors[0])/args.eval_size}")
    print(f"\tSolved Network: {sum(errors[1])/args.eval_size}")
