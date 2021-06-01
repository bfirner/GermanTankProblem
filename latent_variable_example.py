import math
import torch
import random
import sys

from germantankproblem import computeExpectedMaxSN
from networks import (
    LatentNet,
    MaxNetwork,
    RegularNet,
    latent_train_step,
    presolve,
    train_step
)


# This was originally intended as a demonstration of how a latent variable can be used with a neural
# network to solve the German Tank Problem: https://en.wikipedia.org/wiki/German_tank_problem


def makeSerials(num_samples, batch_size, min_serial=1, max_serial=1000):
    """Make batch_size examples with num_samples positive integers from min_serial to max_serial."""
    # The actual maximum serial number for each element of the batch.
    answers = torch.zeros(batch_size, 1).random_(min_serial, max_serial+1)
    # The observed serial numbers, drawn from the multinomial distribution.
    # Create the weights through a few steps. First, create zero weights with an extra column to
    # prevent some indexing from being out of bounds. Set 1 at the highest value and beyond, then
    # subtract from 1 to flip the 0s and 1s and remove the extra column. Final pass to the
    # multinomial.
    weights = torch.zeros(batch_size, max_serial+1)
    weights[torch.arange(answers.size(0)).unsqueeze(1), answers.long()] = 1
    weights = 1 - weights.cumsum(dim=1)[:,:-1]

    observations = torch.multinomial(
        input=weights, num_samples=num_samples, replacement=False).float()
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


    regnet = RegularNet(num_inputs=number_samples, num_outputs=1).cuda()
    regnet.train()
    #reg_optimizer = torch.optim.Adam(regnet.parameters())
    reg_optimizer = torch.optim.SGD(regnet.parameters(), lr=10e-7)

    solvednet = RegularNet(num_inputs=number_samples, num_outputs=1).cuda()
    solvednet.train()
    presolve(solvednet, number_samples, freeze_layers=[2, 4, 5]) # TODO Look at the gradients for layers 0 and 2
    solved_optimizer = torch.optim.SGD(solvednet.parameters(), lr=10e-7)

    latent_inputs=5
    latnet = LatentNet(num_inputs=number_samples, num_outputs=1, latent_size=latent_inputs).cuda()
    latnet.train()
    #lat_optimizer = torch.optim.Adam(latnet.parameters())
    lat_optimizer = torch.optim.SGD(latnet.parameters(), lr=10e-6)

    loss_fn = torch.nn.L1Loss()

    # Now train each of the networks. This will be the maximum N.
    max_number = 1500

    true_latent_sample_rate = 1.0
    for batch_num in range(1, total_batches + 1):
        # Does a bootstrapping period help the latent training? Not really.
        if math.floor(total_batches/10) == batch_num:
            true_latent_sample_rate = 0.8
        if math.floor(total_batches/4)== batch_num:
            true_latent_sample_rate = 0.5
        if total_batches - 500 == batch_num:
            true_latent_sample_rate = 0.1

        if 500 == batch_num:
            reg_optimizer = torch.optim.SGD(regnet.parameters(), lr=10e-7)
            lat_optimizer = torch.optim.SGD(latnet.parameters(), lr=10e-7)

        observations, answers = makeSerials(
            num_samples=number_samples, batch_size=batch_size, min_serial=number_samples,
            max_serial=max_number)

        # Train each of the networks
        #print("Regular network:")
        reg_loss = train_step(regnet, observations, loss_fn, answers, reg_optimizer, print_grad=False)
        #print("Presolved network:")
        solved_loss = train_step(solvednet, observations, loss_fn, answers, solved_optimizer, print_grad=False)
        lat_loss = latent_train_step(
            latnet, observations, loss_fn, answers, lat_optimizer, true_latent_sample_rate)

        # Only useful for debugging.
        if 0 == batch_num%500:
            print(f"Batch {batch_num} loss - regnet:{reg_loss}, solved:{solved_loss}, latent:{lat_loss}")


    # Now evaluate
    batch_size = final_evaluation_size
    regnet.eval()
    solvednet.eval()
    latnet.eval()
    observations, answers = makeSerials(
        num_samples=number_samples, min_serial=number_samples, batch_size=batch_size)

    reg_out = regnet.forward(observations)
    solved_out = solvednet.forward(observations)
    lat_out, latent_vectors = latnet.forward(observations)

    # Make five empty arrays to store some error statistics
    errors = [[] for _ in range(5)]
    for i in range(observations.size(0)):
        samples = observations.long().tolist()

        b_median, b_mean, freq = computeExpectedMaxSN(samples[i], max_number=max_number)

        # Track errors
        errors[0].append(abs(answers[i].item() - b_median))
        errors[1].append(abs(answers[i].item() - freq))
        errors[2].append(abs(answers[i].item() - reg_out[i,0].item()))
        errors[3].append(abs(answers[i].item() - lat_out[i,0].item()))
        errors[4].append(abs(answers[i].item() - solved_out[i,0].item()))

    print("Average absolute errors are:")
    print(f"\tBayes: {sum(errors[0])/batch_size}")
    print(f"\tFrequentist: {sum(errors[1])/batch_size}")
    print(f"\tRegular Network: {sum(errors[2])/batch_size}")
    print(f"\tLatent Network: {sum(errors[3])/batch_size}")
    print(f"\tSolved Network: {sum(errors[4])/batch_size}")
    # Maximum errors are interesting, but are a bit biased by outliers.
    #print("Maximum absolute errors are:")
    #print(f"\tBayes: {max(errors[0])}")
    #print(f"\tFrequentist: {max(errors[1])}")
    #print(f"\tRegular Network: {max(errors[2])}")
    #print(f"\tLatent Network: {max(errors[3])}")
