import torchvision.models as models

from smartsim.ml.torch import DynamicDataGenerator, DataLoader

import torch
import torch.nn as nn
import torch.optim as optim

import horovod.torch as hvd

if __name__ == '__main__':

    # Initialize Horovod
    hvd.init()

    hvd_rank = hvd.rank()
    hvd_size = hvd.size()

    # Pin GPU to be used to process local rank (one GPU per process)
    torch.cuda.set_device(hvd.local_rank())

    torch.multiprocessing.set_start_method('spawn')
    training_set = DynamicDataGenerator(cluster=False,
                                 verbose=True,
                                 init_samples=False,
                                 num_replicas=hvd_size,
                                 replica_rank=hvd_rank)

    trainloader = DataLoader(training_set,
                             batch_size=None,
                             num_workers=2)

    model = models.mobilenet_v2().double().to('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001*hvd_size)
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    print(f"Rank {hvd_rank}: Started training")

    for epoch in range(100):  # loop over the dataset multiple times

        running_loss = 0.0
        epoch_running_loss = 0.0
        if hvd_rank == 0:
            print(f"Epoch {epoch}")
        output_period = 100

        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].double().to('cuda'), data[1].to('cuda')
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_running_loss += loss.item()

            if hvd_rank == 0:
                if i % output_period == (output_period-1):    # print every "output_period" mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / output_period))
                    running_loss = 0.0

        if hvd_rank == 0:    
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, epoch_running_loss / (i+1)))
            epoch_running_loss = 0.0
                
    print('Finished Training')