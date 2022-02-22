import torchvision.models as models

from smartsim.ml.torch import DynamicDataGenerator, DataLoader

import torch
import torch.nn as nn
import torch.optim as optim


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    training_set = DynamicDataGenerator(cluster=False,
                                 shuffle=True,
                                 batch_size=32,
                                 init_samples=False)
    trainloader = DataLoader(training_set,
                             batch_size=None,
                             num_workers=2)
    model = models.mobilenet_v2().double().to('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    print("Started training")

    for epoch in range(50):  # loop over the dataset multiple times

        running_loss = 0.0
        epoch_running_loss = 0.0
        output_period = 100
        print(f"Epoch {epoch}")
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

            if i % output_period == (output_period-1):    # print every "output_period" mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / output_period))
                running_loss = 0.0

        
        print('[%d, %5d] loss: %.3f' %
            (epoch + 1, i + 1, epoch_running_loss / (i+1)))
        epoch_running_loss = 0.0
                
    print('Finished Training')