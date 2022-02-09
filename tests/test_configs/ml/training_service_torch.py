import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from smartsim.ml.torch import DataLoader, DynamicDataGenerator


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.fc1 = nn.Linear(7200, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output


def check_dataloader(dl):
    assert dl.uploader_name == "test_data"
    assert dl.sample_prefix == "test_samples"
    assert dl.target_prefix == "test_targets"
    assert dl.uploader_info == "auto"
    assert dl.num_classes == 2
    assert dl.producer_prefixes == ["test_uploader"]
    assert dl.sub_indices == ["0", "1"]
    assert dl.verbose == True
    assert dl.replica_rank == 0
    assert dl.num_replicas == 1
    assert dl.address == None
    assert dl.cluster == False
    assert dl.shuffle == True
    assert dl.batch_size == 4


# This test should run without error, but no explicit
# assertion is performed
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    training_set = DynamicDataGenerator(
        cluster=False,
        shuffle=True,
        batch_size=4,
        init_samples=False,
        verbose=True,
        uploader_name="test_data",
    )

    trainloader = DataLoader(training_set, batch_size=None, num_workers=2)

    check_dataloader(training_set)
    model = Net(num_classes=training_set.num_classes).double()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    print("Started training")

    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        epoch_running_loss = 0.0
        output_period = 1
        print(f"Epoch {epoch}")
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].double(), data[1]
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

            if i % output_period == (output_period - 1):
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / output_period)
                )
                running_loss = 0.0

        print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, epoch_running_loss / (i + 1)))
        epoch_running_loss = 0.0

    print("Finished Training")
