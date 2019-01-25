import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

#We used the given solution from Sheet08

#load data and transform to tensor with values in range (-1, 1)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
trainset = torchvision.datasets.MNIST(root="./data", train=True,
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)
testset = torchvision.datasets.MNIST(root="./data", train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                         shuffle=False, num_workers=0)


class Ex2Net(nn.Module):
    def __init__(self, hidden_nodes=10):
        """
        Define layers (connections between layers)
        :param hidden_nodes:
        """
        super(Ex2Net, self).__init__()
        self.hidden = hidden_nodes
        # input layer to hidden layer
        self.hidden_layer = nn.Linear(28 * 28, self.hidden)
        # hidden layer to output layer
        self.output_layer = nn.Linear(self.hidden, 10)

    def forward(self, x):
        """
        Define forward propagation.
        Backward propagation definition is done automatically.
        :param x:
        :return:
        """
        # Flatten input
        x = x.view(-1, 28 * 28)
        # calculate and activate hidden layer
        x = F.relu(self.hidden_layer(x))
        # calculate and activate output layer
        x = F.relu(self.output_layer(x))
        return x


def train_with_hidden_layer(nodes, num_epochs=2):
    net = Ex2Net()
    # loss calculation function
    criterion = nn.CrossEntropyLoss()
    
    #Optimizer EXERCISE 4
    optimizer = optim.Adadelta(net.parameters(), lr=0.001)
    #optimizer = optim.Adagrad(net.parameters(), lr=0.001)
    #optimizer = optim.Adam(net.parameters(), lr=0.001)
    #optimizer = optim.RMSprop(net.parameters(), lr=0.001, momentum=0.9)
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #optimizer = optim.LBFGS(net.parameters(), lr=0.001)

    total_loss = 0
    print("TRAINING WITH {} NODE(S)".format(nodes))
    for epoch in range(num_epochs):
        counter = 0
        epoch_loss = 0
        for i, data in enumerate(trainloader):
            images, labels = data
            optimizer.zero_grad()
            # forward propagation for current batch
            outputs = net(images)
            # calculate loss
            loss = criterion(outputs, labels)
            # backward propagation from loss to first layer
            loss.backward()

            #For LGBS optimizer
            #def closure():
            #    return loss

            #set for LGBS step(closure = closure)
            optimizer.step()
            epoch_loss += loss.item()
            counter += 1
        epoch_loss /= counter
        print("Epoch {} loss: {}".format(epoch, epoch_loss))
        total_loss += epoch_loss
    total_loss /= num_epochs
    return net, total_loss


num_nodes = [1, 2, 5, 10, 20]
results = []
for num in num_nodes:
    num_correct = 0
    num_total = 0
    trained_net, total_loss = train_with_hidden_layer(num)
    # testing does not require calculating gradients
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = trained_net(images)
            _, predicted = torch.max(outputs.data, 1)
            num_total += labels.size(0)
            num_correct += (predicted == labels).sum().item()
    test_accuracy = num_correct/num_total
    results.append((num, total_loss, test_accuracy))
print("----------------------- RESULTS -----------------------")
for res in results:
    print("Results using {} node(s) in hidden layer:\n"
          "\tLoss:     {}\n"
          "\tAccuracy: {}".format(res[0], res[1], res[2]))

