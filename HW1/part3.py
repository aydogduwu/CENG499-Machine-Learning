import numpy
import torch
import torch.nn as nn
import numpy as np
import pickle


class MLPModel(nn.Module):
    def __init__(self, hidden_layer_number, neuron_number, activation_function):
        super(MLPModel,self).__init__()
        self.hidden_layer_number = hidden_layer_number
        self.neuron_number = neuron_number
        self.activation_function = activation_function

        if hidden_layer_number == 1:
            self.layer1 = nn.Linear(784,neuron_number)
            self.layer2 = nn.Linear(neuron_number,10)
        if hidden_layer_number == 2:
            self.layer1 = nn.Linear(784,neuron_number)
            self.layer2 = nn.Linear(neuron_number, neuron_number)
            self.layer3 = nn.Linear(neuron_number,10)


        self.activation_function = activation_function()
        self.softmax_function = nn.Softmax(dim=1)

    def forward(self, x):
        if self.hidden_layer_number == 1:
            hidden_output = self.activation_function(self.layer1(x))
            output_layer = self.layer2(hidden_output)

        elif self.hidden_layer_number == 2:
            hidden_output1 = self.activation_function(self.layer1(x))
            hidden_output2 = self.activation_function(self.layer2(hidden_output1))
            output_layer = self.layer3(hidden_output2)

        return output_layer

# function to create model configurations
def create_configurations():
    configurations = []
    configuration1 = MLPModel(1, 16, nn.LeakyReLU)
    optimizer1 = torch.optim.Adam(configuration1.parameters(), 0.0001)
    epoch1 = 250
    print("Model 1 Configurations: Hidden Layer Number: 1, Neuron Number on Each Layer: 16, Learning Rate: 0.0001, Epoch Number: 250, Activation Function: LeakyReLU")

    configuration2 = MLPModel(1, 16, nn.Mish)
    optimizer2 = torch.optim.Adam(configuration2.parameters(), 0.001)
    epoch2 = 500
    print("Model 2 Configurations: Hidden Layer Number: 1, Neuron Number on Each Layer: 16, Learning Rate: 0.001, Epoch Number: 500, Activation Function: Mish")

    configuration3 = MLPModel(1, 32, nn.LeakyReLU)
    optimizer3 = torch.optim.Adam(configuration3.parameters(), 0.0001)
    epoch3 = 250
    print("Model 3 Configurations: Hidden Layer Number: 1, Neuron Number on Each Layer: 32, Learning Rate: 0.0001, Epoch Number: 250, Activation Function: LeakyReLU")

    configuration4 = MLPModel(2, 16, nn.LeakyReLU)
    optimizer4 = torch.optim.Adam(configuration4.parameters(), 0.0001)
    epoch4 = 250
    print("Model 4 Configurations: Hidden Layer Number: 2, Neuron Number on Each Layer: 16, Learning Rate: 0.0001, Epoch Number: 250, Activation Function: LeakyReLU")

    configuration5 = MLPModel(2, 16, nn.Mish)
    optimizer5 = torch.optim.Adam(configuration5.parameters(), 0.001)
    epoch5 = 500
    print("Model 5 Configurations: Hidden Layer Number: 2, Neuron Number on Each Layer: 16, Learning Rate: 0.001, Epoch Number: 500, Activation Function: Mish")

    configuration6 = MLPModel(2, 32, nn.Mish)
    optimizer6 = torch.optim.Adam(configuration6.parameters(), 0.01)
    epoch6 = 500
    print("Model 6 Configurations: Hidden Layer Number: 2, Neuron Number on Each Layer: 32, Learning Rate: 0.01, Epoch Number: 500, Activation Function: Mish")

    configuration7 = MLPModel(2, 32, nn.Mish)
    optimizer7 = torch.optim.Adam(configuration7.parameters(), 0.001)
    epoch7 = 500
    print("Model 7 Configurations: Hidden Layer Number: 2, Neuron Number on Each Layer: 32, Learning Rate: 0.001, Epoch Number: 500, Activation Function: Mish")

    configuration8 = MLPModel(2, 16, nn.Mish)
    optimizer8 = torch.optim.Adam(configuration8.parameters(), 0.01)
    epoch8 = 250
    print("Model 8 Configurations: Hidden Layer Number: 2, Neuron Number on Each Layer: 16, Learning Rate: 0.01, Epoch Number: 250, Activation Function: Mish")

    configuration9 = MLPModel(2, 32, nn.ReLU6)
    optimizer9 = torch.optim.Adam(configuration9.parameters(), 0.001)
    epoch9 = 500
    print("Model 9 Configurations: Hidden Layer Number: 2, Neuron Number on Each Layer: 32, Learning Rate: 0.001, Epoch Number: 500, Activation Function: ReLU6")

    configuration10 = MLPModel(2, 32, nn.ReLU6)
    optimizer10 = torch.optim.Adam(configuration10.parameters(), 0.0001)
    epoch10 = 500
    print("Model 10 Configurations: Hidden Layer Number: 2, Neuron Number on Each Layer: 32, Learning Rate: 0.0001, Epoch Number: 500, Activation Function: ReLU6")

    configurations.append((configuration1, optimizer1, epoch1))
    configurations.append((configuration2, optimizer2, epoch2))
    configurations.append((configuration3, optimizer3, epoch3))
    configurations.append((configuration4, optimizer4, epoch4))
    configurations.append((configuration5, optimizer5, epoch5))
    configurations.append((configuration6, optimizer6, epoch6))
    configurations.append((configuration7, optimizer7, epoch7))
    configurations.append((configuration8, optimizer8, epoch8))
    configurations.append((configuration9, optimizer9, epoch9))
    configurations.append((configuration10, optimizer10, epoch10))

    return configurations


# we load all the datasets of Part 3
x_train, y_train = pickle.load(open("data/mnist_train.data", "rb"))
x_validation, y_validation = pickle.load(open("data/mnist_validation.data", "rb"))
x_test, y_test = pickle.load(open("data/mnist_test.data", "rb"))

x_train = x_train/255.0
x_train = x_train.astype(np.float32)

x_test = x_test / 255.0
x_test = x_test.astype(np.float32)

x_validation = x_validation/255.0
x_validation = x_validation.astype(np.float32)

# and converting them into Pytorch tensors in order to be able to work with Pytorch
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train).to(torch.long)

x_validation = torch.from_numpy(x_validation)
y_validation = torch.from_numpy(y_validation).to(torch.long)

x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test).to(torch.long)

configurations = create_configurations()
repetition_count = 10
loss_function = torch.nn.CrossEntropyLoss()

# create 2d array to store results
rows, cols = 10, 13
result = [[0 for i in range(cols)] for j in range(rows)]
index = 0
for configuration in configurations:  # calculations for each configuration
    for i in range(0, repetition_count):
        validation_accuracy_sum = 0
        for iteration in range(0, configuration[2]): # epoch number loop
            configuration[1].zero_grad()
            predictions = configuration[0](x_train)
            loss_value = loss_function(predictions, y_train)
            loss_value.backward()
            configuration[1].step()

            with torch.no_grad():
                validation_predictions = configuration[0](x_validation)  # predict
                validation_predicted = torch.argmax(validation_predictions, dim=1)  # find index of maximum
                mask = validation_predicted == y_validation  # mask for count
                equals = validation_predicted[mask].size(dim=0)  # number of correctly predicted
                validation_accuracy = (equals / y_validation.size(dim=0)) * 100  # find accuracy
                validation_accuracy_sum += validation_accuracy
        result[index][i] = validation_accuracy_sum / configuration[2]
    index += 1


# assign means, standard deviation, and confidence interval of each model's repetition validation accuracies
for i in range(0, len(result)):
    result[i][10] = float(np.mean(result[i][0:10]))  # mean
    result[i][11] = float(np.std(result[i][0:10]))  # standard deviation
    result[i][12] = ((result[i][10] - 1.96 * (result[i][11] / float(np.sqrt(10)))), (result[i][10] + 1.96 * (result[i][11] / float(np.sqrt(10)))))  # confidence interval


# find maximum mean and index of it
maximum_mean_index = 0
value = result[0][10]
for i in range(0, len(result)):
    if value < result[i][10]:
        value = result[i][10]
        maximum_mean_index = i

# combine train and validation datasets
x_combined = torch.cat((x_train, x_validation), dim=0)
y_combined = torch.cat((y_train, y_validation), dim=0)

# train again the model with maximum mean accuracy
test_results = [0 for x in range(0, repetition_count)]
test_index = 0
for i in range(0, repetition_count):
    test_accuracy_sum = 0
    for iteration in range(0, configurations[maximum_mean_index][2]):
        configurations[maximum_mean_index][1].zero_grad()
        predictions = configurations[maximum_mean_index][0](x_combined)
        loss_value = loss_function(predictions, y_combined)
        loss_value.backward()
        configurations[maximum_mean_index][1].step()

        with torch.no_grad():
            test_predictions = configurations[maximum_mean_index][0](x_test)  # predict
            test_predicted = torch.argmax(test_predictions, dim=1)  # find index of maximum
            mask = test_predicted == y_test  # mask for count
            equals = test_predicted[mask].size(dim=0)  # number of correctly predicted
            test_accuracy = (equals / y_test.size(dim=0)) * 100
            test_accuracy_sum += test_accuracy
    test_results[test_index] = test_accuracy_sum / configurations[maximum_mean_index][2]
    test_index += 1

# find mean, standard deviation, and confidence interval after training the model with highest mean accuracy
best_mean = float(np.mean(test_results))
best_std = np.std(test_results)
best_confidence_interval_lower = best_mean - 1.96 * (best_std / float(np.sqrt(10)))
best_confidence_interval_upper = best_mean + 1.96 * (best_std / float(np.sqrt(10)))

for i in range(0, 10):
    print("Model: %d - Mean Value: %f - Confidence Interval Left: %f - Confidence Interval Right: %f" % (i, result[i][10], result[i][12][0], result[i][12][1]))

print("Best mean value", value)
print("Best Configuration train mean: ", best_mean)
print("Best Configuration Confidence Interval Left: ", best_confidence_interval_lower)
print("Best Configuration Confidence Interval Right: ", best_confidence_interval_upper)

