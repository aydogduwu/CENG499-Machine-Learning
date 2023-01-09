import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt


# this function returns the neural network output for a given dataset and set of parameters
from torch.autograd._functions import tensor


def forward_pass(w1, b1, w2, b2, input_data):
    """
    The network consists of 3 inputs, 16 hidden units, and 3 output units
    The activation function of the hidden layer is sigmoid.
    The output layer should apply the softmax function to obtain posterior probability distribution. And the function should return this distribution
    Here you are expected to perform all the required operations for a forward pass over the network with the given dataset
    """
    # indices of slice fow w1
    indices1 = torch.tensor([x for x in range(0, 16)])
    indices2 = torch.tensor([x for x in range(16, 32)])
    indices3 = torch.tensor([x for x in range(32, 48)])
    indices4 = torch.tensor([x for x in range(48, 64)])

    # slice and transpose w1
    w1_1to16_columned = torch.t(torch.index_select(w1, 0, indices1))
    w1_16to32_columned = torch.t(torch.index_select(w1, 0, indices2))
    w1_32to48_columned = torch.t(torch.index_select(w1, 0, indices3))
    w1_48to64_columned = torch.t(torch.index_select(w1, 0, indices4))

    # get first, second and third columns of input data
    indices_input1 = torch.tensor([0])
    indices_input2 = torch.tensor([1])
    indices_input3 = torch.tensor([2])
    input_data_left = torch.index_select(input_data, 1, indices_input1)
    input_data_middle = torch.index_select(input_data, 1, indices_input2)
    input_data_right = torch.index_select(input_data, 1, indices_input3)

    # multiply first 16 weights with b1
    bias_matrix = w1_1to16_columned * b1
    # multiply second 16 weights with first input column
    x1_matrix = w1_16to32_columned * input_data_left
    # multiply third 16 weights with second input column
    x2_matrix = w1_32to48_columned * input_data_middle
    # multiply last 16 weights with third input column
    x3_matrix = w1_48to64_columned * input_data_right

    # sum all results and take sigmoid of it
    input_layer_output = torch.sigmoid(bias_matrix + x1_matrix + x2_matrix + x3_matrix)

    # indices to slice w2, first 3 is left for b2
    hidden_indices1 = torch.tensor([x for x in range(3, 19)])
    hidden_indices2 = torch.tensor([x for x in range(19, 35)])
    hidden_indices3 = torch.tensor([x for x in range(35, 51)])

    # slice w2, transpose it and multiply with output of first layer
    hidden_x1 = torch.t(torch.index_select(w2, 0, hidden_indices1)) * input_layer_output
    hidden_x2 = torch.t(torch.index_select(w2, 0, hidden_indices2)) * input_layer_output
    hidden_x3 = torch.t(torch.index_select(w2, 0, hidden_indices3)) * input_layer_output

    # sum hidden, w2, and b2 to find output layer
    sum_hidden_1 = torch.sum(hidden_x1, 1) + w2[0] * b2
    sum_hidden_2 = torch.sum(hidden_x2, 1) + w2[1] * b2
    sum_hidden_3 = torch.sum(hidden_x3, 1) + w2[2] * b2

    # transpose them
    summed_hidden_transposed1 = torch.t(sum_hidden_1)
    summed_hidden_transposed2 = torch.t(sum_hidden_2)
    summed_hidden_transposed3 = torch.t(sum_hidden_3)

    # concatenate outputs
    concatenated = torch.cat((summed_hidden_transposed1, summed_hidden_transposed2, summed_hidden_transposed3), dim=1)
    # softmax it
    soft_maxed = torch.softmax(concatenated, dim=1)
    return soft_maxed



# we load all training, validation, and test datasets for the classification task
train_dataset, train_label = pickle.load(open("data/part2_classification_train.data", "rb"))
validation_dataset, validation_label = pickle.load(open("data/part2_classification_validation.data", "rb"))
test_dataset, test_label = pickle.load(open("data/part2_classification_test.data", "rb"))

# when you inspect the training dataset, you are going to see that the class instances are sequential (e.g [1,1,1,1 ... 2,2,2,2,2 ... 3,3,3,3])
# we shuffle the training dataset by preserving instance-label relationship
indices = list(range(len(train_dataset)))
np.random.shuffle(indices)
train_dataset = np.array([train_dataset[i] for i in indices], dtype=np.float32)
train_label = np.array([train_label[i] for i in indices], dtype=np.float32)

# In order to be able to work with Pytorch, all datasets (and labels/ground truth) should be converted into a tensor
# since the datasets are already available as numpy arrays, we simply create tensors from them via torch.from_numpy()
train_dataset = torch.from_numpy(train_dataset)
train_label = torch.from_numpy(train_label)

validation_dataset = torch.from_numpy(validation_dataset)
validation_label = torch.from_numpy(validation_label)

test_dataset = torch.from_numpy(test_dataset)
test_label = torch.from_numpy(test_label)

# You are expected to create and initialize the parameters of the network
# Please do not forget to specify requires_grad=True for all parameters since they need to be trainable.

# w1 defines the parameters between the input layer and the hidden layer
w1 = w1 = torch.from_numpy(np.random.normal(0,1,64).astype(np.float32).reshape(64,1))
# Here you are expected to initialize w1 via the Normal distribution (mean=0, std=1).
w1.requires_grad = True
# b defines the bias parameters for the hidden layer
b1 = b1 = torch.from_numpy(np.random.normal(0,1,1).astype(np.float32).reshape(1,1))
# Here you are expected to initialize b1 via the Normal distribution (mean=0, std=1).
b1.requires_grad = True
# w2 defines the parameters between the hidden layer and the output layer
w2 = torch.from_numpy(np.random.normal(0,1,51).astype(np.float32).reshape(51,1))
# Here you are expected to initialize w2 via the Normal distribution (mean=0, std=1).
w2.requires_grad = True
# and finally, b2 defines the bias parameters for the output layer
b2 = torch.from_numpy(np.random.normal(0,1,1).astype(np.float32).reshape(1,1))
# Here you are expected to initialize b2 via the Normal distribution (mean=0, std=1).
b2.requires_grad = True

# you are expected to use the stochastic gradient descent optimizer
# w1, b1, w2 and b2 are the trainable parameters of the neural network
optimizer = torch.optim.SGD([w1, b1, w2, b2], lr=0.001)

# These arrays will store the loss values incurred at every training iteration
iteration_array = []
train_loss_array = []
validation_loss_array = []

# We are going to perform the backpropagation algorithm 'ITERATION' times over the training dataset
# After each pass, we are calculating the average/mean cross entropy loss over the validation dataset along with accuracy scores on both datasets.
ITERATION = 15000
for iteration in range(1, ITERATION + 1):
    iteration_array.append(iteration)

    # we need to zero all the stored gradient values calculated from the previous backpropagation step.
    optimizer.zero_grad()
    # Using the forward_pass function, we are performing a forward pass over the network with the training data
    train_predictions = forward_pass(w1, b1, w2, b2, train_dataset)
    # here you are expected to calculate the MEAN cross-entropy loss with respect to the network predictions and the training label
    train_mean_crossentropy_loss = torch.mean(-1 * torch.sum(train_label * torch.log(train_predictions), dim=1))

    train_loss_array.append(train_mean_crossentropy_loss.item())
    # We initiate the gradient calculation procedure to get gradient values with respect to the calculated loss
    train_mean_crossentropy_loss.backward()
    # After the gradient calculation, we update the neural network parameters with the calculated gradients.
    optimizer.step()

    # after each epoch on the training data we are calculating the loss and accuracy scores on the validation dataset
    # with torch.no_grad() disables gradient operations, since during testing the validation dataset, we don't need to perform any gradient operations
    with torch.no_grad():
        # Here you are expected to calculate the accuracy score on the training dataset by using the training labels.
        ground_truth = torch.argmax(train_label, dim=1)  # get index of correct label
        predicted = torch.argmax(train_predictions, dim=1)  # get index of predicted index
        mask = predicted == ground_truth  # mask for finding count
        equals = predicted[mask].size(dim=0)  # number of correctly predicted
        train_accuracy = (equals / train_label.size(dim=0)) * 100
        validation_predictions = forward_pass(w1, b1, w2, b2, validation_dataset)

        # Here you are expected to calculate the average/mean cross entropy loss for the validation datasets by using the validation dataset labels.
        validation_mean_crossentropy_loss = torch.mean(-1 * torch.sum((validation_label * torch.log(validation_predictions)), dim=1))

        validation_loss_array.append(validation_mean_crossentropy_loss.item())

        # Similarly, here, you are expected to calculate the accuracy score on the validation dataset
        validation_ground_truth = torch.argmax(validation_label, dim=1)  # get index of correct label index
        validation_predicted = torch.argmax(validation_predictions, dim=1)  # get index of predicted
        mask1 = validation_predicted == validation_ground_truth  # mask for finding count
        equals1 = validation_predicted[mask1].size(dim=0)  # number of correctly predicted
        validation_accuracy = (equals1 / validation_label.size(dim=0)) * 100

    print(
        "Iteration : %d - Train Loss %.4f - Train Accuracy : %.2f - Validation Loss : %.4f Validation Accuracy : %.2f" % (
        iteration + 1, train_mean_crossentropy_loss.item(), train_accuracy, validation_mean_crossentropy_loss.item(),
        validation_accuracy))

# after completing the training, we calculate our network's accuracy score on the test dataset...
# Again, here we don't need to perform any gradient-related operations, so we are using torch.no_grad() function.
with torch.no_grad():
    test_predictions = forward_pass(w1, b1, w2, b2, test_dataset)
    # Here you are expected to calculate the network accuracy score on the test dataset...
    test_ground_truth = torch.argmax(test_label, dim=1)  # get label of correct label index
    test_predicted = torch.argmax(validation_predictions, dim=1)  # get predicted index
    mask2 = test_predicted == test_ground_truth  # mask for count
    equals2 = test_predicted[mask2].size(dim=0)  # number of correctly predicted
    test_accuracy = (equals2 / test_ground_truth.size(dim=0)) * 100
    test_accuracy = torch.tensor([test_accuracy])
    print("Test accuracy : %.2f" % (test_accuracy.item()))

# We plot the loss versus iteration graph for both datasets (training and validation)
plt.plot(iteration_array, train_loss_array, label="Train Loss")
plt.plot(iteration_array, validation_loss_array, label="Validation Loss")
plt.legend()
plt.show()




