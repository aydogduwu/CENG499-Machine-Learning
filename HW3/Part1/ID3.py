import numpy as np


# In the decision tree, non-leaf nodes are going to be represented via TreeNode
class TreeNode:
    def __init__(self, attribute):
        self.attribute = attribute
        # dictionary, k: subtree, key (k) an attribute value, value is either TreeNode or TreeLeafNode
        self.subtrees = {}


# In the decision tree, leaf nodes are going to be represented via TreeLeafNode
class TreeLeafNode:
    def __init__(self, data, label):
        self.data = data
        self.labels = label


class DecisionTree:
    def __init__(self, dataset: list, labels, features, criterion="information gain"):
        """
        :param dataset: array of data instances, each data instance is represented via a Python array
        :param labels: array of the labels of the data instances
        :param features: the array that stores the name of each feature dimension
        :param criterion: depending on which criterion ("information gain" or "gain ratio") the splits are to be performed
        """
        self.dataset = dataset
        self.labels = labels
        self.features = features
        self.criterion = criterion
        # it keeps the root node of the decision tree
        self.root = None

        # further variables and functions can be added...

    def calculate_entropy__(self, dataset, labels):
        """
        :param dataset: array of the data instances
        :param labels: array of the labels of the data instances
        :return: calculated entropy value for the given dataset
        """
        entropy_value = 0.0

        """
        Entropy calculations
        """
        # total number of data instances
        n = len(dataset)
        # dictionary, k: label, v: number of data instances with that label
        label_count = {}
        for label in labels:
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1

        for label in label_count:
            p = label_count[label] / n
            entropy_value -= p * np.log2(p)

        return entropy_value

    def calculate_average_entropy__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an average entropy value is calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute an average entropy value is going to be calculated...
        :return: the calculated average entropy value for the given attribute
        """
        average_entropy = 0.0
        """
            Average entropy calculations
        """
        n = len(dataset)
        # dictionary, k: attribute value, v: number of data instances with that attribute value
        attribute_value_count = {}
        # find index of the attribute in features
        attribute_index = self.features.index(attribute)
        for data_instance in dataset:
            attribute_value = data_instance[attribute_index]
            if attribute_value in attribute_value_count:
                attribute_value_count[attribute_value] += 1
            else:
                attribute_value_count[attribute_value] = 1

        for attribute_value in attribute_value_count:
            # find the data instances with the attribute value
            data_instances = []
            data_labels = []
            for i in range(len(dataset)):
                if dataset[i][attribute_index] == attribute_value:
                    data_instances.append(dataset[i])
                    data_labels.append(labels[i])
            # calculate entropy for the data instances with the attribute value
            entropy = self.calculate_entropy__(data_instances, data_labels)
            # calculate the average entropy
            average_entropy += (attribute_value_count[attribute_value] / n) * entropy

        return average_entropy

    def calculate_information_gain__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an information gain score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the information gain score is going to be calculated...
        :return: the calculated information gain score
        """
        information_gain = 0.0
        """
            Information gain calculations
        """
        # calculate entropy for the dataset
        entropy = self.calculate_entropy__(dataset, labels)
        # calculate average entropy for the dataset
        average_entropy = self.calculate_average_entropy__(dataset, labels, attribute)
        # calculate information gain
        information_gain = entropy - average_entropy

        return information_gain

    def calculate_intrinsic_information__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances on which an intrinsic information score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the intrinsic information score is going to be calculated...
        :return: the calculated intrinsic information score
        """
        intrinsic_info = 0.0
        """
            Intrinsic information calculations for a given attribute
        """

        n = len(dataset)
        # dictionary, k: attribute value, v: number of data instances with that attribute value
        attribute_value_count = {}
        # find index of the attribute in features
        attribute_index = self.features.index(attribute)
        for data_instance in dataset:
            attribute_value = data_instance[attribute_index]
            if attribute_value in attribute_value_count:
                attribute_value_count[attribute_value] += 1
            else:
                attribute_value_count[attribute_value] = 1

        for attribute_value in attribute_value_count:
            p = attribute_value_count[attribute_value] / n
            intrinsic_info -= p * np.log2(p)

        return intrinsic_info

    def calculate_gain_ratio__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances with which a gain ratio is going to be calculated
        :param labels: array of labels of those instances
        :param attribute: for which attribute the gain ratio score is going to be calculated...
        :return: the calculated gain ratio score
        """
        """
            Your implementation
        """
        gain_ratio = 0.0
        # calculate information gain
        information_gain = self.calculate_information_gain__(dataset, labels, attribute)
        # calculate intrinsic information
        intrinsic_info = self.calculate_intrinsic_information__(dataset, labels, attribute)
        # calculate gain ratio
        gain_ratio = information_gain / intrinsic_info

        return gain_ratio

    def ID3__(self, dataset, labels, used_attributes):
        """
        Recursive function for ID3 algorithm
        :param dataset: data instances falling under the current  tree node
        :param labels: labels of those instances
        :param used_attributes: while recursively constructing the tree, already used labels should be stored in used_attributes
        :return: it returns a created non-leaf node or a created leaf node
        """

        # depending on the criterion, calculate the information gain or gain ratio for each attribute
        information_gain = {}
        gain_ratio = {}
        if self.criterion == "information gain":
            information_gain = {}
            for attribute in self.features:
                if attribute not in used_attributes:
                    information_gain[attribute] = self.calculate_information_gain__(dataset, labels, attribute)
        elif self.criterion == "gain ratio":
            gain_ratio = {}
            for attribute in self.features:
                if attribute not in used_attributes:
                    gain_ratio[attribute] = self.calculate_gain_ratio__(dataset, labels, attribute)

        # find the attribute with the highest information gain or gain ratio
        best_attribute = None
        if self.criterion == "information gain":
            best_attribute = max(information_gain, key=information_gain.get)
        elif self.criterion == "gain ratio":
            best_attribute = max(gain_ratio, key=gain_ratio.get)

        # create a non-leaf node with the best attribute
        node = TreeNode(best_attribute)
        # find the index of the best attribute in features
        best_attribute_index = self.features.index(best_attribute)
        # find the attribute values of the best attribute
        best_attribute_values = []
        for data_instance in dataset:
            best_attribute_values.append(data_instance[best_attribute_index])
        best_attribute_values = list(set(best_attribute_values))
        # for each attribute value, create a subtree
        for attribute_value in best_attribute_values:
            # find the data instances with the attribute value
            data_instances = []
            data_labels = []
            for i in range(len(dataset)):
                if dataset[i][best_attribute_index] == attribute_value:
                    data_instances.append(dataset[i])
                    data_labels.append(labels[i])
            # if all the data instances have the same label, create a leaf node
            if len(set(data_labels)) == 1:
                # create a leaf node with the label
                leaf_node = TreeLeafNode(data_labels[0], attribute_value)
                # add the leaf node to the current node's subtree's dictionary
                node.subtrees[attribute_value] = leaf_node
            # if there are no more attributes to be used, create a leaf node
            elif len(used_attributes) == len(self.features):
                # find the most common label
                most_common_label = max(set(data_labels), key=data_labels.count)
                # create a leaf node with the most common label
                leaf_node = TreeLeafNode(most_common_label, attribute_value)
                # add the leaf node to the current node's subtree's dictionary
                node.subtrees[attribute_value] = leaf_node
            # if there are more attributes to be used, create a subtree
            else:
                # add the attribute to used_attributes
                used_attributes.append(best_attribute)
                # create a subtree
                subtree = self.ID3__(data_instances, data_labels, used_attributes)
                # add the subtree to the current node's subtree's dictionary
                node.subtrees[attribute_value] = subtree
                # remove the attribute from used_attributes
                used_attributes.remove(best_attribute)
        return node



    def predict(self, x):
        """
        :param x: a data instance, 1 dimensional Python array 
        :return: predicted label of x
        
        If a leaf node contains multiple labels in it, the majority label should be returned as the predicted label
        """
        predicted_label = None
        """
            Your implementation
        """
        # find the root node
        node = self.root
        leaf = False
        # while the node is not a leaf node
        while not isinstance(node, TreeLeafNode):
            # find the index of the attribute in features
            attribute_index = self.features.index(node.attribute)
            # find the attribute value of the data instance
            attribute_value = x[attribute_index]
            # find the subtree with the attribute value
            node = node.subtrees[attribute_value]
            if isinstance(node, TreeLeafNode):
                leaf = node
        # find the most common label in the leaf node
        predicted_label = leaf.data

        return predicted_label

    def train(self):
        self.root = self.ID3__(self.dataset, self.labels, [])
        print("Training completed")