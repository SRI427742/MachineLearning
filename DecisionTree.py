#!/usr/bin/env python
# coding: utf-8

# In[1]:


# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Homework for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
# Anjum Chida (anjum.chida@utdallas.edu)
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import os
import graphviz
import os
os.chdir('/Users/sriharsha/Desktop/Sri Harsha/Classes/Sem3/Machine Learning/Assignments/ML_Assignment2')


# In[2]:


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """
    subset={}
    x_frequency=np.unique(x)
    for i in x_frequency:
        l=[]
        for j in range(len(x)):
            if x[j]==i:
                l.append(j)
        subset[i]=l
    return subset
    raise Exception('Function not yet implemented!')


# In[3]:


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    z=0
    unique, counts = np.unique(y, return_counts=True)
    y_class={unique[i]: counts[i] for i in range(len(unique))}
    cnt=len(y)
    for i in y_class:
        z = z + (-1)*(y_class[i]/cnt)*(np.log2(y_class[i]/cnt))
    return z
    raise Exception('Function not yet implemented!')


# In[4]:


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    
    x_indexes= partition(x)
    cnt = len(x)
    s = 0
    for i in x_indexes:
        s = s + ( len( x_indexes[i] ) / cnt ) * ( entropy( y[x_indexes[i]] ) )
    IG=entropy(y) - s
    return IG
    raise Exception('Function not yet implemented!')


# In[6]:


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    tree={}
    y_unique,y_frequency=np.unique(y,return_counts=True)
    y_unique=list(y_unique)
    y_frequency=list(y_frequency)
    if len(y_unique) == 1:
        return y_unique[0]
    if len(y_unique) == 0:
        return 0
    if attribute_value_pairs==None:
        attribute_value_pairs=[]
        for i in range(len(x[0])):
            x_unique=np.unique(x[:,i])
            for j in range(len(x_unique)):
                attribute_value_pairs.append((i,x_unique[j]))
    if len(attribute_value_pairs) == 0 or depth == max_depth:
         return y_unique[np.argmax(y_frequency)]
    
    IG = {}

    for attribute, attribute_value in attribute_value_pairs:
        sub_attribute = x[:,attribute]
        for i in range(len(sub_attribute)):
            sub_attribute = (np.array(x)[:, attribute] == attribute_value).astype(int)
        IG[(attribute,attribute_value)] = mutual_information(sub_attribute,y)
    maxIG_Attribute=max(IG, key=IG.get) 
    x_subset = (np.array(x)[:, maxIG_Attribute[0]] == maxIG_Attribute[1]).astype(int)
    x_subset_split = partition(x_subset)
    updated_attribute_value_pairs = attribute_value_pairs.copy()
    updated_attribute_value_pairs.remove(maxIG_Attribute)
    
    for index in x_subset_split:
        x_id3 = x[x_subset_split[index]]
        y_id3 = y[x_subset_split[index]]
        b = bool(index)
        tree[(maxIG_Attribute[0],maxIG_Attribute[1],b)] = id3(x_id3, y_id3, attribute_value_pairs = updated_attribute_value_pairs, depth = depth + 1, max_depth = max_depth)
    return tree
    raise Exception('Function not yet implemented!')


# In[7]:


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """
    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    
    for nd in tree:
        nd_attribute = nd[0]
        nd_value = nd[1]
        nd_result = nd[2]
        if (x[nd_attribute] == nd_value) == nd_result:
            if type(tree[nd]) is dict:
                return predict_example(x, tree[nd])
            else:
                return tree[nd]
    
    raise Exception('Function not yet implemented!')


# In[8]:


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # INSERT YOUR CODE HERE
    total_errors = 0
    
    for i in range(len(y_pred)):
        if y_pred[i] != y_true[i]:
            total_errors = total_errors + 1
    return total_errors/len(y_true)
    raise Exception('Function not yet implemented!')


# In[9]:


def pretty_print(tree, depth=0):
    """
    Pretty prints the decision tree to the console. Use print(tree) to print the raw nested dictionary representation
    DO NOT MODIFY THIS FUNCTION!
    """
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


# In[10]:


def render_dot_file(dot_string, save_file, image_format='png'):
    """
    Uses GraphViz to render a dot file. The dot file can be generated using
        * sklearn.tree.export_graphviz()' for decision trees produced by scikit-learn
        * to_graphviz() (function is in this file) for decision trees produced by  your code.
    DO NOT MODIFY THIS FUNCTION!
    """
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


# In[11]:


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    """
    Converts a tree to DOT format for use with visualize/GraphViz
    DO NOT MODIFY THIS FUNCTION!
    """

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid


# In[13]:


if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, max_depth=3)

    # Pretty print it to console
    pretty_print(decision_tree)

    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree)
    render_dot_file(dot_str, './my_learned_tree')

    # Compute the test error 
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)

    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
    
    y_pred_trn = [predict_example(x, decision_tree) for x in Xtrn]
    trn_err = compute_error(ytrn, y_pred_trn )
    
    print('Train Error = {0:4.2f}%.'.format(trn_err * 100))


# In[26]:


#b
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.metrics import confusion_matrix
testError = []
trainError = []
trainDataset = ['./monks-1.train','./monks-2.train','./monks-3.train']
testDataset = ['./monks-1.test','./monks-2.test','./monks-3.test']

for dataset in range(0,3):
    print("Error Metrics for",trainDataset[dataset],"and",testDataset[dataset],"datasets :")
    trainData = np.genfromtxt(trainDataset[dataset], missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = trainData[:, 0]
    Xtrn = trainData[:, 1:]
    testData = np.genfromtxt(testDataset[dataset], missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = testData[:, 0]
    Xtst = testData[:, 1:]
   
    for depth in range(1,11):
        tree = id3(Xtrn, ytrn,max_depth=depth)
        yTrainPred=[]
        yTestPred=[]
        for j in Xtrn:
            yTrainPred.append(predict_example(j, tree))
        for j in Xtst:
            yTestPred.append(predict_example(j, tree))  
        trn_err = compute_error(ytrn, yTrainPred)
        tst_err = compute_error(ytst, yTestPred)
        testError.append(tst_err * 100)
        trainError.append(trn_err * 100)
        print("Test Data error for the depth",depth,":",testError[depth-1])
        print("Train Data error for the depth",depth,":",trainError[depth-1]) 
    avgTrainError=sum(trainError[dataset*10 : dataset*10 + 11])/10
    avgTestError=sum(testError[dataset*10 : dataset*10 + 11])/10
    print("Average Training error for the ",trainDataset[dataset],":",avgTrainError)
    print("Average Test error for the",testDataset[dataset],":",avgTestError)
    plt.figure() 
    plt.plot(testError[dataset*10 : dataset*10 + 11])
    plt.plot(trainError[dataset*10 : dataset*10 + 11])
    plt.ylabel('Test and Train error')
    plt.legend(['Test Error', 'Train Error'])
    plt.title('Learning Curves for the datasets:'+trainDataset[dataset]+' and '+testDataset[dataset] )
    plt.show()


# In[22]:


#c
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
trainDataset = './monks-1.train'
testDataset = './monks-1.test'
print("Confusion matrix for ",trainDataset,"data:")
trainData = np.genfromtxt(trainDataset, missing_values=0, skip_header=0, delimiter=',', dtype=int)
ytrn = trainData[:, 0]
Xtrn = trainData[:, 1:]
testData = np.genfromtxt(testDataset, missing_values=0, skip_header=0, delimiter=',', dtype=int)
ytst = testData[:, 0]
Xtst = testData[:, 1:]
for depth in [1,3,5]:
        tree = id3(Xtrn, ytrn,max_depth=depth)
        yTestPred=[]
        for j in Xtst:
            yTestPred.append(predict_example(j, tree))  
        tst_err = compute_error(ytst, yTestPred)
        testError.append(tst_err * 100)
        print("Confusion Matrix for depth",depth,":") 
        print(confusion_matrix(ytst, yTestPred)) 


# In[23]:


#d
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier  

depth = [1,3,5]
trainData = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
ytrn = trainData[:, 0]
Xtrn = trainData[:, 1:]

testData = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
ytst = testData[:, 0]
Xtst = testData[:, 1:]

for i in range(0,3):
    decision_tree_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 0, max_depth = depth[i]) 
    decision_tree_entropy.fit(Xtrn, ytrn)
    y_pred = decision_tree_entropy.predict(Xtst)
    print("Confusion Matrix for depth", depth[i])
    print(confusion_matrix(ytst, y_pred))


# In[30]:


# e
from sklearn.model_selection import train_test_split
test_fraction = 0.3 
N = np.genfromtxt('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', missing_values=0, skip_header=0, delimiter=',', dtype=int)
y = N[:, 10]
X = N[:, 1:10]
Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size = test_fraction, random_state=42)

print("Decision tree implementation with our alogoithm") 
depth=[1,3,5]
for i in depth:
    decision_tree = id3(Xtrn, ytrn,max_depth=i)
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)           
    print("Confusion Matrix for depth:",i,":") 
    print(confusion_matrix(ytst, y_pred))    
            
print("Decision tree implementation with sckit learn")
for i in depth:
    decision_tree_e = DecisionTreeClassifier(criterion = "entropy", random_state = 0, max_depth = i, min_samples_leaf = 5) 

    
    decision_tree_e.fit(Xtrn, ytrn)
    y_pred = decision_tree_e.predict(Xtst)
    print("Confusion Matrix for depth:",i,":")
    print(confusion_matrix(ytst, y_pred))


# In[ ]:




