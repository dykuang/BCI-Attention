# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 10:51:19 2020

@author: dykua

functions for visualization
"""
import numpy as np
import matplotlib.pyplot as plt

def score_bar(datalist, colorlist, labellist, namelist,
              width=0.15, ylim = [0., 1.1], alpha = 0.5, figsize=(8,30),):
    '''
    datalist: a list of data to plot, each member is a numpy array
    colorlist: a list of color for each group member
    labellist: a list, name for each group
    namelist: a list, name for the legend
    '''
    # Setting the positions and width for the bars
    pos = list(range(len(labellist))) 
#    width = 0.1 
        
    # Plotting the bars
    fig, ax = plt.subplots(figsize=figsize)
    num_data = len(datalist)
    # Create a bar with pre_score data,
    # in position pos,
    for i in range(num_data):
        plt.bar([p+width*i for p in pos], datalist[i], 
                width=width, alpha=alpha, color=colorlist[i], label=namelist[i])
 
    
    # Set the y axis label
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Models')
    # Set the chart's title
    # ax.set_title('Summaries')
    
    # Set the position of the x ticks
    ax.set_xticks([p + 1.0 * width for p in pos])
    
    # Set the labels for the x ticks
    ax.set_xticklabels(labellist)
    
    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos)-width, max(pos)+2*width)
    plt.ylim(ylim)
    plt.xticks(rotation = 0)
    # Adding the legend and showing the plot
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()
       

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    cm: the confusion matrix
    classes: a list, name for each class
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'

    # Compute confusion matrix
#    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
#    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True',
           xlabel='Predicted')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
