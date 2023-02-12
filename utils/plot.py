from matplotlib import pyplot as plt 
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix
from typing import List


def draw(y_train:List[float], y_val:List[float], name:str, model:str) -> None:
    """
    Visualise the change of loss or accuracy over epochs
    @params:
    y_train: average values from training for each epoch
    y_val: average values from validation for each epoch
    name: Loss or Accuracy
    model: LSTM or Tranformer
    """
    assert len(y_val) == len(y_train), "y_train and y_val must have the same length"
    epoch = len(y_train)
    x = [i for i in range(0, epoch)]
    # plot 2 curves
    plt.plot(x, y_train, color='b', label='train '+name)
    plt.plot(x, y_val, color='y', label="validation "+name)
    # set label
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    # set title and legend
    title = model + '_' + name
    plt.title(title)
    plt.legend()
    # save figure and clear
    plt.savefig('../outputs/'+ title +'.jpg')
    plt.clf()


def draw_confusion_matrix(y_true:List[int], y_pred:List[int], model:str) -> None:
    """Draw consufion matrix to visualise classification result"""
    assert len(y_pred) == len(y_true), "y_true and y_pred must have the same length"
    # calculate confusion matrix
    matrix = confusion_matrix(y_true, y_pred)
    # draw figure
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    thresh = matrix.max() / 2.
    for i in range(len(matrix)):    
        for j in range(len(matrix[i])):    
            plt.text(j, i, format(matrix[i][j]), 
                    ha="center", va="center", 
                    color="white" if matrix[i, j] >= thresh else "black")
    # set title and label
    indices = range(len(matrix))
    classes = ['Spruce', 'Beech', 'Pine', 'Douglas fir', 'Oak']
    title = f'{model} confusion matrix'
    plt.xticks(indices, classes, rotation=45)
    plt.yticks(indices, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # save figure and clear
    plt.savefig('../outputs/'+ title +'.jpg')
    plt.clf()