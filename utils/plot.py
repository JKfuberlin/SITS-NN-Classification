from matplotlib import pyplot as plt 
from matplotlib.ticker import MaxNLocator
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
    x = [i for i in range(1, epoch+1)]
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
    plt.cla()