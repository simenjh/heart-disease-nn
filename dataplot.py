import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing



def plot_learning_curves(costs_train, costs_test, m_examples):
    plt.style.use("seaborn")
    plt.plot(m_examples, costs_train, label='Training cost')
    plt.plot(m_examples, costs_test, label='Cross-validation cost')
    plt.ylabel('Cost', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title('Learning curves', fontsize=18, y=1.03)
    plt.legend()
    plt.ylim(0, 1)
    plt.show()
