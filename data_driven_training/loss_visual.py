import matplotlib.pyplot as plt
import numpy as np


class LossEvaluation:
    loss_training = []  # Training loss values
    loss_test = []  # Test loss values
    epoch_train = []
    epoch_test = []
    max_epoch = None
    fig, ax = None, None  # Handle for the figure and axes

    @staticmethod
    def update_loss_train(loss_terms, epoch):
        LossEvaluation.loss_training.append(loss_terms)
        LossEvaluation.epoch_train.append(epoch)
        LossEvaluation.update_plot()

    @staticmethod
    def update_loss_test(loss_term, epoch):
        LossEvaluation.loss_test.append(loss_term)
        LossEvaluation.epoch_test.append(epoch)
        LossEvaluation.update_plot()

    @staticmethod
    def init_plot(max_epoch):
        plt.ion()  # Turn on interactive mode
        LossEvaluation.max_epoch = max_epoch
        LossEvaluation.fig, LossEvaluation.ax = plt.subplots(figsize=(10, 5))
        LossEvaluation.ax.set_title("Loss Evolution During Training")
        LossEvaluation.ax.set_xlabel("Epoch")
        LossEvaluation.ax.set_ylabel("Loss")
        LossEvaluation.ax.set_yscale('log')
        LossEvaluation.ax.grid(True)
        plt.show()

    @staticmethod
    def update_plot():
        LossEvaluation.ax.clear()
        LossEvaluation.ax.set_title("Loss Evolution During Training")
        LossEvaluation.ax.set_xlabel("Epoch")
        LossEvaluation.ax.set_ylabel("Loss")
        LossEvaluation.ax.set_yscale('log')
        # Dash the test loss line
        LossEvaluation.ax.plot(LossEvaluation.epoch_train, LossEvaluation.loss_training, label='Training Loss')
        LossEvaluation.ax.plot(LossEvaluation.epoch_test, LossEvaluation.loss_test, label='Test Loss', linestyle='--')
        LossEvaluation.ax.legend()
        LossEvaluation.ax.grid(True)
        plt.draw()
        plt.pause(0.01)  # Pause to allow the plot to update

