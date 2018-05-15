"""Use this to complete part 4"""

import mlp
import matplotlib.pyplot as plt

EPOCHS = 25

if __name__ == '__main__':
    train_loader = mlp.get_cifar10_data(train=True)
    validation_loader = mlp.get_cifar10_data(train=False)
    fl_train_loss1, fl_accv1 = mlp.four_NN_train_and_val(train_loader, validation_loader, epoch=EPOCHS)
    train_loss1, accv1 = mlp.sigmoid_NN_train_and_val(train_loader, validation_loader, epoch=EPOCHS)

    #train_loss2, accv2 = mlp.four_NN_train_and_val(train_loader, validation_loader, lr=0.01)
    #train_loss3, accv3 = mlp.four_NN_train_and_val(train_loader, validation_loader, lr=0.001)

    epochs = range(1, EPOCHS + 1) 

    plt.figure(1)
    plt.plot(epochs, fl_train_loss1, '-b', label='4-layer sigmoid')
    plt.plot(epochs, train_loss1, '-r', label='2-layer sigmoid')
    plt.legend(loc='lower right')
    plt.xlabel('Number of epochs')
    plt.ylabel('Average nll')
    plt.title('Negative Log Likelihood Loss on Training Data as a Function of Epochs')
    plt.savefig("fl_training_loss.png")
    print 'Plot saved as "fl_training_loss.png"'

    plt.figure(2)
    plt.plot(epochs, fl_accv1, '-b', label='4-layer sigmoid')
    plt.plot(epochs, accv1, '-r', label='2-layer sigmoid')
    plt.legend(loc='lower right')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.title('Classifier Accuracy on Testing Data as a Function of Epochs')
    plt.savefig("fl_accuracy.png")
    print 'Plot saved as "fl_accuracy.png"'
