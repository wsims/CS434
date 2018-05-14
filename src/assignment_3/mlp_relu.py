"""Use this to complete part 2"""

import mlp
import matplotlib.pyplot as plt

EPOCHS = 25

if __name__ == '__main__':
    train_loader = mlp.get_cifar10_data(train=True)
    validation_loader = mlp.get_cifar10_data(train=False)
    train_loss1, accv1 = mlp.relu_NN_train_and_val(train_loader, validation_loader, 
                                                   lr=0.1, epochs=EPOCHS)
    train_loss2, accv2 = mlp.relu_NN_train_and_val(train_loader, validation_loader, 
                                                   lr=0.01, epochs=EPOCHS)
    train_loss3, accv3 = mlp.relu_NN_train_and_val(train_loader, validation_loader, 
                                                   lr=0.001, epochs=EPOCHS)
    train_loss4, accv4 = mlp.relu_NN_train_and_val(train_loader, validation_loader, 
                                                   lr=0.0001, epochs=EPOCHS)

    epochs = range(1, EPOCHS + 1)

    plt.figure(1)
    plt.plot(epochs, train_loss1, '-b', label='lr=0.1')
    plt.plot(epochs, train_loss2, '-r', label='lr=0.01')
    plt.plot(epochs, train_loss3, '-g', label='lr=0.001')
    plt.plot(epochs, train_loss4, '-p', label='lr=0.0001')
    plt.legend(loc='lower right')
    plt.xlabel('Number of epochs')
    plt.ylabel('Average nll')
    plt.title('Negative Log Likelihood Loss on Training Data as a Function of Epochs')
    plt.savefig("relu_training_loss.png")
    print 'Plot saved as "relu_training_loss.png"'

    plt.figure(2)
    plt.plot(epochs, accv1, '-b', label='lr=0.1')
    plt.plot(epochs, accv2, '-r', label='lr=0.01')
    plt.plot(epochs, accv3, '-g', label='lr=0.001')
    plt.plot(epochs, accv4, '-p', label='lr=0.0001')
    plt.legend(loc='lower right')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.title('Classifier Accuracy on Testing Data as a Function of Epochs')
    plt.savefig("relu_accuracy.png")
    print 'Plot saved as "relu_accuracy.png"'
    
