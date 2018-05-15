"""Use this to complete part 2

    Usage:
        $ python mlp_relu.py

    Trains a three layer neural network using the relu function as the
        activation function.  Trains on four different learning rates
        (0.1, 0.01, 0.001, 0.0001) and plots the results.  Finally,
        performs a classifier test on the testing data set from CIFAR10
        and prints the results.

"""

import mlp
#import matplotlib.pyplot as plt

EPOCHS = 100

if __name__ == '__main__':
    train_loader = mlp.get_cifar10_data(train=True)
    validation_loader = mlp.get_cifar10_data(train=False)
    test_loader = mlp.get_cifar10_test_data()

    # training and validation
    train_loss1, accv1, model1 = mlp.relu_NN_train_and_val(train_loader, validation_loader, 
                                                   lr=0.1, epochs=EPOCHS)
    train_loss2, accv2, model2 = mlp.relu_NN_train_and_val(train_loader, validation_loader, 
                                                   lr=0.01, epochs=EPOCHS)
    train_loss3, accv3, model3 = mlp.relu_NN_train_and_val(train_loader, validation_loader, 
                                                   lr=0.001, epochs=EPOCHS)
    train_loss4, accv4, model4 = mlp.relu_NN_train_and_val(train_loader, validation_loader, 
                                                   lr=0.0001, epochs=EPOCHS)

    epochs = range(1, EPOCHS + 1)

    # Training loss plot
    #plt.figure(1)
    #plt.plot(epochs, train_loss1, '-b', label='lr=0.1')
    #plt.plot(epochs, train_loss2, '-r', label='lr=0.01')
    #plt.plot(epochs, train_loss3, '-g', label='lr=0.001')
    #plt.plot(epochs, train_loss4, '-p', label='lr=0.0001')
    #plt.legend(loc='lower right')
    #plt.xlabel('Number of epochs')
    #plt.ylabel('Average loss')
    #plt.title('Negative Log Loss on Training Data as a Function of Epochs')
    #plt.savefig("relu_training_loss.png")
    #print 'Plot saved as "relu_training_loss.png"'

    # Validation accuracy plot
    #plt.figure(2)
    #plt.plot(epochs, accv1, '-b', label='lr=0.1')
    #plt.plot(epochs, accv2, '-r', label='lr=0.01')
    #plt.plot(epochs, accv3, '-g', label='lr=0.001')
    #plt.plot(epochs, accv4, '-p', label='lr=0.0001')
    #plt.legend(loc='lower right')
    #plt.xlabel('Number of epochs')
    #plt.ylabel('Accuracy')
    #plt.title('Classifier Accuracy on Validation Data as a Function of Epochs')
    #plt.savefig("relu_accuracy.png")
    #print 'Plot saved as "relu_accuracy.png"'

    # Determine which model is best and then perform validation on test data
    modelv = [model1, model2, model3, model4]
    model_accuracy = [accv1[EPOCHS-1], accv2[EPOCHS-1], accv3[EPOCHS-1], accv4[EPOCHS-1]]
    
    model_index = model_accuracy.index(max(model_accuracy))
    best_model = modelv[model_index]
    print("\nBest model -- Learning rate of %f" % (10**(-1*(model_index + 1))))

    print("Results of validation on testing set:")
    lossv, accv = [], []
    mlp.validate(lossv, accv, best_model, test_loader)
    
