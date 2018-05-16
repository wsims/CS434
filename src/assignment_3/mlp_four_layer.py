"""Use this to complete part 4

    Usage:
        $ python mlp_four_layer.py

"""


import mlp
#import matplotlib.pyplot as plt

EPOCHS = 1

if __name__ == '__main__':
    train_loader = mlp.get_cifar10_data(train=True)
    validation_loader = mlp.get_cifar10_data(train=False)
    test_loader = mlp.get_cifar10_test_data()
    
    fl_train_loss1, fl_accv1, four_layer_model = mlp.four_NN_train_and_val(train_loader, validation_loader, epochs=EPOCHS)
    train_loss1, accv1, three_layer_model = mlp.sigmoid_NN_train_and_val(train_loader, validation_loader, epochs=EPOCHS)

    #train_loss2, accv2 = mlp.four_NN_train_and_val(train_loader, validation_loader, lr=0.01)
    #train_loss3, accv3 = mlp.four_NN_train_and_val(train_loader, validation_loader, lr=0.001)

    epochs = range(1, EPOCHS + 1) 

    #plt.figure(1)
    #plt.plot(epochs, fl_train_loss1, '-b', label='4-layer w/ sigmoid activation')
    #plt.plot(epochs, train_loss1, '-r', label='3-layer w/ sigmoid activation')
    #plt.legend(loc='lower right')
    #plt.xlabel('Number of epochs')
    #plt.ylabel('Average Negative Log Loss')
    #plt.title('Negative Log Likelihood Loss on Training Data as a Function of Epochs')
    #plt.savefig("fl_training_loss.png")
    #print 'Plot saved as "fl_training_loss.png"'

    #plt.figure(2)
    #plt.plot(epochs, fl_accv1, '-b', label='4-layer w/ sigmoid activation')
    #plt.plot(epochs, accv1, '-r', label='3-layer w/ sigmoid activaction')
    #plt.legend(loc='lower right')
    #plt.xlabel('Number of epochs')
    #plt.ylabel('Accuracy (Percentage)')
    #plt.title('Classifier Accuracy on Validation Data as a Function of Epochs')
    #plt.savefig("fl_accuracy.png")
    #print 'Plot saved as "fl_accuracy.png"'

    print("Results of validation on testing set with four-layer net:")
    lossv, accv = [], []
    mlp.validate(lossv, accv, four_layer_model, test_loader)

    print("Results of validation on testing set with three-layer net:")
    lossv, accv = [], []
    mlp.validate(lossv, accv, three_layer_model, test_loader)

    
