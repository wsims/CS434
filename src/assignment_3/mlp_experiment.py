"""Use this to complete part 3"""

import mlp
import matplotlib.pyplot as plt

LEARNING_RATE = 0.1
WEIGHT_DECAY = 0
DROPOUT = 0.2
MOMENTUM = 0.5
EPOCHS = 25

if __name__ == '__main__':
    train_loader = mlp.get_cifar10_data(train=True)
    validation_loader = mlp.get_cifar10_data(train=False)
    # Drop-out experiments
    do_train_loss1, do_accv1 = mlp.sigmoid_NN_train_and_val(train_loader, validation_loader, 
                                                      lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, 
                                                      dropout=0, momentum=MOMENTUM, epochs=EPOCHS)
    do_train_loss2, do_accv2 = mlp.sigmoid_NN_train_and_val(train_loader, validation_loader, 
                                                      lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, 
                                                      dropout=0.2, momentum=MOMENTUM, epochs=EPOCHS)
    do_train_loss3, do_accv3 = mlp.sigmoid_NN_train_and_val(train_loader, validation_loader, 
                                                      lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, 
                                                      dropout=0.5, momentum=MOMENTUM, epochs=EPOCHS)
    do_train_loss4, do_accv4 = mlp.sigmoid_NN_train_and_val(train_loader, validation_loader, 
                                                      lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, 
                                                      dropout=0.7, momentum=MOMENTUM, epochs=EPOCHS)
    # Momentum experiments
    m_train_loss1, m_accv1 = mlp.sigmoid_NN_train_and_val(train_loader, validation_loader, 
                                                      lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, 
                                                      dropout=DROPOUT, momentum=0.25, epochs=EPOCHS)
    m_train_loss2, m_accv2 = mlp.sigmoid_NN_train_and_val(train_loader, validation_loader, 
                                                      lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, 
                                                      dropout=DROPOUT, momentum=0.5, epochs=EPOCHS)
    m_train_loss3, m_accv3 = mlp.sigmoid_NN_train_and_val(train_loader, validation_loader, 
                                                      lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, 
                                                      dropout=DROPOUT, momentum=0.75, epochs=EPOCHS)
    m_train_loss4, m_accv4 = mlp.sigmoid_NN_train_and_val(train_loader, validation_loader, 
                                                      lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, 
                                                      dropout=DROPOUT, momentum=0.99, epochs=EPOCHS)
    # Weight decay experiments
    wd_train_loss1, wd_accv1 = mlp.sigmoid_NN_train_and_val(train_loader, validation_loader, 
                                                      lr=LEARNING_RATE, weight_decay=0.1, 
                                                      dropout=DROPOUT, momentum=MOMENTUM, epochs=EPOCHS)
    wd_train_loss2, wd_accv2 = mlp.sigmoid_NN_train_and_val(train_loader, validation_loader, 
                                                      lr=LEARNING_RATE, weight_decay=0.01, 
                                                      dropout=DROPOUT, momentum=MOMENTUM, epochs=EPOCHS)
    wd_train_loss3, wd_accv3 = mlp.sigmoid_NN_train_and_val(train_loader, validation_loader, 
                                                      lr=LEARNING_RATE, weight_decay=0.001, 
                                                      dropout=DROPOUT, momentum=MOMENTUM, epochs=EPOCHS)
    wd_train_loss4, wd_accv4 = mlp.sigmoid_NN_train_and_val(train_loader, validation_loader, 
                                                      lr=LEARNING_RATE, weight_decay=0.0001, 
                                                      dropout=DROPOUT, momentum=MOMENTUM, epochs=EPOCHS)

    epochs = range(1, EPOCHS + 1)

    plt.figure(1)
    plt.plot(epochs, do_train_loss1, '-b', label='do=0')
    plt.plot(epochs, do_train_loss2, '-r', label='do=0.2')
    plt.plot(epochs, do_train_loss3, '-g', label='do=0.5')
    plt.plot(epochs, do_train_loss4, '-p', label='do=0.7')
    plt.legend(loc='lower right')
    plt.xlabel('Number of epochs')
    plt.ylabel('Average nll')
    plt.title('Negative Log Likelihood Loss on Training Data as a Function of Epochs')
    plt.savefig("do_training_loss.png")
    print 'Plot saved as "do_training_loss.png"'

    plt.figure(2)
    plt.plot(epochs, do_accv1, '-b', label='do=0')
    plt.plot(epochs, do_accv2, '-r', label='do=0.2')
    plt.plot(epochs, do_accv3, '-g', label='do=0.5')
    plt.plot(epochs, do_accv4, '-p', label='do=0.7')
    plt.legend(loc='lower right')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.title('Classifier Accuracy on Testing Data as a Function of Epochs')
    plt.savefig("do_accuracy.png")
    print 'Plot saved as "do_accuracy.png"'

    plt.figure(3)
    plt.plot(epochs, m_train_loss1, '-b', label='m=0.25')
    plt.plot(epochs, m_train_loss2, '-r', label='m=0.5')
    plt.plot(epochs, m_train_loss3, '-g', label='m=0.75')
    plt.plot(epochs, m_train_loss4, '-p', label='m=0.99')
    plt.legend(loc='lower right')
    plt.xlabel('Number of epochs')
    plt.ylabel('Average nll')
    plt.title('Negative Log Likelihood Loss on Training Data as a Function of Epochs')
    plt.savefig("m_training_loss.png")
    print 'Plot saved as "m_training_loss.png"'

    plt.figure(4)
    plt.plot(epochs, m_accv1, '-b', label='m=0.25')
    plt.plot(epochs, m_accv2, '-r', label='m=0.5')
    plt.plot(epochs, m_accv3, '-g', label='m=0.75')
    plt.plot(epochs, m_accv4, '-p', label='m=0.99')
    plt.legend(loc='lower right')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.title('Classifier Accuracy on Testing Data as a Function of Epochs')
    plt.savefig("m_accuracy.png")
    print 'Plot saved as "m_accuracy.png"'

    plt.figure(5)
    plt.plot(epochs, wd_train_loss1, '-b', label='wd=0.1')
    plt.plot(epochs, wd_train_loss2, '-r', label='wd=0.01')
    plt.plot(epochs, wd_train_loss3, '-g', label='wd=0.001')
    plt.plot(epochs, wd_train_loss4, '-p', label='wd=0.0001')
    plt.legend(loc='lower right')
    plt.xlabel('Number of epochs')
    plt.ylabel('Average nll')
    plt.title('Negative Log Likelihood Loss on Training Data as a Function of Epochs')
    plt.savefig("wd_training_loss.png")
    print 'Plot saved as "wd_training_loss.png"'

    plt.figure(6)
    plt.plot(epochs, wd_accv1, '-b', label='wd=0.1')
    plt.plot(epochs, wd_accv2, '-r', label='wd=0.01')
    plt.plot(epochs, wd_accv3, '-g', label='wd=0.001')
    plt.plot(epochs, wd_accv4, '-p', label='wd=0.0001')
    plt.legend(loc='lower right')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.title('Classifier Accuracy on Testing Data as a Function of Epochs')
    plt.savefig("wd_accuracy.png")
    print 'Plot saved as "wd_accuracy.png"'
