"""Use this to complete part 3"""

import mlp
#import matplotlib.pyplot as plt

LEARNING_RATE = 0.1
WEIGHT_DECAY = 0
DROPOUT = 0.2
MOMENTUM = 0.5
EPOCHS = 100 

if __name__ == '__main__':
    train_loader = mlp.get_cifar10_data(train=True)
    validation_loader = mlp.get_cifar10_data(train=False)
    test_loader = mlp.get_cifar10_test_data()
    # Drop-out experiments
    do_train_loss1, do_accv1, do_model1 = mlp.sigmoid_NN_train_and_val(train_loader, validation_loader, 
                                                      lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, 
                                                      dropout=0, momentum=MOMENTUM, epochs=EPOCHS)
    do_train_loss2, do_accv2, do_model2 = mlp.sigmoid_NN_train_and_val(train_loader, validation_loader, 
                                                      lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, 
                                                      dropout=0.2, momentum=MOMENTUM, epochs=EPOCHS)
    do_train_loss3, do_accv3, do_model3 = mlp.sigmoid_NN_train_and_val(train_loader, validation_loader, 
                                                      lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, 
                                                      dropout=0.5, momentum=MOMENTUM, epochs=EPOCHS)
    do_train_loss4, do_accv4, do_model4 = mlp.sigmoid_NN_train_and_val(train_loader, validation_loader, 
                                                      lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, 
                                                      dropout=0.7, momentum=MOMENTUM, epochs=EPOCHS)
    # Momentum experiments
    m_train_loss1, m_accv1, m_model1 = mlp.sigmoid_NN_train_and_val(train_loader, validation_loader, 
                                                      lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, 
                                                      dropout=DROPOUT, momentum=0.25, epochs=EPOCHS)
    m_train_loss2, m_accv2, m_model2 = mlp.sigmoid_NN_train_and_val(train_loader, validation_loader, 
                                                      lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, 
                                                      dropout=DROPOUT, momentum=0.5, epochs=EPOCHS)
    m_train_loss3, m_accv3, m_model3 = mlp.sigmoid_NN_train_and_val(train_loader, validation_loader, 
                                                      lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, 
                                                      dropout=DROPOUT, momentum=0.75, epochs=EPOCHS)
    m_train_loss4, m_accv4, m_model4 = mlp.sigmoid_NN_train_and_val(train_loader, validation_loader, 
                                                      lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, 
                                                      dropout=DROPOUT, momentum=0.90, epochs=EPOCHS)
    # Weight decay experiments
    wd_train_loss1, wd_accv1, wd_model1 = mlp.sigmoid_NN_train_and_val(train_loader, validation_loader, 
                                                      lr=LEARNING_RATE, weight_decay=0.1, 
                                                      dropout=DROPOUT, momentum=MOMENTUM, epochs=EPOCHS)
    wd_train_loss2, wd_accv2, wd_model2 = mlp.sigmoid_NN_train_and_val(train_loader, validation_loader, 
                                                      lr=LEARNING_RATE, weight_decay=0.01, 
                                                      dropout=DROPOUT, momentum=MOMENTUM, epochs=EPOCHS)
    wd_train_loss3, wd_accv3, wd_model3 = mlp.sigmoid_NN_train_and_val(train_loader, validation_loader, 
                                                      lr=LEARNING_RATE, weight_decay=0.001, 
                                                      dropout=DROPOUT, momentum=MOMENTUM, epochs=EPOCHS)
    wd_train_loss4, wd_accv4, wd_model4 = mlp.sigmoid_NN_train_and_val(train_loader, validation_loader, 
                                                      lr=LEARNING_RATE, weight_decay=0.0001, 
                                                      dropout=DROPOUT, momentum=MOMENTUM, epochs=EPOCHS)

    epochs = range(1, EPOCHS + 1)

    #plt.figure(1)
    #plt.plot(epochs, do_train_loss1, '-b', label='do=0')
    #plt.plot(epochs, do_train_loss2, '-r', label='do=0.2')
    #plt.plot(epochs, do_train_loss3, '-g', label='do=0.5')
    #plt.plot(epochs, do_train_loss4, '-p', label='do=0.7')
    #plt.legend(loc='lower right')
    #plt.xlabel('Number of epochs')
    #plt.ylabel('Average Negative Log Loss')
    #plt.title('Negative Log Likelihood Loss on Training Data as a Function of Epochs')
    #plt.savefig("do_training_loss.png")
    #print 'Plot saved as "do_training_loss.png"'

    #plt.figure(2)
    #plt.plot(epochs, do_accv1, '-b', label='do=0')
    #plt.plot(epochs, do_accv2, '-r', label='do=0.2')
    #plt.plot(epochs, do_accv3, '-g', label='do=0.5')
    #plt.plot(epochs, do_accv4, '-p', label='do=0.7')
    #plt.legend(loc='lower right')
    #plt.xlabel('Number of epochs')
    #plt.ylabel('Accuracy (Percentage)')
    #plt.title('Classifier Accuracy on Validation Data as a Function of Epochs')
    #plt.savefig("do_accuracy.png")
    #print 'Plot saved as "do_accuracy.png"'

    #plt.figure(3)
    #plt.plot(epochs, m_train_loss1, '-b', label='m=0.25')
    #plt.plot(epochs, m_train_loss2, '-r', label='m=0.5')
    #plt.plot(epochs, m_train_loss3, '-g', label='m=0.75')
    #plt.plot(epochs, m_train_loss4, '-p', label='m=0.90')
    #plt.legend(loc='lower right')
    #plt.xlabel('Number of epochs')
    #plt.ylabel('Average Negative Log Loss')
    #plt.title('Negative Log Likelihood Loss on Training Data as a Function of Epochs')
    #plt.savefig("m_training_loss.png")
    #print 'Plot saved as "m_training_loss.png"'

    #plt.figure(4)
    #plt.plot(epochs, m_accv1, '-b', label='m=0.25')
    #plt.plot(epochs, m_accv2, '-r', label='m=0.5')
    #plt.plot(epochs, m_accv3, '-g', label='m=0.75')
    #plt.plot(epochs, m_accv4, '-p', label='m=0.90')
    #plt.legend(loc='lower right')
    #plt.xlabel('Number of epochs')
    #plt.ylabel('Accuracy (Percentage)')
    #plt.title('Classifier Accuracy on Validation Data as a Function of Epochs')
    #plt.savefig("m_accuracy.png")
    #print 'Plot saved as "m_accuracy.png"'

    #plt.figure(5)
    #plt.plot(epochs, wd_train_loss1, '-b', label='wd=0.1')
    #plt.plot(epochs, wd_train_loss2, '-r', label='wd=0.01')
    #plt.plot(epochs, wd_train_loss3, '-g', label='wd=0.001')
    #plt.plot(epochs, wd_train_loss4, '-p', label='wd=0.0001')
    #plt.legend(loc='lower right')
    #plt.xlabel('Number of epochs')
    #plt.ylabel('Average Negative Log Loss')
    #plt.title('Negative Log Likelihood Loss on Training Data as a Function of Epochs')
    #plt.savefig("wd_training_loss.png")
    #print 'Plot saved as "wd_training_loss.png"'

    #plt.figure(6)
    #plt.plot(epochs, wd_accv1, '-b', label='wd=0.1')
    #plt.plot(epochs, wd_accv2, '-r', label='wd=0.01')
    #plt.plot(epochs, wd_accv3, '-g', label='wd=0.001')
    #plt.plot(epochs, wd_accv4, '-p', label='wd=0.0001')
    #plt.legend(loc='lower right')
    #plt.xlabel('Number of epochs')
    #plt.ylabel('Accuracy (Percentage)')
    #plt.title('Classifier Accuracy on Validation Data as a Function of Epochs')
    #plt.savefig("wd_accuracy.png")
    #print 'Plot saved as "wd_accuracy.png"'

    # Determine which model is best and then perform validation on test data
    do_list = ['0','0.2','0.5','0.7'] 
    modelv = [do_model1, do_model2, do_model3, do_model4]
    model_accuracy = [do_accv1[EPOCHS-1], do_accv2[EPOCHS-1], do_accv3[EPOCHS-1], do_accv4[EPOCHS-1]]
    
    model_index = model_accuracy.index(max(model_accuracy))
    best_model = modelv[model_index]
    print("\nBest drop out value of %s") % do_list[model_index]

    print("Results of validation on testing set:")
    lossv, accv = [], []
    mlp.validate(lossv, accv, best_model, test_loader)
    
    # Determine which model is best and then perform validation on test data
    m_list = ['0.25','0.5','0.75','0.9'] 
    modelv = [m_model1, m_model2, m_model3, m_model4]
    model_accuracy = [m_accv1[EPOCHS-1], m_accv2[EPOCHS-1], m_accv3[EPOCHS-1], m_accv4[EPOCHS-1]]
    
    model_index = model_accuracy.index(max(model_accuracy))
    best_model = modelv[model_index]
    print("\nBest momentum model value of %s") % m_list[model_index] 

    print("Results of validation on testing set:")
    lossv, accv = [], []
    mlp.validate(lossv, accv, best_model, test_loader)
    
    # Determine which model is best and then perform validation on test data
    wd_list = ['0.1','0.01','0.001','0.0001'] 
    modelv = [wd_model1, wd_model2, wd_model3, wd_model4]
    model_accuracy = [wd_accv1[EPOCHS-1], wd_accv2[EPOCHS-1], wd_accv3[EPOCHS-1], wd_accv4[EPOCHS-1]]
    
    model_index = model_accuracy.index(max(model_accuracy))
    best_model = modelv[model_index]
    print("\nBest weight decay value of %s") % wd_list[model_index]

    print("Results of validation on testing set:")
    lossv, accv = [], []
    mlp.validate(lossv, accv, best_model, test_loader)
