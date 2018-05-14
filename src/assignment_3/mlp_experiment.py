"""Use this to complete part 3"""

import mlp

LEARNING_RATE = 0.1
WEIGHT_DECAY = 0
DROPOUT = 0.2
MOMENTUM = 0.5

if __name__ == '__main__':
    train_loader = mlp.get_cifar10_data(train=True)
    validation_loader = mlp.get_cifar10_data(train=False)
    train_loss1, accv1 = mlp.sigmoid_NN_train_and_val(train_loader, validation_loader, 
                                                      lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, 
                                                      dropout=DROPOUT, momentum=MOMENTUM)
