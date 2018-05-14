import mlp

if __name__ == '__main__':
    train_loader = mlp.get_cifar10_data(train=True)
    validation_loader = mlp.get_cifar10_data(train=False)
    train_loss1, accv1 = mlp.sigmoid_NN_train_and_val(train_loader, validation_loader, lr=0.1)
    #train_loss2, accv2 = mlp.sigmoid_NN_train_and_val(train_loader, validation_loader, lr=0.01)
    #train_loss3, accv3 = mlp.sigmoid_NN_train_and_val(train_loader, validation_loader, lr=0.001)
