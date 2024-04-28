import os
import argparse

import numpy as np

from Model import Cifar
from DataReader import load_data, train_valid_split,train_valid_split_v2,train_valid_split_balanced,load_testing_images
from ImageUtils import visualize

def configure():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=28, help="depth")
    parser.add_argument("--mode",type=str,default="predict",help="train,test or predict")
    parser.add_argument("--widen_factor", type=int, default=10, help='width factor for Wide ResNet')
    parser.add_argument("--dropout_rate", type=float, default=0.3, help='dropout rate for Wide ResNet')
    parser.add_argument("--batch_size", type=int, default=128, help='training batch size')
    parser.add_argument("--num_classes", type=int, default=10, help='number of classes')
    parser.add_argument("--save_interval", type=int, default=10,
                        help='save the checkpoint when epoch MOD save_interval == 0')
    parser.add_argument("--weight_decay", type=float, default=2e-4, help='weight decay rate')
    parser.add_argument("--modeldir", type=str, default='../saved_ckpt', help='model directory')
    parser.add_argument("--initial_lr", type=float, default=0.1, help='initial learning rate')
    parser.add_argument("--lr_interval", type=int, default=25,
                        help='learning rate decreases by a factor after these many epochs')
    parser.add_argument("--lr_dec_factor", type=int, default=10,
                        help='learning rate decreases by this factor after a few epochs')
    return parser.parse_args()

def main(config):
    model = Cifar(config).cuda()

    # data_dir = "/scratch/user/rithvik/CSE636/working/hw2_server/cifar-10-batches-py"
    #cifar-10-data-dir
    data_dir = "../cifar-10-batches-py"
    save_dir = 'code2/images'
    # private_data_dir = '.../.../private_test_images_2024.npy'
    private_data_dir = ".../.../private_test_images_2024.npy"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if config.mode=="train":
        print("--- Preparing Data ---")

        x_train, y_train, x_test, y_test = load_data(data_dir)
        # 80:20 random split
        x_train_new, y_train_new, x_valid, y_valid = train_valid_split(x_train, y_train)


        # train_valid_split v2 returns whole dataset without any split
        # x_train_new, y_train_new, x_valid, y_valid = train_valid_split_v2(x_train, y_train)

        # This is for balanced class wise training
        # x_train_new, y_train_new, x_valid, y_valid = train_valid_split_balanced(x_train, y_train)


        print('x_train shape : ' ,x_train_new.shape)

        print('y_train shape : ' ,y_train_new.shape)

        print('x_test shape : ', x_test.shape)

        print('y_test shape : ', y_test.shape)

        # model = Cifar(config)

        print(model)
        print('\n')
        print('---training---\n')
        # First step: use the train_new set and the valid set to choose hyperparameters.
        model.train(x_train_new, y_train_new,x_valid,y_valid, max_epoch=200,save_dir='images/')


    elif config.mode=='validate':
        print('----CIFAR 10 VALIDATION-----\n')
        x_train, y_train, x_test, y_test = load_data(data_dir)
        #train_valid_split loads equal class distribution
        x_train_new, y_train_new, x_valid, y_valid = train_valid_split(x_train, y_train)
        #train_valid_split v2 returns whole dataset without any split
        # x_train_new, y_train_new, x_valid, y_valid = train_valid_split_v2(x_train, y_train)
        x_train_new, y_train_new, x_valid, y_valid = train_valid_split_balanced(x_train, y_train)
        model.test_or_validate(x_valid, y_valid, [200],
                               save_dir='images/')

    elif config.mode=='test':
        print('-----PUBLIC DATASET TESTING----')
        x_train, y_train, x_test, y_test = load_data(data_dir)
        print('x test shape ', x_test.shape)
        print('y test shape', y_test.shape)
        print('------Testing Dataset----\n')
        model.test_or_validate(x_test, y_test, [200],
                               save_dir='images/')

    elif config.mode=="predict":
        x_test = load_testing_images(private_data_dir)
        print('-----UNSEEN DATASET-----')
        print(x_test.shape)
        #visualize(x_test[0],'test.png')
        print('\n')
        preds = model.predict_prob(x_test,200)
        y = preds
        np.save('predictions',y)
        print('preds shape',y.shape)









if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = configure()
    main(config)
