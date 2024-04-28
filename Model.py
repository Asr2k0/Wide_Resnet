import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from NetWork import Wide_ResNet
from ImageUtils import parse_record, save_loss_curve, save_actual_vs_predicted_images

class Cifar(nn.Module):
    def __init__(self, config):
        super(Cifar, self).__init__()
        self.config = config
        self.network = Wide_ResNet(
            depth=self.config.depth,
            widen_factor=self.config.widen_factor,
            dropout_rate=self.config.dropout_rate,
            num_classes=self.config.num_classes
        )
        # Define cross entropy loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.network.parameters(), momentum=0.9, lr=0.1,
                                         weight_decay=0.0001)

    def train(self, x_train, y_train, x_val, y_val, max_epoch, save_dir):

        self.network.train()
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.config.batch_size
        train_losses = []
        val_losses = []
        print('### Training... ###')
        for epoch in range(1, max_epoch + 1):
            start_time = time.time()
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]
            epoch_loss = 0
            if epoch in [75,150]:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] /=10

            for i in range(num_batches):
                start_idx = i * self.config.batch_size
                end_idx = min(start_idx + self.config.batch_size, num_samples)
                x_train_batch = np.empty((end_idx - start_idx, 3, 32, 32))
                for j in range(start_idx, end_idx):
                    x_train_batch[j - start_idx] = parse_record(curr_x_train[j], training=True)
                inputs = torch.from_numpy(x_train_batch).float().to('cuda')
                labels = torch.from_numpy(curr_y_train[start_idx:end_idx]).long().to('cuda')
                outputs = self.network(inputs)
                loss = self.criterion(outputs, labels)
                for param in self.network.parameters():
                    loss += self.config.weight_decay * torch.norm(param)
                epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)

            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, epoch_loss / num_batches, duration))
            train_losses.append(epoch_loss / num_batches)

            # Calculate validation loss
           # val_loss = self.calculate_validation_loss(x_val, y_val)
           # val_losses.append(val_loss)
           # print('Validation Loss: {:.6f}'.format(val_loss))

            if epoch % self.config.save_interval == 0:
                self.save(epoch)
        save_loss_curve(train_losses, os.path.join(save_dir, 'train_loss_curve_epoch.png'))
        save_loss_curve(val_losses, os.path.join(save_dir, 'val_loss_curve_epoch.png'))



    def test_or_validate(self, x, y, checkpoint_num_list,save_dir):
        self.network.eval()
        print('### Test or Validation ###')
        for checkpoint_num in checkpoint_num_list:
            print(f'-----Checkpoint Number -{checkpoint_num}-----')
            checkpointfile = os.path.join(self.config.modeldir, 'model-%d.ckpt' % (checkpoint_num))
            self.load(checkpointfile)

            preds = []
            actual_labels = []
            predicted_labels = []
            for i in tqdm(range(x.shape[0])):
                inputs = parse_record(x[i:i + 1], training=False)
                actual_labels.append(y[i])
                inputs = inputs.reshape(1, 3, 32, 32)
                inputs_tensor = torch.from_numpy(inputs).float().to('cuda')
                outputs = self.network(inputs_tensor)
                _, predicted = torch.max(outputs, 1)
                preds.append(predicted.item())
                predicted_labels.append(predicted.item())
            y = torch.tensor(y)

            preds = torch.tensor(preds)
            accuracy = (torch.sum(preds == y).item() / y.shape[0]) * 100

            print('Test accuracy: {:.4f}%'.format(accuracy))
            save_actual_vs_predicted_images(x[:5], actual_labels[:5], predicted_labels[:5],
                                            os.path.join('.', f'actual_vs_predicted_epoch_{checkpoint_num}.png'))

            print('--------------------------------------------')

    def predict_prob(self, x, checkpoint_num):
        self.network.eval()
        print('### PRIVATE TESTING SET ###')
        prob_array = np.empty((x.shape[0], 10))
        checkpointfile = os.path.join(self.config.modeldir, 'model-%d.ckpt' % (checkpoint_num))
        self.load(checkpointfile)
        p = []
        for i in tqdm(range(x.shape[0])):
            inputs = parse_record(x[i:i + 1], training=False)
            inputs = inputs.reshape(1, 3, 32, 32)
            inputs_tensor = torch.from_numpy(inputs).float().to('cuda')
            outputs = self.network(inputs_tensor)
            _, predicted = torch.max(outputs, 1)
            prob = nn.Softmax(dim=1)(outputs)
            # Move the tensor to CPU and then convert to NumPy array
            # p.append(predicted)
            prob_array[i] = prob[0].cpu().detach().numpy()

        return prob_array

    def save(self, epoch):
        checkpoint_path = os.path.join(self.config.modeldir, 'model-%d.ckpt' % (epoch))
        os.makedirs(self.config.modeldir, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")

    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))