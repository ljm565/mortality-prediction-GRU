import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import pickle
import numpy as np
import time

from config import Config
from utils_func import *
from utils_data import DLoader
from model import MortalityPredGRU



class Trainer:
    def __init__(self, config:Config, device:torch.device, mode:str, continuous:int):
        self.config = config
        self.device = device
        self.mode = mode
        self.continuous = continuous
        self.dataloaders = {}

        # if continuous, load previous training info
        if self.continuous:
            with open(self.config.loss_data_path, 'rb') as f:
                self.loss_data = pickle.load(f)

        # path, data params
        self.base_path = self.config.base_path
        self.model_path = self.config.model_path

        # train params
        self.batch_size = self.config.batch_size
        self.epochs = self.config.epochs
        self.lr = self.config.lr

        # data condition
        self.min_los, self.max_los = self.config.min_los, self.config.max_los
        self.max_chartevent_time = self.config.max_chartevent_time
        self.max_seq = self.config.max_seq

        if not os.path.isfile(self.base_path + '/data/ICU_patients.pkl'):
            # preprocessing ICU stay and admission data
            ICU_patients = dict4icuPatient(self.base_path + '/data/ICUSTAYS.csv', self.min_los, self.max_los)
            admin_patients = dict4admissions(self.base_path + '/data/ADMISSIONS.csv')

            # labeling
            ICU_patients = labeling(ICU_patients, admin_patients)

            # make data features
            ICU_patients = make_x_data(self.base_path + '/data/CHARTEVENTS.csv', ICU_patients, self.max_chartevent_time, self.max_seq)

            # compensate patients features who has no data
            ICU_patients = compensate_x_data(ICU_patients)

            # save the intermediate data
            with open(self.base_path + '/data/ICU_patients.pkl', 'wb') as f:
                pickle.dump(ICU_patients, f)
        
        else:
            with open(self.base_path + '/data/ICU_patients.pkl', 'rb') as f:
                ICU_patients = pickle.load(f)
        
        train_icu, test_icu = divide_dataset(ICU_patients)
        dict4baseInfo, dict4itemid = make_feature_dict(train_icu, self.config.diag_topk, self.config.itemid_topk, 'ETHNICITY', 'ADMISSION_TYPE', 'DIAGNOSIS', 'ITEMID')
        train_icu, test_icu = make_feature(train_icu, dict4baseInfo, dict4itemid), make_feature(test_icu, dict4baseInfo, dict4itemid)

        if self.mode == 'train':
            self.trainset = DLoader(train_icu, self.max_seq)
            self.testset = DLoader(test_icu, self.max_seq)
            self.dataloaders['train'] = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=4)
            self.dataloaders['test'] = DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        else:
            self.testset = DLoader(test_icu, self.max_seq)
            self.dataloaders['test'] = DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        
        # model, optimizer, loss
        self.model = MortalityPredGRU(self.config, len(dict4baseInfo), len(dict4itemid), self.max_seq, self.device).to(self.device)
        self.criterion = nn.BCELoss()
        if self.mode == 'train':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            if self.continuous:
                self.check_point = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(self.check_point['model'])
                self.optimizer.load_state_dict(self.check_point['optimizer'])
                del self.check_point
                torch.cuda.empty_cache()
        elif self.mode == 'test':
            self.check_point = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(self.check_point['model'])
            self.model.eval()
            del self.check_point
            torch.cuda.empty_cache()

        
    def train(self):
        early_stop = 0
        best_acc = 0 if not self.continuous else self.loss_data['best_acc']
        train_loss_history = [] if not self.continuous else self.loss_data['train_loss_history']
        val_loss_history = [] if not self.continuous else self.loss_data['val_loss_history']
        best_epoch_info = 0 if not self.continuous else self.loss_data['best_epoch']

        for epoch in range(self.epochs):
            start = time.time()
            print(epoch+1, '/', self.epochs)
            print('-'*10)
            for phase in ['train', 'test']:
                print('Phase: {}'.format(phase))
                if phase == 'train':
                    self.model.train()                
                else:
                    self.model.eval()                    

                total_loss, total_acc, total1, pred1 =  0, 0, 0, 0
                for i, (baseInfo, charteventInfo, y) in enumerate(self.dataloaders[phase]):
                    batch_size = baseInfo.size(0)
                    baseInfo, charteventInfo, y = baseInfo.to(self.device), charteventInfo.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase=='train'):
                        output = self.model(baseInfo, charteventInfo)
                        loss = self.criterion(output, y)
                        predict = (output > 0.5).float()
                        acc = (predict==y).float().sum()/batch_size

                        for pred, real in zip(predict, y):
                            if real == 1:
                                total1 += 1
                                if pred == 1:
                                    pred1 += 1
                        
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                        
                    total_loss += loss.item()*batch_size
                    total_acc += acc * batch_size
                    if i % 100 == 0:
                        print('Epoch {}: {}/{} step loss: {}, acc: {}'.format(epoch+1, i, len(self.dataloaders[phase]), loss.item(), acc))
                epoch_loss = total_loss/len(self.dataloaders[phase].dataset)
                epoch_acc = total_acc/len(self.dataloaders[phase].dataset)
                one_acc = pred1/total1
                print('{} loss: {:4f}, acc: {:4f}, one acc: {:4f}\n'.format(phase, epoch_loss, epoch_acc, one_acc))

                if phase == 'train':
                    train_loss_history.append(epoch_loss)
                if phase == 'test':
                    val_loss_history.append(epoch_loss)

                    # save best model
                    early_stop += 1
                    if one_acc + epoch_acc > best_acc:
                        early_stop = 0
                        best_acc = one_acc + epoch_acc
                        best_epoch = best_epoch_info + epoch + 1
                        save_checkpoint(self.model_path, self.model, self.optimizer)

            print("time: {} s\n".format(time.time() - start))
            print('\n'*2)

            # early stopping
            if early_stop == self.config.early_stop_criterion:
                break

        print('best val bleu: {:4f}, best epoch: {:d}\n'.format(best_acc, best_epoch))
        self.loss_data = {'best_epoch': best_epoch, 'best_acc': best_acc, 'train_loss_history': train_loss_history, 'val_loss_history': val_loss_history}
        return self.loss_data
    

    def test(self):        
        # statistics of the test set
        phase = 'test'
        with torch.no_grad():
            self.model.eval()
            total_loss, acc = 0, 0
            for idx, (baseInfo, charteventInfo, y) in enumerate(self.dataloaders[phase]):
                batch_size = baseInfo.size(0)
                baseInfo, charteventInfo, y = baseInfo.to(self.device), charteventInfo.to(self.device), y.to(self.device)
                output = self.model(baseInfo, charteventInfo)
                loss = self.criterion(output, y)

                total_loss += loss.item()*batch_size
                predict = (output > 0.5).float()
                acc += (predict==y).float().sum().item()

                if idx == 0:
                    pred_np_train = predict.detach().cpu().numpy()
                    y_np = y.detach().cpu().numpy()
                    output_np = output.detach().cpu().numpy()
                else:
                    pred_new = predict.detach().cpu().numpy()
                    pred_np_train = np.concatenate((pred_np_train, pred_new), axis=0)
                    y_new = y.detach().cpu().numpy()
                    y_np = np.concatenate((y_np, y_new), axis=0)
                    output_new = output.detach().cpu().numpy()
                    output_np = np.concatenate((output_np, output_new), axis=0)

            total_loss = total_loss / len(self.dataloaders[phase].dataset)
            epoch_acc = acc/len(self.dataloaders[phase].dataset)
            print('test loss: {:4f}, test acc: {}'.format(total_loss, epoch_acc))

            print('AUROC of test data: ', roc_auc_score(y_np, output_np))
            print('AUPRC of test data: ', average_precision_score(y_np, output_np))


           


            