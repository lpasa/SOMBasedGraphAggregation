import torch
import os
import datetime
import time
from numpy.linalg import svd

predict_fn = lambda output: output.max(1, keepdim=True)[1].detach().cpu()

_SOM_LAYERS=3

def prepare_log_files(test_name, log_dir):
    '''
    create a log file where test information and results will be saved
    :param test_name: name of the test
    :param log_dir: directory where the log files will be created
    :return: return a log file for each sub set (training, test, validation)
    '''
    train_log = open(os.path.join(log_dir, (test_name + "_train")), 'w+')
    test_log = open(os.path.join(log_dir, (test_name + "_test")), 'w+')
    valid_log = open(os.path.join(log_dir, (test_name + "_valid")), 'w+')

    for f in (train_log, test_log, valid_log):
        f.write("test_name: %s \n" % test_name)
        f.write(str(datetime.datetime.now()) + '\n')
        f.write("#epoch \t split \t loss \t acc \t avg_epoch_time \t avg_epoch_cost \n")

    return train_log, test_log, valid_log


class modelImplementation_GraphBinClassifier(torch.nn.Module):
    '''
        general implementation of training routine for a GNN that perform graph classification
    '''
    def __init__(self, model, criterion, device='cpu'):
        super(modelImplementation_GraphBinClassifier, self).__init__()
        self.model = model

        self.criterion = criterion
        self.device = device

    def stop_grad(self, phase):
        for name, param in self.model.named_parameters():

            if phase=="conv":
                if "som" in name or "out" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad =True
            elif phase == "readout":
                if "out" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad =True
            elif phase == "fine_tuning":
                if "som" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad =True

    def set_optimizer(self,lr_conv, lr_som, lr_reaout, lr_fine_tuning,weight_decay=0):
        '''
        set the optimizer for the training phase
        :param weight_decay: amount of weight decay to apply during training
        '''
        #------------------#
        train_out_stage_params=[]
        train_conv_stage_params=[]
        fine_tune_stage_par=[]
        for name, param in self.model.named_parameters():
            if not "som" in name:
                
                if "out" in name:
                    train_out_stage_params.append(param)

                else:
                    train_conv_stage_params.append(param)

                fine_tune_stage_par.append(param)


        self.conv_optimizer = torch.optim.AdamW(train_conv_stage_params, lr=lr_conv,weight_decay=weight_decay)
        self.out_optimizer = torch.optim.AdamW(train_out_stage_params, lr=lr_reaout,weight_decay=weight_decay)
        self.fine_tune_optimizer = torch.optim.AdamW(fine_tune_stage_par, lr=lr_fine_tuning, weight_decay=weight_decay)
        self.lr_som=lr_som


    def train_test_model(self, split_id, loader_train, loader_test, loader_valid, n_epochs_conv, n_epochs_readout,
                         n_epochs_fine_tuning, n_epochs_som, test_epoch, early_stopping_threshold,
                         early_stopping_threshold_som, max_n_epochs_without_improvements, test_name="", log_path="."):
        '''
        method that perform training of a given model, and test it after a given number of epochs
        :param split_id: numeric id of the considered split (use to identify the current split in a cross-validation setting)
        :param loader_train: loader of the training set
        :param loader_test: loader of the test set
        :param loader_valid: load of the validation set
        :param n_epochs: number of training epochs
        :param test_epoch: the test phase is performed every test_epoch epochs
        :param test_name: name of the test
        :param log_path: past where the logs file will be saved
        '''

        print("train conv_part")
        self.stop_grad("conv")
        self.training_phase(n_epochs=n_epochs_conv,
                            optimizer=self.conv_optimizer,
                            loader_train=loader_train,
                            loader_test=loader_test,
                            loader_valid=loader_valid,
                            test_epoch=test_epoch,
                            log_file_name="_conv_part_"+test_name,
                            split_id=split_id,
                            log_path=log_path,
                            use_conv_out=True,
                            test_name=test_name,
                            early_stopping_threshold=early_stopping_threshold,
                            max_n_epochs_without_improvements=max_n_epochs_without_improvements)

        print("train som")
        
        #load best model from previuos step
        self.load_model(test_name)

        train_log, test_log, valid_log = prepare_log_files("_som_part_"+test_name + "--split-" + str(split_id), log_path)

        train_loss, n_samples = 0.0, 0

        epoch_time_sum = 0

        best_epoch_som=0
        best_som_loss_so_far=-1
        som_n_epochs_without_improvements=0

        for epoch in range(n_epochs_som):
            self.model.train()

            epoch_start_time = time.time()
            for batch in loader_train:

                data = batch.to(self.device)
                _, h_conv, out = self.model(data)

                h_conv_1 = h_conv[:,0:self.model.out_channels]
                h_conv_2 = h_conv[:,self.model.out_channels:self.model.out_channels+self.model.out_channels*2]
                h_conv_3 = h_conv[:,self.model.out_channels*3: self.model.out_channels*3+self.model.out_channels*3]


                som_1_loss=self.model.som1.self_organizing(h_conv_1, epoch, n_epochs_som, self.lr_som)
                som_2_loss=self.model.som2.self_organizing(h_conv_2, epoch, n_epochs_som, self.lr_som)
                som_3_loss=self.model.som3.self_organizing(h_conv_3, epoch, n_epochs_som, self.lr_som)

                train_loss += som_1_loss + som_2_loss + som_3_loss
                n_samples += len(batch)

            epoch_time = time.time() - epoch_start_time
            epoch_time_sum += epoch_time

            if epoch % test_epoch == 0:
                print("split : ", split_id, " -- epoch : ", epoch, " -- loss: ", train_loss / n_samples)


                train_log.write(
                    "{:d}\t{:d}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        epoch,
                        split_id,
                        0,#loss_train_set,
                        0,#acc_train_set,
                        epoch_time_sum / test_epoch,
                        train_loss / n_samples))

                # early_stopping
                if (train_loss / n_samples) < best_som_loss_so_far or best_som_loss_so_far == -1:
                    best_som_loss_so_far = train_loss / n_samples
                    som_n_epochs_without_improvements = 0
                    best_epoch_som=epoch
                    print("--ES--")
                    print("save_new_best_model, with loss:",best_som_loss_so_far)
                    print("------")

                    self.save_model(test_name)

                elif (train_loss / n_samples) >= best_som_loss_so_far + early_stopping_threshold_som:
                    som_n_epochs_without_improvements += 1
                else:
                    som_n_epochs_without_improvements = 0

                if som_n_epochs_without_improvements >= max_n_epochs_without_improvements:
                    
                    print("___Early Stopping at epoch ",best_epoch_som,"____" )
                    break
                
                train_loss, n_samples = 0, 0
                epoch_time_sum = 0

                
                
        #load best model from previuos step
        self.load_model(test_name)

        print("train read_out part")
        self.stop_grad("readout")
        self.training_phase(n_epochs=n_epochs_readout,
                            optimizer=self.out_optimizer,
                            loader_train=loader_train,
                            loader_test=loader_test,
                            loader_valid=loader_valid,
                            test_epoch=test_epoch,
                            log_file_name=test_name,
                            split_id=split_id,
                            log_path=log_path,
                            use_conv_out=False,
                            test_name=test_name,
                            early_stopping_threshold=early_stopping_threshold,
                            max_n_epochs_without_improvements=max_n_epochs_without_improvements
                            )


        print("fine_tune model part")
        
        #load best model from previuos step
        self.load_model(test_name)
        self.stop_grad("fine_tuning")
        self.training_phase(n_epochs=n_epochs_fine_tuning,
                            optimizer=self.fine_tune_optimizer,
                            loader_train=loader_train,
                            loader_test=loader_test,
                            loader_valid=loader_valid,
                            test_epoch=test_epoch,
                            log_file_name="_fine_tuning_"+test_name,
                            split_id=split_id,
                            log_path=log_path,
                            use_conv_out=False,
                            test_name=test_name,
                            early_stopping_threshold=early_stopping_threshold,
                            max_n_epochs_without_improvements=max_n_epochs_without_improvements
                            )
        os.remove('./' + test_name + '.pt')



    def training_phase(self,n_epochs, optimizer, loader_train, loader_test, loader_valid, test_epoch, log_file_name,
                       split_id, log_path, use_conv_out, test_name, early_stopping_threshold,
                       max_n_epochs_without_improvements):

        train_log, test_log, valid_log = prepare_log_files(log_file_name + "--split-" + str(split_id), log_path)

        train_loss, n_samples = 0.0, 0

        epoch_time_sum = 0
        
        best_epoch=0
        best_loss_so_far=-1
        n_epochs_without_improvements=0

        for epoch in range(n_epochs):
            self.model.train()

            epoch_start_time = time.time()
            for batch in loader_train:

                data = batch.to(self.device)

                optimizer.zero_grad()

                if use_conv_out:
                    _, _, out = self.model(data, True)
                else:
                    out, _, _ = self.model(data)

                loss = self.criterion(out, data.y)

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(out)
                n_samples += len(out)

            epoch_time = time.time() - epoch_start_time
            epoch_time_sum += epoch_time

            if epoch % test_epoch == 0:

                if use_conv_out:
                    acc_train_set, correct_train_set, n_samples_train_set, loss_train_set = self.eval_model(
                        loader_train,
                        "conv_out")
                    acc_test_set, correct_test_set, n_samples_test_set, loss_test_set = self.eval_model(loader_test,
                                                                                                        "conv_out")
                    acc_valid_set, correct_valid_set, n_samples_valid_set, loss_valid_set = self.eval_model(
                        loader_valid,
                        "conv_out")
                else:
                    acc_train_set, correct_train_set, n_samples_train_set, loss_train_set = self.eval_model(
                        loader_train)
                    acc_test_set, correct_test_set, n_samples_test_set, loss_test_set = self.eval_model(loader_test)
                    acc_valid_set, correct_valid_set, n_samples_valid_set, loss_valid_set = self.eval_model(
                        loader_valid)

                print("epoch : ", epoch, " -- loss: ", train_loss / n_samples, "--- valid_loss: ",loss_valid_set)

                print("split : ", split_id, " -- training acc : ",
                      (acc_train_set, correct_train_set, n_samples_train_set), " -- test_acc : ",
                      (acc_test_set, correct_test_set, n_samples_test_set),
                      " -- valid_acc : ", (acc_valid_set, correct_valid_set, n_samples_valid_set))
                print("------")

                train_log.write(
                    "{:d}\t{:d}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        epoch,
                        split_id,
                        loss_train_set,
                        acc_train_set,
                        epoch_time_sum / test_epoch,
                        train_loss / n_samples))

                train_log.flush()

                test_log.write(
                    "{:d}\t{:d}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        epoch,
                        split_id,
                        loss_test_set,
                        acc_test_set,
                        epoch_time_sum / test_epoch,
                        train_loss / n_samples))

                test_log.flush()

                valid_log.write(
                    "{:d}\t{:d}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        epoch,
                        split_id,
                        loss_valid_set,
                        acc_valid_set,
                        epoch_time_sum / test_epoch,
                        train_loss / n_samples))

                valid_log.flush()


                if loss_valid_set < best_loss_so_far or best_loss_so_far==-1:
                    best_loss_so_far=loss_valid_set
                    n_epochs_without_improvements = 0
                    best_epoch=epoch
                    print("--ES--")
                    print("save_new_best_model, with loss:",best_loss_so_far)
                    print("------")
                    self.save_model(test_name)

                elif loss_valid_set >= best_loss_so_far + early_stopping_threshold:
                    n_epochs_without_improvements+=1
                else:
                    n_epochs_without_improvements = 0

                if n_epochs_without_improvements >= max_n_epochs_without_improvements:
                    
                    print("___Early Stopping at epoch ",best_epoch,"____" )
                    break


                train_loss, n_samples = 0, 0
                epoch_time_sum = 0



    def eval_model(self, loader, sub_model="read_out"):
        '''
        function that compute the accuracy of the model given a dataset
        :param loader: dataset used to evaluate the model performance
        :return: accuracy, number samples classified correctly, total number of samples, average loss
        '''
        self.model.eval()
        correct = 0
        n_samples = 0
        loss = 0.0
        for batch in loader:
            data = batch.to(self.device)

            if sub_model =="conv_out":
                _, _, model_out = self.model(data,True)
            else:
                model_out, _, _ = self.model(data)

            pred = predict_fn(model_out)
            n_samples += len(model_out)
            correct += pred.eq(data.y.detach().cpu().view_as(pred)).sum().item()
            loss += self.criterion(model_out, data.y).item() * len(model_out)

        acc = 100. * correct / n_samples
        return acc, correct, n_samples, loss / n_samples

    def save_model(self,test_name):
        torch.save(self.model.state_dict(), './'+test_name+'.pt')

    def load_model(self,test_name):
        self.model.load_state_dict(torch.load('./'+test_name+'.pt'))