import numpy as np
import pandas as pd

import copy
import math

# pytorch imports
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score

"""
Implementing these explicitly here, replace if I every make an outside AL package
"""


"""
Model Utils - model parameter upload, reading params correctly, etc
"""

def get_model(config_dict, dev = 'cpu'):
    model_type_dict = {'ensemble':EnsembleModel, 'dropout':BaseModel}
    return model_type_dict[config_dict['uncertainty_method']](config_dict, dev)

"""
Datasets/Dataloaders
"""

def get_dataloaders(data, config_dict, fp_col = "fp", label_col = "label", 
                        batch_size = 128, num_workers = 0, val = False, predict = False, seed = 27):
    
    if config_dict['uncertainty_method'] != 'ensemble' or val or predict:
        return DataLoader(FingerprintDataset(data, fp_col = fp_col, label_col = label_col, 
                            output_type = config_dict['output_type'], 
                            options = config_dict['output_options'],
                            predict = predict),
                            batch_size = batch_size, num_workers = num_workers,
                            shuffle = not (val or predict)) # don't shuffle validation
    else:
        np.random.seed(seed) # reproducibility
        # need to create multiple dataloaders and return them all
        if config_dict['method_dict']['data_split'] == 'bootstrap': # sample with replacement
            indices = [np.random.choice(data.shape[0], size = data.shape[0]) for i in range(config_dict['method_dict']['committee_size'])]
        else:  # 
            shuffled_idx = np.random.choice(data.shape[0], size = data.shape[0], replace = False)
            step = math.ciel(data.shape[0] / config_dict['method_dict']['committee_size'])
            indices = [shuffled_idx[i * step : (i+1) * step] for i in range(config_dict['method_dict']['committee_size'])]
    
        return [DataLoader(FingerprintDataset(data.iloc[ind], fp_col = fp_col, label_col = label_col, 
                            output_type = config_dict['output_type'], options = config_dict['output_options']),
                            batch_size = batch_size, num_workers = num_workers,
                            shuffle = True) for ind in indices]  # ensemble training, always shuffle
    

def collapse(input_array):
    return sum(input_array > 0.5) - 1
    
class FingerprintDataset(Dataset):
    
    def __init__(self, df, fp_col = 'fp', label_col = "label", 
            output_type = 'ordinal', options = None, predict = False):

        # are provided columns valid
        if not predict and (fp_col not in df.columns or label_col not in df.columns):
            raise ValueError("Dataframe does not contain fingerprint column and/or label column.")

        # is provided output_type valid
        if output_type not in ['ordinal', 'classes']:
            raise ValueError("Invalid output type, allowed options: [ordinal, classification]")

        if options is None:
            if output_type == 'ordinal': # if ordinal labels, must provide ordering
                raise ValueError("Must provide options for ordinal dataset.")
            else:
                if not predict:
                    self.options = sorted(list(set(df[label_col].values))) # infer from provided labels

        else: self.options = options # set options

        # make sure all labels are valid options
        if not predict:
            if not all([label in set(self.options) for label in df[label_col].values]):
                raise ValueError(f"Not all found labels are allowed - allowed labels: {set(self.options)}," + \
                    f" found labels: {set(df[label_col].values)}")

        self.df = df 
        self.fp_col = fp_col
        self.label_col = label_col
        self.output_type = output_type
        self.predict = predict

        if self.options:
            self.convert = dict([(option, i) for i, option in enumerate(self.options)])

        if not predict:
            self.format_labels()
       
    def __len__(self):
        return self.df.shape[0]

    def _convert_ordinal(self, label):
        num_1s = self.convert[label] + 1
        return np.concatenate((np.ones(num_1s), np.zeros(len(self.options) - num_1s)))

    def _convert_classification(self, label):
        zeros = np.zeros(len(self.options))
        zeros[self.convert[label]] = 1
        return zeros

    def format_labels(self):
        func_dict = {'ordinal':self._convert_ordinal, 'classes':self._convert_classification}
        self.df['converted'] = self.df[self.label_col].map(func_dict[self.output_type])

    def reverse_convert(self, preds):
        if self.output_type == "ordinal":
            preds_idx = np.apply_along_axis(lambda x: sum(x > 0.5) - 1, 1, preds)
            last_one = np.apply_along_axis(lambda x: np.nonzero(x > 0.5)[0].max(), 1, preds)

            # check corner case where 1 happens after a 0
            adj_idx = (preds_idx != last_one).nonzero()[0]

            if adj_idx.shape[0] > 0:
                options = np.tril(np.ones(preds.shape[1]))
                adj_preds = (((preds[adj_idx][:, None, :] - options[None, :, :])**2).sum(-1)**0.5).argmin(axis=1)
                preds_idx[adj_idx] = adj_preds

            return [self.options[idx] for idx in preds_idx]
        else: # classes, not ordinal labels - return closest option
            if not self.options:
                return np.argmax(preds, axis = 1)
            return [self.options[idx] for idx in np.argmax(preds, axis = 1)]


    def __getitem__(self, idx):
        features = self.df[self.fp_col].iloc[idx].astype(np.float32)
        if self.predict:
            return features
        label = self.df.converted.iloc[idx].astype(np.float32)
        return features, label


"""
Models 
"""
# TODO: Update input size to match input fingerprint size
class ModelInterface(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

    def create_dataloaders(self, train_data, val_data = None, predict = False, **kwargs):
        if predict:
            return get_dataloaders(train_data, self.config_dict, predict = True, **kwargs)

        train_loader = get_dataloaders(train_data, self.config_dict, **kwargs)
        if val_data is not None:
            val_loader = get_dataloaders(val_data, self.config_dict, val = True, **kwargs)
            return train_loader, val_loader
        return train_loader

    def train(self):
        raise NotImplementedError

    def train_single_epoch(self, trainloader, valloader = None):
        raise NotImplementedError

    def predict(self, x, return_uncertainty = False):
        raise NotImplementedError

    def save_model(self, outfile):
        raise NotImplementedError

    def load_from_state(self, filematch):
        raise NotImplementedError


class BaseModel(ModelInterface):
    """
    Generic model, performs regression. Can be overwritten later if requested
    """
    def __init__(self, config_dict, dev = None):
        super().__init__()
        self.config_dict = config_dict
        # layers
        self.layers = [4096] + [config_dict['n_neurons']] * config_dict['hidden_layers'] + [len(config_dict['output_options'])] 

        layer_list = []
        for i in range(len(self.layers) - 1):
            input_size = self.layers[i]
            curr_size = self.layers[i + 1]
            
            layer_list.append(nn.Linear(input_size, curr_size))
            if i < len(self.layers) - 2:
                layer_list.append(nn.ReLU(inplace=False))
                layer_list.append(nn.Dropout(p = config_dict['dropout']))
            else:
                layer_list.append(torch.nn.Sigmoid() if config_dict['output_type'] == 'ordinal' else torch.nn.Softmax())

        self.net = nn.Sequential(*layer_list)

        # output, loss
        self.loss = nn.BCELoss()
        self.dev = dev
        self.to(dev)
        
        # optimizer
        self.optim = torch.optim.Adam(self.parameters(), lr = config_dict['learning_rate'], weight_decay = config_dict['weight_decay'])
        self.output_options = config_dict['output_options']

    def forward(self, x):
        return self.net(x)

    def train_single_epoch(self, trainloader, valloader = None):
        epoch_losses = []
        for features, labels in trainloader:
            features = features.to(self.dev, dtype=torch.float)
            labels = labels.to(self.dev, dtype=torch.float)

            self.optim.zero_grad()
            loss = self.loss(self(features), labels)
            epoch_losses.append(loss.item())
            loss.backward()
            self.optim.step()

        if valloader is not None:
            val_losses = []
            with torch.no_grad():
                for features, labels in valloader:
                    loss = self.loss(self(features.to(self.dev)), labels.to(self.dev))
                    val_losses.append(loss.item())
            return np.mean(epoch_losses, val_losses)
        return np.mean(epoch_losses)

    def predict(self, x, return_uncertainty = False):
        if return_uncertainty: raise ValueError("BaseModel does not calculate uncertainty.")

        with torch.no_grad():
            return self.net(x)   

    def save_model(self, outfile):
        torch.save(self.state_dict(), outfile + ".pt")  

    def load_from_state(self, infile):
        self.load_state_dict(torch.load(infile + ".pt"))

    
class EnsembleModel(ModelInterface):
    
    def __init__(self, config_dict, dev = None):
        super().__init__()
        self.config_dict = config_dict

        self.split_data = config_dict['method_dict']['data_split']
        self.output_options = config_dict['output_options']
        self.output_type = config_dict['output_type']
        self.dev = dev

        # create base classifiers for ensemble members
        self.members = []
        for i in range(config_dict['method_dict']['committee_size']):
            self.members.append(BaseModel(config_dict, dev))

    def train_single_epoch(self, trainloader, valloader = None):
        train_epoch_losses = []
        for i, c in enumerate(self.members):
            train_epoch_losses.append(c.train_single_epoch(trainloader[i]))

        val_loss = -1
        if valloader is not None:
            val_epoch_losses = []
            with torch.no_grad(): #dont update gradients for validation
                for values, labels in valloader:
                    values = values.to(self.dev, dtype=torch.float)
                    labels = labels.to(self.dev, dtype=torch.float)

                    output = self.predict_val(values)
                    loss = self.members[0].loss(output, labels)

                    val_epoch_losses.append(loss.item())
                val_loss = np.mean(val_epoch_losses)

        return train_epoch_losses, val_loss

    
    def predict_val(self, x):
        predictions = []
        for c in self.members:
            predictions.append(c.predict(x))
        pred_tensor = torch.stack(predictions)
        return torch.mean(pred_tensor, dim=0)
        
    
    def predict(self, x, return_uncertainty = True):
        predictions = []
        for c in self.members:
            predictions.append(c.predict(x).cpu())
        all_preds = np.stack(predictions) # (3, 50, n_options)
        means = np.mean(all_preds, axis = 0) # (50, n_options)
        if not return_uncertainty:
            return means
        cross_var = np.var(all_preds, axis = 0)
        variance = np.mean(np.var(all_preds, axis = 0), axis = 1)
        return means, variance

    def save_model(self, outfile):
        for i, member in enumerate(self.members):
            member.save_model(f"{outfile}-{i}")
    
    def load_from_state(self, infile):
        for i, member in enumerate(self.members):
            member.load_from_state(f"{infile}-{i}")

"""
Old/deprecated model definitions, need to change later

class OrdinalClassifier(nn.Module):
    def __init__(self, layers, lr, add_data, sele_crit, input_size, labels, loss, dev = None):
        super().__init__()
        
        # setup weights
        self.linears = nn.ModuleList([None]*(len(layers)-1))
        for i in range(len(layers) - 1):
            self.linears[i] = torch.nn.Linear(layers[i], layers[i+1])

        self.dev = dev#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.dev)
        
        self.optim = torch.optim.Adam(self.parameters(), lr = lr)
        self.loss = loss
        self.output = torch.nn.Sigmoid()
        self.options = labels
        
        # save other params
        self.add_data = add_data
        self.sele_crit = sele_crit
        
    def forward(self, x):
        h = x
        for linear in self.linears[:-1]:
            h = F.relu(linear(h))
        return self.output(self.linears[-1](h))
    
    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)
        
    def train(self, x_data, y_data, loss_thresh, bs = 128, 
              n_epochs = 1000, verbose = False):

        #Update model, save

        self.to(self.dev)
            
        self.X_training = x_data
        self.y_training = y_data
            
        trainloader = DataLoader(FPDataset_ordinal(self.X_training, self.y_training),
                                    batch_size = bs, shuffle = True, num_workers = 0)
        losses = []
        for epoch in range(n_epochs + 1):
            epoch_losses = []
            for batch_idx, examples in enumerate(trainloader):
                values, labels = examples 
                values = values.to(self.dev, dtype=torch.float)
                labels = labels.to(self.dev, dtype=torch.float)

                self.optim.zero_grad() #zero gradients
                output = self.forward(values) #feed forward
                
                loss = self.loss(output, labels) #get loss
                #loss = F.nll_loss(output, labels.long(), size_average = False)
                
                loss.backward()
                self.optim.step()
                
                epoch_losses.append(loss.item())
            if verbose and epoch % int(n_epochs/10) == 0:
                print(f"Epoch {epoch}: {np.mean(epoch_losses)}")
                #losses.append(np.mean(epoch_losses))
            losses.append(np.mean(epoch_losses))
            #if losses[-1] < loss_thresh:
            #    return losses
        return losses
    
    # functions to allow for validation curve
    def setup_trainloader(self, x_data, y_data, bs = 128):
        self.trainloader = DataLoader(FPDataset_ordinal(x_data, y_data, self.options),
                                    batch_size = bs, shuffle = True, num_workers = 0)
        
    def train_single_epoch(self):
        epoch_losses = []
        for batch_idx, examples in enumerate(self.trainloader):
            values, labels = examples
            values = values.to(self.dev, dtype=torch.float)
            labels = labels.to(self.dev, dtype=torch.float)

            self.optim.zero_grad()
            output = self.forward(values)
            
            loss = self.loss(output, labels)
            
            loss.backward()
            self.optim.step()
            
            epoch_losses.append(loss.item())
        return np.mean(epoch_losses)   

    class FPDataset_ordinal(Dataset):
    
    def __init__(self, X, y, options):
        self.X = X
        self.y = y
        self.options = np.sort(options)
        self.convert = {}
        
        for i, option in enumerate(self.options):
            self.convert[option] = i
       
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        features = torch.from_numpy(self.X.iloc[idx].to_numpy())
        #print(self.y.iloc[idx])
        num_1s = self.convert[self.y.iloc[idx]] + 1
        label = torch.from_numpy(np.concatenate((np.ones(num_1s), np.zeros(len(self.options) - num_1s))))
        return features, label

"""