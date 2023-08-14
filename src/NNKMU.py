import torch
import numpy as np
from nnk_utils.nnk_means import NNK_Means, kmeans_plusplus, NNK_EC_Means
from scipy.spatial.distance import cdist
from tqdm import tqdm


class NNKMU():
    def __init__(self, num_epochs=15, n_components=100, top_k=20, 
                use_error_based_buffer=False, use_residual_update=False, 
                nnk_tol=-1, metric='error', model=None, ep=None, weighted=False,
                num_warmup=5, num_cooldown=2):
        
        self.epochs = num_epochs
        self.n_components = n_components
        self.top_k = top_k
        self.use_error_based_buffer = use_error_based_buffer
        self.use_residual_update = use_residual_update
        self.nnk_tol = nnk_tol
        self.metric = metric
        self.model = model
        self.ep = ep
        self.weighted = weighted
        self.warmup = num_warmup
        self.cooldown = num_cooldown

    def train(self, dataloader, warm_up=True):

        for batch_x in dataloader:
            batch_x = batch_x.cuda()
            if self.ec:
                _, _, _, _, _, error = self.model(batch_x,  update_cache=True, update_dict=False, warm_up=warm_up)
            else: 
                _, _, _, _, _, error = self.model(batch_x,  update_cache=True, update_dict=False)

        return error
    
    def eval(self, dataloader):
        x_opt_list, indices_list, error_list = [], [], []
        for batch_x in dataloader:
            batch_x = batch_x.cuda()
            _, _, _, x_opt, indices, error = self.model(batch_x, update_cache=False, update_dict=False)
            x_opt_list, indices_list, error_list = x_opt_list + [x_opt], indices_list + [indices], error_list + [error]
        x_opt, indices, error = torch.cat(x_opt_list, dim=0), torch.cat(indices_list, dim=0), torch.cat(error_list, dim=0)
        
        return x_opt, indices, error
    
    def fit(self, X_train, y_train=None, batch_size=32, shuffle=True, num_workers=1, drop_last=False):        

        X_train = torch.from_numpy(X_train)
        if self.ep == None or self.ep == 0.0:
            self.model = NNK_Means(n_components=self.n_components, n_nonzero_coefs=self.top_k, n_classes=None, 
                                use_error_based_buffer=self.use_error_based_buffer, 
                                use_residual_update=self.use_residual_update, 
                                nnk_tol=self.nnk_tol)
            self.ec=False
        else: 
            self.model = NNK_EC_Means(n_components=self.n_components, n_nonzero_coefs=self.top_k, n_classes=None, 
                                use_error_based_buffer=self.use_error_based_buffer, 
                                use_residual_update=self.use_residual_update, 
                                nnk_tol=self.nnk_tol, ep=self.ep, weighted_ec=self.weighted)
            self.ec=True
        
        train_loader = torch.utils.data.DataLoader(X_train.float(), batch_size=batch_size, 
                                                   shuffle=shuffle, num_workers=num_workers, 
                                                   drop_last=drop_last)
        
        init_indices = kmeans_plusplus(X_train.float(), self.n_components)        
        self.model.initialize_dictionary(X_train.float()[init_indices])

        for i in tqdm(range(self.epochs)):
            if not(self.ec) or (i < self.warmup or i >= (self.epochs - self.cooldown)):
                error = self.train(train_loader)
                self.model.update_dict()
            else:
                error = self.train(train_loader, warm_up=False)
                self.model.update_dict(warm_up=False)
        
        self.n_components = self.model.n_components

        return error.cpu()

    def get_codes(self, X, batch_size=32, shuffle=False, num_workers=1, drop_last=False):
        data_loader = torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
        x_opt, indices, _ = self.eval(data_loader)
        x_opt = x_opt.cpu()
        indices = indices.cpu()
        sparse_codes = np.zeros((len(x_opt), self.n_components), dtype=np.float32)
        sparse_codes[np.arange(len(x_opt))[:, None], indices] = x_opt

        return torch.tensor(sparse_codes)
    
    def hamming_distance(self, test_codes, train_codes):
        distances = torch.cdist(torch.tensor(test_codes), torch.tensor(train_codes), p=0)
        min_distances, min_indices = torch.min(distances, dim=1)
        return min_distances

    def manhattan_distance(self, test_codes, train_codes):
        distances = torch.cdist(torch.tensor(test_codes), torch.tensor(train_codes), p=1)
        min_distances, min_indices = torch.min(distances, dim=1)
        return min_distances

    def euclidean_distance(self, test_codes, train_codes):
        distances = torch.cdist(torch.tensor(test_codes), torch.tensor(train_codes), p=2)
        min_distances, min_indices = torch.min(distances, dim=1)
        return min_distances

    def mahalanobis_distance(self, test_data, train_data):
        if isinstance(train_data, torch.Tensor) and isinstance(test_data, torch.Tensor):
            train_data_np = train_data.numpy()
            test_data_np = test_data.numpy()
        else:
            train_data_np = train_data
            test_data_np = test_data 

        cov_matrix = np.cov(train_data_np, rowvar=False)

        # Compute the inverse of the covariance matrix
        inv_cov_matrix = np.linalg.inv(cov_matrix)

        mean = np.mean(train_data_np, axis=0)
        diff = test_data_np - mean

        # Compute the Mahalanobis distance
        distances = cdist(diff, np.zeros_like(mean)[None, :], metric='mahalanobis', VI=inv_cov_matrix)

        return torch.from_numpy(distances.squeeze())
    
    def save_model(self, file):
        torch.save(self.model, file)

    def predict_score(self, X_test, X_train=None, multi_eval=False, eval_metrics=None):
        if self.metric != 'error' and X_train is None: 
            raise RuntimeError('Using metric ' + self.metric + ' without providing X_train')

        eval_loader = torch.utils.data.DataLoader(X_test, batch_size=32, shuffle=False, num_workers=1, drop_last=False)
        x_opt, indices, error = self.eval(eval_loader)

        if self.metric == 'error':
            return error.cpu()
        else: 
            test_codes = np.zeros((len(x_opt), self.n_components), dtype=np.float32)
            for i in range(len(x_opt)):
                for j in range(len(x_opt[i])):
                    test_codes[i][indices[i][j]] = x_opt[i][j]

            train_loader = torch.utils.data.DataLoader(X_train, batch_size=32, shuffle=False, num_workers=1, drop_last=False)
            x_opt, indices, error = self.eval(train_loader)

            train_codes = np.zeros((len(x_opt), self.n_components), dtype=np.float32)
            for i in range(len(x_opt)):
                for j in range(len(x_opt[i])):
                    train_codes[i][indices[i][j]] = x_opt[i][j]

            if self.metric == 'hamming':
                return (self.hamming_distance(test_codes, train_codes)).cpu()
            elif self.metric == 'manhattan':
                return (self.manhattan_distance(test_codes, train_codes)).cpu()
            elif self.metric == 'euclid':
                return (self.euclidean_distance(test_codes, train_codes)).cpu()
            elif self.metric == 'mahalanobis':
                return (self.mahalanobis_distance(test_codes, train_codes)).cpu()
            else:
                raise NotImplementedError("unrecognized metric.")


    def predict_score_multi(self, X_test, X_train, eval_metrics=None):

        eval_loader = torch.utils.data.DataLoader(X_test, batch_size=32, shuffle=False, num_workers=1, drop_last=False)
        x_opt, indices, error = self.eval(eval_loader)

        results = {}
        test_codes = np.zeros((len(x_opt), self.n_components), dtype=np.float32)
        for i in range(len(x_opt)):
            for j in range(len(x_opt[i])):
                test_codes[i][indices[i][j]] = x_opt[i][j]

        train_loader = torch.utils.data.DataLoader(X_train, batch_size=32, shuffle=False, num_workers=1, drop_last=False)
        x_opt, indices, train_error = self.eval(train_loader)

        train_codes = np.zeros((len(x_opt), self.n_components), dtype=np.float32)
        for i in range(len(x_opt)):
            for j in range(len(x_opt[i])):
                train_codes[i][indices[i][j]] = x_opt[i][j]
        for m in eval_metrics:
            if m == 'error':
                results[m] = error.cpu()
            elif m == "hamming":
                results[m] = (self.hamming_distance(test_codes, train_codes)).cpu()
            elif m == 'manhattan': 
                results[m] = (self.manhattan_distance(test_codes, train_codes)).cpu()
            elif m == 'euclid':
                results[m] = (self.euclidean_distance(test_codes, train_codes)).cpu()
            elif m == 'mahalanobis':
                results[m] = (self.mahalanobis_distance(test_codes, train_codes)).cpu()
            else:
                raise NotImplementedError("unrecognized metric" + m)
        return results