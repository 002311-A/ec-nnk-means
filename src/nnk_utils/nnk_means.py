import torch
import torch.nn as nn

def approximate_nnk(AtA, b, x_init, x_tol=1e-6, num_iter=100, eta=None):
    """Performs approimate nnk using iterative thresholding similar to ISTA"""
    if eta is None:  # this slows down the solve - can get a fixed eta by taking mean over some sample
        values, indices = torch.max(torch.linalg.eigvalsh(AtA).abs(), 1, keepdim=True)
        eta = 1. / values.unsqueeze(2)

    b = b.unsqueeze(2)
    x_opt = x_init.unsqueeze(2)
    for t in range(num_iter):
        grad = b.sub(torch.bmm(AtA, x_opt))
        x_opt = x_opt.add(eta * grad).clamp(min=torch.cuda.FloatTensor([0.]), max=b)

    x_opt[x_opt < x_tol] = 0
    error = - 2*torch.sum(x_opt*b.sub(0.5*torch.bmm(AtA, x_opt)), dim=1)
    return x_opt.squeeze(-1), error.squeeze(-1)

def kmeans_plusplus(X, n_components):
    """ Utils function for obtianing indices for initialization of dictionary atoms"""
    n_samples = X.shape[0]
    indices = torch.zeros(n_components).long()

    indices[0] = torch.randint(n_samples, size=(1,))
    dists = torch.sum((X - X[indices[0]]) ** 2, 1)

    for i in range(1, n_components):
        index = torch.multinomial(dists, 1)
        indices[i] = index
        dists = torch.minimum(dists, torch.sum((X - X[index]) ** 2, 1))
    
    return indices


# %% NNK Means
class NNK_Means(nn.Module):
    def __init__(self, n_components=100, n_nonzero_coefs=50, momentum=1.0, n_classes=None, influence_tol=1e-4, optim_itr=1000, optim_lr=None, optim_tol=1e-6, 
                    use_error_based_buffer=True, use_residual_update=False,  **kwargs):
        """
        Learn a dictionary representation in an online manner based on nonnegative sparse coding leveraging local neighborhoods
        objective: \sum_{i=1}^N ||x_n - Dw_n||^2 with constraints w_n > 0
        
        n_components: No. of dictionary atoms to learn
        n_nonzero_coeffs: Initial "k" nearest neigbors to use for NNK sparse coding
        momentum: The dictionary update cache is acummulated over each forward call - Mometum weighs the current update before addition
            - Call self.reset_cache() after forward call and momemtum=1 to remove accumulated cache
        n_classes: No. of classes in the input data 
            - Set to zero for regression scenario
            - Set to None for no labels
        influence_tol: Tolerance value to remove atoms that are not used for representation
        optim_itr, optim_lr, optim_tol: Approximate NNK parameters
            - Set optim_lr to None to set learning rate automatically using the max eigenvalue of local AtA
        use_error_based_buffer - strategy to use for saving some data for replacing unused atoms 
            - NNK coding error based (default), random
        use_residual_update: Use error residual each atom is responsible for to update the dictionary 
        kwargs: Other arguments that gets used by derived classes
        """
        super(NNK_Means, self).__init__()
        self.dictionary_atoms = []
        self.dictionary_atoms_norm = []
        self.atom_labels = []
        
        self.data_cache = None
        self.label_cache = None
        self.influence_cache = None
        self.momentum = momentum
        self.influence_tol = influence_tol
        
        self.n_classes = n_classes
        self.n_components = n_components
        
        #%% NNK optimization parameters
        self.n_nonzero_coefs = n_nonzero_coefs
        self.optim_itr = optim_itr
        self.optim_lr = optim_lr
        self.optim_tol = optim_tol
        
        #%% maintain buffer to replace dictionary atoms
        self.dictionary_data_buffer = []
        self.dictionary_label_buffer = []
        self.associated_error = None
        self.use_error_based_buffer = use_error_based_buffer
        
        self.use_residual_update = use_residual_update
        self.kwargs = kwargs
    
    @torch.no_grad()
    def _process_data(self, data):
        return nn.functional.normalize(data, dim=1)
    
    def _process_labels(self, labels):
        if self.n_classes > 0:
            return nn.functional.one_hot(labels, self.n_classes).float()
        #return labels.float()
    
    @torch.no_grad()
    def initialize_dictionary(self, initial_data, initial_labels=None):
        self.dictionary_atoms = initial_data.cuda()
        self.dictionary_atoms_norm = self._process_data(self.dictionary_atoms)
        if self.n_classes is not None:
            self.atom_labels = self._process_labels(initial_labels).cuda()
        
        self._set_cache()
    
    def _set_cache(self):
        self.dictionary_data_buffer = torch.clone(self.dictionary_atoms) #.cuda()
        self.data_cache = torch.zeros_like(self.dictionary_atoms) #.cuda()
        
        self.associated_error = torch.zeros(self.n_components).cuda()
        
        if self.n_classes is not None:
            self.dictionary_label_buffer = torch.clone(self.atom_labels)
            self.label_cache = torch.zeros_like(self.atom_labels) #.cuda()
            
        self.influence_cache = torch.zeros((self.n_components, self.n_components), dtype=torch.float32).cuda() # , self.n_components
    
    def reset_cache(self):
        self._set_cache()
    
    @torch.no_grad()
    def _update_cache(self, batch_W, batch_data, batch_label):
        self.data_cache = self.data_cache + self.momentum*torch.sparse.mm(batch_W, batch_data)
        self.influence_cache = self.influence_cache + self.momentum*torch.sparse.mm(batch_W, batch_W.t()) # (1-self.momentum)*
        if self.n_classes is not None:
            self.label_cache = self.label_cache + self.momentum*torch.sparse.mm(batch_W, batch_label)
    
    @torch.no_grad()
    def _update_buffer(self, batch_data, batch_label=None, error=1):
        indices = torch.arange(self.n_components) # set default to maintain the data buffer
        if self.use_error_based_buffer:
            if error.min() > self.associated_error.min():
                self.associated_error, indices = torch.topk(torch.cat((self.associated_error, error)), self.n_components, sorted=True)
        
        else: # Randomly substitute elements in buffer with elements from batch_data
            indices = torch.randint(0, self.n_components + batch_data.shape[0], size=(self.n_components,), 
                                    device=self.dictionary_data_buffer.device)
        
        temp_data_buffer = torch.cat((self.dictionary_data_buffer, batch_data))
        self.dictionary_data_buffer = temp_data_buffer[indices]
        
        if self.n_classes is not None:
            temp_label_buffer = torch.cat((self.dictionary_label_buffer, batch_label))
            self.dictionary_label_buffer = temp_label_buffer[indices]
    
    def _calculate_similarity(self, input1, input2, batched_inputs=False):
        input1 = input1.cuda()
        input2 = input2.cuda()
        if batched_inputs:
            return torch.bmm(input1, input2.transpose(1,2)) 
            
        return input1 @ input2.t()
        
    @torch.no_grad()
    def _sparse_code(self, batch_data):
        similarities = self._calculate_similarity(batch_data, self.dictionary_atoms_norm)
        sub_similarities, sub_indices = torch.topk(similarities, self.n_nonzero_coefs, dim=1)
        support_matrix = self.dictionary_atoms_norm[sub_indices]
        support_similarites = self._calculate_similarity(support_matrix, support_matrix, batched_inputs=True)
        x_opt, error = approximate_nnk(support_similarites, sub_similarities, sub_similarities, x_tol=self.optim_tol,
                                                    num_iter=self.optim_itr, eta=self.optim_lr)
        self_similarities = torch.sum(batch_data*batch_data, axis=1)
        self_similarities = self_similarities.cuda()
        error = (self_similarities + error) / self_similarities ## We need the self_similarities to compute the error

        x_opt = nn.functional.normalize(x_opt, p=1, dim=1) # the normalization provides shift invariance w.r.t origin

        return x_opt, sub_indices, error
    
    @torch.no_grad()
    def _update_dict_inv(self):
        nonzero_indices = torch.nonzero(self.influence_cache.diag() > self.influence_tol).squeeze(0) 
        n_nonzero = len(nonzero_indices)
        if not(n_nonzero < self.n_components):
            WWt_inv = torch.linalg.inv(self.influence_cache)
            self.dictionary_atoms = WWt_inv @ self.data_cache
            if self.n_classes is not None:
                self.atom_labels = WWt_inv @ self.label_cache
        
        self.dictionary_atoms_norm = self._process_data(self.dictionary_atoms)
    
    @torch.no_grad()
    def _update_dict_residual(self):
        n_nonzero = 0
        for i in range(self.n_components):
            influence_i = self.influence_cache[i]
            if influence_i[i] < self.influence_tol:
                self.dictionary_atoms[i] = self.dictionary_data_buffer[n_nonzero]
                if self.n_classes is not None:
                    self.atom_labels[i] = self.dictionary_label_buffer[n_nonzero]
                n_nonzero += 1
                
            else:
                self.dictionary_atoms[i] += (self.data_cache[i] - influence_i @ self.dictionary_atoms)/influence_i[i]

        self.dictionary_atoms_norm = self._process_data(self.dictionary_atoms)    
    
    @torch.no_grad()
    def update_dict(self):
        if self.use_residual_update:
            self._update_dict_residual()
        else:
            self._update_dict_inv()
        
    
    def forward(self, batch_data, batch_label=None, update_cache=True, update_dict=True, get_codes=False):
        # batch_data = nn.functional.normalize(batch_data, dim=1)
        batch_size = batch_data.shape[0]
        
        x_opt, indices, error = self._sparse_code(batch_data)
    
        if update_cache:
            batch_row_indices = torch.arange(0, batch_size, dtype=torch.long).cuda().unsqueeze(1)
            batch_W = torch.sparse_coo_tensor(torch.stack((indices.ravel(), torch.tile(batch_row_indices, [1, self.n_nonzero_coefs]).ravel()), 0), x_opt.ravel(),
                                 (self.n_components, batch_size), dtype=torch.float32) #  # batch_row_indices.ravel()
            if self.n_classes is not None:
                batch_label = self._process_labels(batch_label)
                
            self._update_cache(batch_W, batch_data, batch_label)# 
            self._update_buffer(batch_data, batch_label, error)
        if update_dict:
            self.update_dict()
            
        interpolated = torch.bmm(x_opt.unsqueeze(1), self.dictionary_atoms[indices]).squeeze(1)
        label_interpolated = None
        if self.n_classes is not None:
            label_interpolated = torch.bmm(x_opt.unsqueeze(1), self.atom_labels[indices]).squeeze(1)
            
        if get_codes: 
            return batch_data, interpolated, label_interpolated, batch_W.t().to_dense()

        return batch_data, interpolated, label_interpolated, x_opt, indices, error 

class NNK_EC_Means(NNK_Means):
    def __init__(self, ep=0.01, weighted_ec=False, n_components=1000, n_nonzero_coefs=50, momentum=1.0, n_classes=None, influence_tol=1e-4, optim_itr=1000, optim_lr=None, optim_tol=1e-6, 
                    use_error_based_buffer=True, use_residual_update=False,  **kwargs):
        
        super().__init__(n_components, n_nonzero_coefs, momentum, n_classes, influence_tol, optim_itr, optim_lr, optim_tol, 
                    use_error_based_buffer, use_residual_update,  **kwargs)
        self.ep = ep
        self.weighted_ec = weighted_ec
        self.dict_counts = torch.zeros(self.n_components).long().cuda()
        self.dict_probs = torch.ones(self.n_components).cuda()
        self.dict_weights = torch.zeros(self.n_components).cuda()

    def _calculate_similarity(self, input1, input2, batched_inputs=False):
        k = super()._calculate_similarity(input1, input2, batched_inputs)
        if batched_inputs:
            return k
        return k + (self.ep * torch.log(self.dict_probs.unsqueeze(0)))
    
    def forward(self, batch_data, batch_label=None, update_cache=True, update_dict=False, get_codes=False, warm_up=True):
            batch_data = self._process_data(batch_data)
            batch_data, interpolated, label_interpolated, x_opt, indices, error =  super().forward(batch_data, batch_label, update_cache, update_dict)

            x_opt_flat = x_opt.flatten()
            indices_flat = indices.flatten()
            mask = x_opt_flat.nonzero(as_tuple=True)[0]

            x_opt_nonzero = x_opt_flat[mask]
            indices_nonzero = indices_flat[mask]

            indices_nonzero = indices_nonzero.long()

            self.dict_counts.scatter_add_(0, indices_nonzero, torch.ones_like(indices_nonzero).cuda())
            self.dict_weights.scatter_add_(0, indices_nonzero, x_opt_nonzero)

            if not(warm_up):
                if self.weighted_ec:
                    self.dict_probs = self.dict_weights / torch.sum(self.dict_weights)
                else: 
                    self.dict_probs = self.dict_counts / torch.sum(self.dict_counts)

            return batch_data, interpolated, label_interpolated, x_opt, indices, error
    
    @torch.no_grad()
    def update_dict(self, warm_up=True):
        if self.use_residual_update:
            self._update_dict_residual()
        else:
            self._update_dict_inv()
        
        if not(warm_up):
            if self.weighted_ec:
                self.dict_probs = self.dict_weights / torch.sum(self.dict_weights)
            else: 
                self.dict_probs = self.dict_counts / torch.sum(self.dict_counts)

            unused_atoms = (self.dict_counts == 0).nonzero(as_tuple=True)[0]

            mask = torch.ones(self.dictionary_atoms.shape[0], dtype=torch.bool)
            mask[unused_atoms] = False
            filtered_atoms = self.dictionary_atoms[mask]

            self.n_components = filtered_atoms.shape[0]

            self.dictionary_atoms = filtered_atoms
            self.dictionary_atoms_norm = self._process_data(self.dictionary_atoms)

            self.dict_counts = torch.zeros(self.n_components).long().cuda()
            self.dict_weights = torch.zeros(self.n_components).cuda()

            ## Keep the probabilities from the previous epoch for the first batch of the next one
            self.dict_probs = self.dict_probs[mask] 

            super().reset_cache()

