from likelihood import likelihood

class acquisition_fcn():
    def __init__(self, acquisition, ens_model, inputs):
        self.acquisition = acquisition
        self.model = ens_model
        self.likelihood = likelihood
        self.inputs = inputs
        self.x = inputs.draw_samples(n_samples=int(1e2), sample_method="lhs") # sample_method = "uni" or "lhs"
        
    def us(self):
        acq = self.likelihood(self.model, self.inputs, weight_type='nominal')._evaluate_raw(self.x).flatten()*self.model._predict_mean_var(self.x)[1].flatten()
        return acq
        
    def us_lw(self):
        acq = self.likelihood(self.model, self.inputs, weight_type='importance')._evaluate_raw(self.x).flatten()*self.model._predict_mean_var(self.x)[1].flatten()
        return acq
    
    def us_lgw(self):
        acq = self.likelihood(self.model, self.inputs, weight_type='importance_ho')._evaluate_raw(self.x).flatten()*self.model._predict_mean_var(self.x)[1].flatten()
        return acq
    
    def eval_acq(self):
        if self.acquisition == "us":
            self.acq_val = self.us()
            
        elif self.acquisition == "us_lw":
            self.acq_val = self.us_lw()
            
        elif self.acquisition == "us_lgw":
            self.acq_val = self.us_lgw()
            
        else:
            print('Acquisition function not available!')
        
        return self.acq_val
    
    @staticmethod
    def check_acquisition(acquisition):
        assert(acquisition.lower() in ["us", "us_lw", "us_lgw"])
        return acquisition.lower()