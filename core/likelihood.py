from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
from utils import custom_KDE

class likelihood():
    
    def __init__(self, ens_model, inputs, weight_type="importance", 
                  c_w2=1, c_w3=1, tol=1e-5):

        self.model = ens_model
        self.inputs = inputs
        self.weight_type = self.check_weight_type(weight_type)
        self.c_w2 = c_w2
        self.c_w3 = c_w3
        self.tol = tol

    def update_model(self, model):
        self.model = model
        self._prepare_likelihood()
        return self

    def evaluate(self, x):
        w = self._evaluate_raw(x)
        return w

    def _evaluate_raw(self, x):
        
        fx = self.inputs.pdf(x)
        
        if self.weight_type == "nominal": 
            w = fx
            
        elif self.weight_type == "importance":
            mu = self.model._predict_mean_var(x)[0].flatten()
            x2, y = custom_KDE(mu, weights=fx, bw=.04).evaluate()
            self.fy_interp = InterpolatedUnivariateSpline(x2, y, k=1)
            fy = self.fy_interp(mu) 
            w = fx/fy
            
        elif self.weight_type == "importance_ho":
            mu = self.model._predict_mean_var(x)[0].flatten()
            x2, y = custom_KDE(mu, weights=fx, bw=.04).evaluate()
            self.fy_interp = InterpolatedUnivariateSpline(x2, y, k=1)
            fy = self.fy_interp(mu) 
            Jac_vecs, Hess_mat = self.model._predictive_jac_hess(x, compute_hess=True)
            fy_jac = self.fy_interp.derivative()(mu)
            term_temp = np.array([np.sum(np.array([Jac_vecs[:,i]*Hess_mat[:,i,j] for i in range(Jac_vecs.shape[1])]), axis=0) for j in range(Jac_vecs.shape[1])])
            term = np.sum(np.array([Jac_vecs[:,i]*term_temp.T[:,i] for i in range(Jac_vecs.shape[1])]),axis=0)
            term2 = fx*np.abs(fy_jac)/(2*self.fy_interp(mu)**2)*term/(np.linalg.norm(Jac_vecs, axis=1)**4 + self.c_w3*self.tol)
            
            w =  self.c_w3*np.abs(term2)
                
        return w[:,None]
    

    @staticmethod
    def check_weight_type(weight_type):
        assert(weight_type.lower() in ["nominal", "importance", "importance_ho"])
        return weight_type.lower()