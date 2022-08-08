from ensemble_model import UQ_NN
from base_model import MyModel
import time
from acquisition_fcn import acquisition_fcn
import numpy as np


class active_sampling():
    def __init__(self, data, obj_fcn, inputs=None, epochs=10000, batch_size=500):
        self.data = data
        self.inputs = inputs
        self.epochs = epochs
        self.obj_fcn = obj_fcn
        self.batch_size = batch_size
        
        
        self.ens_model = UQ_NN(MyModel, self.data, epochs=self.epochs, ens_verbose=False)
    
    def optimize(self, acquisition, n_iter, num_restarts=0, callback=True):
        
        self.acquisition = acquisition
        
        ens_model_list = []

        for ii in range(n_iter+1):
            if ii == 0:
                ens_model_list.append(self.ens_model)
                
            tic = time.time()
            
            acq = acquisition_fcn(acquisition=acquisition, ens_model=self.ens_model, inputs=self.inputs)
            
            x_opt_index = np.argmax(acq.eval_acq())
            
            x_opt = acq.x[x_opt_index]

            x_opt = np.atleast_2d(x_opt)
            y_opt = self.obj_fcn().evaluate(x_opt)            

            self.data[0] = np.vstack((self.data[0], x_opt))
            self.data[1] = np.vstack((self.data[1], y_opt))
            
            self.ens_model = UQ_NN(MyModel, self.data, epochs=self.epochs, ens_verbose=False)
            ens_model_list.append(self.ens_model)

            if callback:
                self._callback(ii, time.time()-tic)

           
        return ens_model_list

    @staticmethod
    def _callback(ii, time):
         m, s = divmod(time, 60)
         print("Iteration {:3d} \t Optimization completed in {:02d}:{:02d}"
               .format(ii, int(m), int(s)))  