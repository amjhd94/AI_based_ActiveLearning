import sys
sys.path.append('core/')
from active_sampling import active_sampling
from inputs import *
from core.utils import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()
sns.set_style("whitegrid")

#%% Objective function

class obj_fcn():
    def evaluate(self, x):
        y = 0*(x<0) + np.sqrt(np.abs(x))*(x>=0)
        return y
    
domain = [ [-1, 1] ]
inputs = UniformInputs(domain)
pts = inputs.draw_samples(n_samples=int(1e3), sample_method="grd")
x = pts
y = obj_fcn().evaluate(x)

pdf_orig_data = custom_KDE(y, bw=0.01)
_, orig_pdf = pdf_orig_data.evaluate()

#%% Initial datasetand

n_init = 2
x_init = np.random.uniform(low=-1, high=1, size=(n_init,1))

y_init =  0*(x_init<0) + np.sqrt(np.abs(x_init))*(x_init>=0)

#%% Active sampling

n_iter = 40
data_init = [x_init, y_init]

NN_active_sampling = active_sampling(data_init, obj_fcn, inputs=inputs, epochs=1500, batch_size=1)
ens_model_list = NN_active_sampling.optimize(acquisition='us', n_iter=n_iter)

#%% Testing the final model 

x_test = np.random.uniform(low=-1, high=1, size=(1000,1))
x_test = np.sort(x_test, axis=0)

log_pdf_error = []
R2_map = []
for i in range(0, n_iter+1, 1):
    print(i)
    ens_model_i = ens_model_list[i]
    
    mean, var = ens_model_i._predict_mean_var(x_test)
    
    pdf_model_data = custom_KDE(mean, bw=0.01)
    _, model_pdf = pdf_model_data.evaluate()
    
    x_min = min( pdf_model_data.data.min(), pdf_orig_data.data.min() )
    x_max = max( pdf_model_data.data.max(), pdf_orig_data.data.max() )
    rang = x_max-x_min
    x_eva = np.linspace(x_min - 0.01*rang,
                        x_max + 0.01*rang, 1024)
    
    yb, yt = pdf_model_data.evaluate(x_eva), pdf_orig_data.evaluate(x_eva)
    log_yb, log_yt = np.log(yb), np.log(yt)
    
    np.clip(log_yb, -6, None, out=log_yb)
    np.clip(log_yt, -6, None, out=log_yt)
    
    log_diff = np.abs(log_yb-log_yt)
    noInf = np.isfinite(log_diff)
    
    log_pdf_error.append(np.trapz(log_diff[noInf], x_eva[noInf]))
    
    
    tipping_point_interval = ((x > -.1)*(x < .1)).flatten()
    
    corr_matrix_map = np.corrcoef(y.flatten(), mean.flatten())
    corr_map = corr_matrix_map[0,1]
    R_sq_map = corr_map**2
    R2_map.append(R_sq_map)
    
    if i%10 == 0:    
        plt.figure()
        plt.plot(x, y, 'k', lw=2, label='Orig. fcn.')
        plt.plot(x_test, mean, '--r', lw=2, label='model mean')
        plt.ylim([-.2, 1.5])
        plt.scatter(NN_active_sampling.data[0][:n_init], NN_active_sampling.data[1][:n_init], s=100, c='y', label='Initial points')
        plt.scatter(NN_active_sampling.data[0][n_init:(i+n_init+1)], NN_active_sampling.data[1][n_init:(i+n_init+1)], s=100, c='r', label='Sampled points')
        plt.xlabel('$\mu$')
        plt.ylabel('$r = \sqrt{x^2+y^2}$')
        plt.legend(loc='upper left')
        plt.tight_layout()

plt.figure()
plt.subplot(1,2,1)
plt.plot(np.array(log_pdf_error))
plt.xlabel('Iterations')
plt.ylabel('log pdf error')
plt.subplot(1,2,2)
plt.plot(np.array(R2_map))
plt.xlabel('Iterations')
plt.ylabel('function and model $R^2$')


