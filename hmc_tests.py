from HMC_sampler import sampler
import torch
from   torch.autograd import Variable
import numpy as np
#Regular run
hmc = sampler(sample_size=100,position_dim=5)
sample,_= hmc.main_hmc_loop()

print "Init place"
hmc = sampler(sample_size=100,init_position=np.array([1.0,2.1,3.1,-1.]))
sample,_= hmc.main_hmc_loop()
print sample

class test_potnetial:

    def __init__(self, weight_matrix):
        self.weight_matrix = weight_matrix


    def calc_potential_energy (self, xx):
        potential_energy=torch.dot(xx,torch.matmul(self.weight_matrix,xx))
        return potential_energy
#Regular run
print "Potential"


Amat = Variable(torch.FloatTensor([[2, .0, .0, -0.], [0., 2.0, 0., 0.], [0., 0., 2., 0.], [0.0, 0., 0, 2.]]), requires_grad=False)
hmc = sampler(sample_size=100,position_dim=4,potential_struct=test_potnetial(Amat))
sample,_= hmc.main_hmc_loop()
print sample

print "Init vel"
hmc = sampler(sample_size=100,position_dim=4,init_velocity=np.array([1.0,2.1,3.1,-1.]))
sample,_= hmc.main_hmc_loop()
print sample

