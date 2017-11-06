import torch


class Binary_potnetial:

    def __init__(self, weight_matrix,bias_array=None):
        self.weight_matrix = weight_matrix
        if bias_array is None :
            self.bias_array = None
        else:
            self.bias_array= bias_array


    def calc_potential_energy (self, binary_config):
        potential_energy=torch.dot(binary_config,torch.matmul(self.weight_matrix, binary_config))
        if self.bias_array is None:
            return potential_energy

        potential_energy +=torch.dot(binary_config,self.bias_array)
        return potential_energy

#
# binary = Binary_potnetial(torch.FloatTensor([[3.,0.],[.0,3.]]),torch.FloatTensor([3.,0.]))
# print binary.calc_potential_energy(torch.FloatTensor([1.0,1.0]))