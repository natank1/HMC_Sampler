import torch

class vanilla_potential :
    def __init__(self):
        return
    'Calculating the potential energy'
    def calc_potential_energy(self, yy):
        return 0.5*torch.dot(yy,yy)

#
