import torch

class default_potential :
    def __init__(self):
        return
    'Calculating the potential energy'
    def calc_potential_energy(self, yy):
        return 0.5*torch.dot(yy,yy)

# poten=default_potential()
# vv=poten.calc_potential_energy( torch.FloatTensor([3,1]))
# print vv
#
