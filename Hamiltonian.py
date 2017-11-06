import torch
# import binary_potential
from default_potential import vanilla_potential as df_poten



class hamilton_operator :

    def __init__(self, potential_struct=None):

        if potential_struct is None :
            self.potential=df_poten()
        else :
            self.potential =potential_struct
        return

        '''Kinetic energy (assuming mass is 1'''
    def kinetic_energy(self, velocity):
            return 0.5 * torch.dot(velocity, velocity)

    def hamiltonian_measure(self,position, velocity, pot_val=None):
            """Computes the Hamiltonian of the current position, velocity pair
            H = U(x) + K(v)
            U is the potential energy and is = -log_posterior(x)
            Parameters
            ----------
            position : tf.Variable
                Position or state vector x (sample from the target distribution)
            velocity : tf.Variable
                Auxiliary velocity variable
            energy_function
                Function from state to position to 'energy'
                 = -log_posterior
            Returns
            -------
            hamitonian : float
            """

            kinetic_val = self.kinetic_energy(velocity)

            if pot_val is None:

                ham_val = self.potential.calc_potential_energy(position) + kinetic_val
            else:
                ham_val = pot_val + kinetic_val

            return ham_val
