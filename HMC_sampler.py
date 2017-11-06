import torch
import numpy as np
from  Hamiltonian import hamilton_operator as hamilton
from torch.autograd import Variable
# from default_potential import vanilla_potential as df_poten
# from binary_potential import   Binary_potnetial as bp_poten


class sampler:

    def __init__(self,sample_size, potential_struct=None,init_position=None, init_velocity= None,position_dim =None, step_size=0.05,num_steps_in_leap=20,acceptance_thr =None,duplicate_samples=False):

        self.step_size =step_size
        self.init_velocity =None
        self.half_step = 0.5*self.step_size
        self.sample_size =sample_size
        self.num_steps_in_leap = num_steps_in_leap
        self.acceptance_thr = acceptance_thr
        self.duplicate_samples = duplicate_samples
        if potential_struct is None :
            self.hamiltonian_obj = hamilton()
        else :
            self.hamiltonian_obj = hamilton(potential_struct)
        if init_velocity is None and  init_position is None and ((position_dim is None) or (position_dim<=0)  ):
            print "Neither veloctiy nor postion and nor the diemsnion has been given. Bye Bye! "
            exit(222)

        if init_velocity is None and init_position is None and (position_dim>0):
            self.pos_dim = position_dim
            # self.init_velocity = np.random.multivariate_normal(np.zeros(self.pos_dim),np.eye(self.pos_dim,self.pos_dim))
            self.init_position = np.random.multivariate_normal(np.zeros(self.pos_dim),np.eye(self.pos_dim, self.pos_dim))
        if init_velocity is None and init_position is not None :
                self.pos_dim = len(init_position)
                self.init_position =init_position
                # self.init_velocity = np.random.multivariate_normal(np.zeros(self.pos_dim), np.eye(self.pos_dim, self.pos_dim))
        if init_velocity is not None and init_position is  None:
            self.pos_dim = len(init_velocity)
            self.init_velocity = init_velocity
            self.init_position = np.random.multivariate_normal(np.zeros(self.pos_dim),
                                                               np.eye(self.pos_dim, self.pos_dim))

        if init_velocity is not None and init_position is not None:
            ll_pos= len(init_position)
            if not (ll_pos==len(init_velocity)):
                print "Lengths of init position and init velocity are equal  fix them please.Bye! "
                exit(333)
            '''Lenghts are give an equalt'''
            self.init_position = init_position
            self.init_velocity = init_velocity
            self.pos_dim = ll_pos
        self.gradient = torch.ones(2 * self.pos_dim)
        return


    def main_hmc_loop(self):
        bad_decline_cntr = 0
        sample_array= np.array([self.init_position],dtype=np.float64)
        for sample in xrange (self.sample_size):
            if sample==0 and self.init_velocity is not None:
                tmp_tensor = np.concatenate((sample_array[-1], self.init_velocity), 0)
            else :
                rand_init_velcotiy = np.random.multivariate_normal(np.zeros(self.pos_dim),np.eye(self.pos_dim, self.pos_dim))
                tmp_tensor = np.concatenate((sample_array[-1],rand_init_velcotiy),0)

            phase_tensor= Variable(torch.FloatTensor(tmp_tensor),requires_grad=True)

            new_sample = self.leap_frog_step(phase_tensor)

            if self.duplicate_samples or not(np.array_equal(new_sample,sample_array[-1])):
                sample_array = np.vstack((sample_array, new_sample))
            else :
                bad_decline_cntr+=1

        if  bad_decline_cntr > 100:
            print "Look out: Many Metropolis Hastings declines. Sample is smaller than the required"
        return sample_array, bad_decline_cntr



    def leap_frog_step(self,phase_tensor):
        on_going_phase = phase_tensor
        orig_hamitlonian = self.hamiltonian_obj.hamiltonian_measure(on_going_phase[:self.pos_dim], on_going_phase[self.pos_dim:])
        orig_hamitlonian.backward(self.gradient)
        phase_grad = on_going_phase.grad

        for step in xrange(self.num_steps_in_leap):
            # print "step=",step
            tmp_array = torch.cat((on_going_phase[:self.pos_dim] + self.step_size * phase_grad[self.pos_dim:],
                                   on_going_phase[self.pos_dim:] - self.half_step * phase_grad[:self.pos_dim]), 0)
            xx = Variable(torch.FloatTensor(tmp_array[:self.pos_dim].data), requires_grad=True)

            potential= self.hamiltonian_obj.potential.calc_potential_energy(xx)
            # potential = binary_potential(xx, weight_mat, bias_array)
            potential.backward(self.gradient[:self.pos_dim])
            tmp_array[self.pos_dim:] = tmp_array[self.pos_dim:] - self.half_step * xx.grad

            velocity = Variable(tmp_array[self.pos_dim:].data, requires_grad=True)
            on_goingrig_hamitlonian = self.hamiltonian_obj.hamiltonian_measure(tmp_array[:self.pos_dim], velocity,   pot_val=potential.data[0])

            "Prepare Hamiltonian for next iteration"
            on_goingrig_hamitlonian.backward(self.gradient[self.pos_dim:])
            phase_grad = torch.cat((xx.grad, velocity.grad), 0)

        # current_hamiltonian = hamiltonian_measure(tmp_array[:pos_dim], tmp_array[pos_dim:], potential_function, weight_mat, bias_array, pot_val= potential.data[0])
        p_accept = min(1.0, np.exp(orig_hamitlonian.data[0] - on_goingrig_hamitlonian.data[0]))

        if self.acceptance_thr is None :
            thr = np.random.uniform()
        else:
            thr = self.acceptance_thr
        if p_accept > thr:
            termination_val= tmp_array[:self.pos_dim]
        else:
            termination_val= phase_tensor[:self.pos_dim]
        return termination_val.data.numpy()

