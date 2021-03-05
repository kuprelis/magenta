# Copyright 2019-2020, the e-prop team:
# Guillaume Bellec, Franz Scherr, Anand Subramoney, Elias Hajek, Darjan Salaj, Robert Legenstein, Wolfgang Maass
# from the Institute for theoretical computer science, TU Graz, Austria.

import tensorflow.compat.v1 as tf
import numpy as np
from collections import namedtuple

rnn = tf.nn.rnn_cell


def pseudo_derivative(v_scaled, dampening_factor):
    """
    Define the pseudo derivative used to derive through spikes.

    Args:
      v_scaled: scaled version of the voltage being 0 at threshold and -1 at rest
      dampening_factor: parameter that stabilizes learning
    """
    return tf.maximum(1 - tf.abs(v_scaled), 0) * dampening_factor


@tf.custom_gradient
def SpikeFunction(v_scaled, dampening_factor):
    """
    The TensorFlow function which is defined as a Heaviside function (to compute the spikes),
    but with a gradient defined with the pseudo derivative.

    Args:
      v_scaled: scaled version of the voltage being -1 at rest and 0 at the threshold
      dampening_factor: parameter to stabilize learning

    Returns:
      The spike tensor.
    """
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, dtype=tf.float32)

    def grad(dy):
        dE_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)
        dE_dv_scaled = dE_dz * dz_dv_scaled

        return [dE_dv_scaled, tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="SpikeFunction"), grad


ALIFCellStateTuple = namedtuple('ALIFCellStateTuple',
                                ('s', 'z', 'z_local', 'r'))


class ALIFCell(rnn.BasicRNNCell):

    def __init__(self,
                 n_in,
                 n_rec,
                 tau=20.,
                 thr=0.03,
                 dt=1.,
                 dtype=tf.float32,
                 dampening_factor=0.3,
                 tau_adaptation=200.,
                 beta=1.6,
                 stop_z_gradients=False,
                 n_refractory=1):
        """
        A TensorFlow RNN cell model to simulate Adaptive Leaky Integrate and Fire (ALIF) neurons.

        Args:
          n_in: number of input neurons
          n_rec: number of recurrent neurons
          tau: membrane time constant
          thr: spike threshold voltage
          dt: time step
          dtype: data type
          dampening_factor: pseudo-derivative parameter for learning stabilization
          tau_adaptation: time constant of adaptive threshold decay
          beta: impact of adapting thresholds
          stop_z_gradients: if true, some gradients are stopped to get an equivalence between e-prop and BPTT
          n_refractory: number of refractory time steps
        """

        if tau_adaptation is None:
            raise ValueError("alpha parameter for adaptive bias must be set")
        if beta is None:
            raise ValueError("beta parameter for adaptive bias must be set")

        self.n_refractory = n_refractory
        self.tau_adaptation = tau_adaptation
        self.beta = beta
        self.decay_b = np.exp(-dt / tau_adaptation)

        if np.isscalar(tau):
            tau = tf.ones(n_rec, dtype=dtype) * np.mean(tau)
        if np.isscalar(thr):
            thr = tf.ones(n_rec, dtype=dtype) * np.mean(thr)

        tau = tf.cast(tau, dtype=dtype)
        dt = tf.cast(dt, dtype=dtype)

        self.dampening_factor = dampening_factor
        self.stop_z_gradients = stop_z_gradients
        self.dt = dt
        self.n_in = n_in
        self.n_rec = n_rec
        self.data_type = dtype

        self._num_units = self.n_rec

        self.tau = tau
        self._decay = tf.exp(-dt / tau)
        self.thr = thr

        with tf.variable_scope('InputWeights'):
            self.w_in_var = tf.Variable(np.random.randn(n_in, n_rec) /
                                        np.sqrt(n_in),
                                        dtype=dtype)
            self.w_in_val = tf.identity(self.w_in_var)

        with tf.variable_scope('RecWeights'):
            self.w_rec_var = tf.Variable(np.random.randn(n_rec, n_rec) /
                                         np.sqrt(n_rec),
                                         dtype=dtype)
            self.recurrent_disconnect_mask = np.diag(np.ones(n_rec, dtype=bool))
            self.w_rec_val = tf.where(
                self.recurrent_disconnect_mask, tf.zeros_like(self.w_rec_var),
                self.w_rec_var)  # Disconnect self-connection

        self.variable_list = [self.w_in_var, self.w_rec_var]
        self.built = True

    @property
    def state_size(self):
        return ALIFCellStateTuple(s=tf.TensorShape((self.n_rec, 2)),
                                  z=self.n_rec,
                                  r=self.n_rec,
                                  z_local=self.n_rec)

    @property
    def output_size(self):
        return [self.n_rec, tf.TensorShape((self.n_rec, 2))]

    def zero_state(self, batch_size, dtype, n_rec=None):
        if n_rec is None:
            n_rec = self.n_rec

        s0 = tf.zeros(shape=(batch_size, n_rec, 2), dtype=dtype)
        z0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
        z_local0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
        r0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)

        return ALIFCellStateTuple(s=s0, z=z0, r=r0, z_local=z_local0)

    def compute_z(self, v, b):
        adaptive_thr = self.thr + b * self.beta
        v_scaled = (v - adaptive_thr) / self.thr
        z = SpikeFunction(v_scaled, self.dampening_factor)
        z = z * 1 / self.dt
        return z

    def compute_v_relative_to_threshold_values(self, hidden_states):
        v = hidden_states[..., 0]
        b = hidden_states[..., 1]

        adaptive_thr = self.thr + b * self.beta
        v_scaled = (v - adaptive_thr) / self.thr
        return v_scaled

    def __call__(self,
                 inputs,
                 state,
                 scope=None,
                 dtype=tf.float32,
                 stop_gradient=None):

        decay = self._decay
        z = state.z
        z_local = state.z_local
        s = state.s
        r = state.r
        v, b = s[..., 0], s[..., 1]

        # This stop_gradient allows computing e-prop with auto-diff.
        #
        # needed for correct auto-diff computation of gradient for threshold adaptation
        # stop_gradient: forward pass unchanged, gradient is blocked in the backward pass
        use_stop_gradient = stop_gradient if stop_gradient is not None else self.stop_z_gradients
        if use_stop_gradient:
            z = tf.stop_gradient(z)

        new_b = self.decay_b * b + z_local  # threshold update does not have to depend on the stopped-gradient-z, it's local

        i_t = tf.matmul(inputs, self.w_in_val) + tf.matmul(
            z, self.w_rec_val)  # gradients are blocked in spike transmission
        I_reset = z * self.thr * self.dt
        new_v = decay * v + i_t - I_reset

        # Spike generation
        is_refractory = r > 0
        zeros_like_spikes = tf.zeros_like(z)
        new_z = tf.where(is_refractory, zeros_like_spikes,
                         self.compute_z(new_v, new_b))
        new_z_local = tf.where(is_refractory, zeros_like_spikes,
                               self.compute_z(new_v, new_b))
        new_r = r + self.n_refractory * new_z - 1
        new_r = tf.stop_gradient(
            tf.clip_by_value(new_r, 0., float(self.n_refractory)))
        new_s = tf.stack((new_v, new_b), axis=-1)

        new_state = ALIFCellStateTuple(s=new_s,
                                       z=new_z,
                                       r=new_r,
                                       z_local=new_z_local)
        return [new_z, new_s], new_state

    def compute_eligibility_traces(self, v_scaled, z_pre, z_post, is_rec):

        n_neurons = tf.shape(z_post)[2]
        rho = self.decay_b
        beta = self.beta
        alpha = self._decay
        n_ref = self.n_refractory

        # everything should be time major
        z_pre = tf.transpose(z_pre, perm=[1, 0, 2])
        v_scaled = tf.transpose(v_scaled, perm=[1, 0, 2])
        z_post = tf.transpose(z_post, perm=[1, 0, 2])

        psi_no_ref = self.dampening_factor / self.thr * tf.maximum(
            0., 1. - tf.abs(v_scaled))

        update_refractory = lambda refractory_count, z_post:\
            tf.where(z_post > 0,tf.ones_like(refractory_count) * (n_ref - 1),tf.maximum(0, refractory_count - 1))

        refractory_count_init = tf.zeros_like(z_post[0], dtype=tf.int32)
        refractory_count = tf.scan(update_refractory,
                                   z_post[:-1],
                                   initializer=refractory_count_init)
        refractory_count = tf.concat(
            [[refractory_count_init], refractory_count], axis=0)

        is_refractory = refractory_count > 0
        psi = tf.where(is_refractory, tf.zeros_like(psi_no_ref), psi_no_ref)

        update_epsilon_v = lambda epsilon_v, z_pre: alpha[
            None, None, :] * epsilon_v + z_pre[:, :, None]
        epsilon_v_zero = tf.ones((1, 1, n_neurons)) * z_pre[0][:, :, None]
        epsilon_v = tf.scan(
            update_epsilon_v,
            z_pre[1:],
            initializer=epsilon_v_zero,
        )
        epsilon_v = tf.concat([[epsilon_v_zero], epsilon_v], axis=0)

        update_epsilon_a = lambda epsilon_a, elems:\
                (rho - beta * elems['psi'][:, None, :]) * epsilon_a + elems['psi'][:, None, :] * elems['epsi']

        epsilon_a_zero = tf.zeros_like(epsilon_v[0])
        epsilon_a = tf.scan(
            fn=update_epsilon_a,
            elems={
                'psi': psi[:-1],
                'epsi': epsilon_v[:-1],
                'previous_epsi': shift_by_one_time_step(epsilon_v[:-1])
            },
            initializer=epsilon_a_zero)

        epsilon_a = tf.concat([[epsilon_a_zero], epsilon_a], axis=0)

        e_trace = psi[:, :, None, :] * (epsilon_v - beta * epsilon_a)

        # everything should be time major
        e_trace = tf.transpose(e_trace, perm=[1, 0, 2, 3])
        epsilon_v = tf.transpose(epsilon_v, perm=[1, 0, 2, 3])
        epsilon_a = tf.transpose(epsilon_a, perm=[1, 0, 2, 3])
        psi = tf.transpose(psi, perm=[1, 0, 2])

        if is_rec:
            identity_diag = tf.eye(n_neurons)[None, None, :, :]
            e_trace -= identity_diag * e_trace
            epsilon_v -= identity_diag * epsilon_v
            epsilon_a -= identity_diag * epsilon_a

        return e_trace, epsilon_v, epsilon_a, psi

    def compute_loss_gradient(self,
                              learning_signal,
                              z_pre,
                              z_post,
                              v_post,
                              b_post,
                              decay_out=None,
                              zero_on_diagonal=None):
        thr_post = self.thr + self.beta * b_post
        v_scaled = (v_post - thr_post) / self.thr

        e_trace, epsilon_v, epsilon_a, _ = self.compute_eligibility_traces(
            v_scaled, z_pre, z_post, zero_on_diagonal)

        if decay_out is not None:
            e_trace_time_major = tf.transpose(e_trace, perm=[1, 0, 2, 3])
            filtered_e_zero = tf.zeros_like(e_trace_time_major[0])
            filtering = lambda filtered_e, e: filtered_e * decay_out + e * (
                1 - decay_out)
            filtered_e = tf.scan(filtering,
                                 e_trace_time_major,
                                 initializer=filtered_e_zero)
            filtered_e = tf.transpose(filtered_e, perm=[1, 0, 2, 3])
            e_trace = filtered_e

        gradient = tf.einsum('btj,btij->ij', learning_signal, e_trace)
        return gradient, e_trace, epsilon_v, epsilon_a


def shift_by_one_time_step(tensor, initializer=None):
    """
    Shift the input on the time dimension by one.

    Args:
      tensor: a tensor of shape (trial, time, neuron)
      initializer: pre-prend this as the new first element on the time dimension

    Returns:
      A shifted tensor of shape (trial, time, neuron)
    """
    with tf.name_scope('TimeShift'):
        assert tensor.dtype in [tf.float16, tf.float32, tf.float64]
        r_shp = range(len(tensor.get_shape()))
        transpose_perm = [1, 0] + list(r_shp)[2:]
        tensor_time_major = tf.transpose(tensor, perm=transpose_perm)

        if initializer is None:
            initializer = tf.zeros_like(tensor_time_major[0])

        shifted_tensor = tf.concat(
            [initializer[None, :, :], tensor_time_major[:-1]], axis=0)

        shifted_tensor = tf.transpose(shifted_tensor, perm=transpose_perm)
    return shifted_tensor
