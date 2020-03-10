import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.activations import relu as Relu, softmax as Softmax
import math

class BNNLayer(tf.keras.layers.Layer):
    """
    Base class for BNN layers.

    Parameters
    ==========

    num_inputs: int
        Number of inputs to the layer.
    num_outputs: int
        Number of outputs from the layer.
    activation:
        Activation from tensorflow.keras.activations.
    """

    def __init__(self, num_inputs, num_outputs, activation):
        super(BNNLayer, self).__init__(dtype=tf.float32)
        self.num_outputs = num_outputs
        self.activation = activation
        self.kernel_mu = self.add_variable("kernel_mu",
                                           shape=[num_inputs,
                                                  self.num_outputs],
                                           initializer=tf.keras.initializers.TruncatedNormal(),
                                           dtype=tf.float32)
        self.kernel_rho = self.add_variable("kernel_sigma",
                                            shape=[num_inputs,
                                                   self.num_outputs],
                                            initializer=tf.keras.initializers.TruncatedNormal(),
                                            dtype=tf.float32)

    def _reparametrize(self):
        """
        Abstract method which implements the
        reparametrisation technique.
        """
        return None

    def call(self, input, weights):
        prod = self.activation(tf.matmul(input, weights))
        return tf.clip_by_value(prod, 1e-6, 1.)

class BNNLayer_Normal_Normal(BNNLayer):
    """
    BNN layer which implements reparametrisation
    trick from N(0, 1) to any N(mu, sigma).
    """

    def _reparametrize(self):
        eps_w_shape = self.kernel_mu.shape
        eps_w = tf.math.abs(tf.random.normal(eps_w_shape, 0, 1, dtype=tf.float32))
        term_w = tf.math.multiply(eps_w,
                                  tf.math.add(tf.math.log(tf.math.exp(self.kernel_rho)),
                                  tf.constant(1., shape=eps_w_shape, dtype=tf.float32)))
        return tf.math.add(self.kernel_mu, term_w)

class BNN_Normal_Normal(tf.keras.Model):
    """
    BNN model which uses BNN_Layer_Normal_Normal
    stacks of layers.
    """

    def __init__(self, input_shape=None):
        super(BNN_Normal_Normal, self).__init__()
        self.InputLayer = tf.keras.layers.InputLayer(input_shape=(input_shape[1],),
                            batch_size=input_shape[0], dtype=tf.float32)
        self.Dense_1 = BNNLayer_Normal_Normal(int(input_shape[-1]), 200, activation=Relu)
        self.Dense_2 = BNNLayer_Normal_Normal(200, 200, activation=Relu)
        self.Output = BNNLayer_Normal_Normal(200, 10, activation=Softmax)

    def run(self, inputs, *weights):
        layer_output_1 = self.InputLayer(inputs)
        layer_output_2 = self.Dense_1(layer_output_1, weights[0])
        layer_output_3 = self.Dense_2(layer_output_2, weights[1])
        output = self.Output(layer_output_3, weights[2])
        return output

    def log_prior(self, weights):
        """
        Computes the natural logarithm of scale
        mixture prior of weights.

        Note
        ====

        The two standard deviations of the scale mixture are,
        exp(-1) and exp(-2). The weight of both normal distributions
        is 0.5.
        """
        shape = weights.shape
        sigma_1 = tf.constant(math.exp(-1), shape=shape, dtype=tf.float32)
        sigma_2 = tf.constant(math.exp(-6), shape=shape, dtype=tf.float32)
        pdf = lambda w, sigma: ((tf.math.exp(-0.5*(tf.math.square(w))/(tf.math.square(sigma))))/
                                (tf.math.sqrt(2*math.pi)*tf.math.abs(sigma)))
        part_1 = tf.clip_by_value(0.5*pdf(weights, sigma_1), 1e-10, 1.)
        part_2 = tf.clip_by_value(0.5*pdf(weights, sigma_2), 1e-10, 1.)
        return tf.math.reduce_sum(tf.math.log(part_1 + part_2))

    def log_posterior(self, weights, mu, rho):
        """
        Computes the natural logarithm of Gaussian
        posterior on weights.
        """
        sigma = tf.math.log(1 + tf.math.exp(rho))
        pdf = lambda w, mu, sigma: ((tf.math.exp(-0.5*(tf.math.square(w - mu))/(tf.math.square(sigma))))/
                                    (tf.math.sqrt(2*math.pi)*tf.math.abs(sigma)))
        log_q = tf.math.log(tf.clip_by_value(pdf(weights, mu, sigma), 1e-10, 1.))
        return tf.math.reduce_sum(log_q)

    def get_loss(self, inputs, targets, samples, weight=1., inference=False):
        """
        Computes the total training loss.

        Parameters
        ==========

        inputs
            Input to the layers.
        targets
            True targets that the model wants to learn from.
        weight
            Weight given to loss of each batch. By default, 1.
        """
        loss = tf.constant(0., dtype=tf.float32)
        outputs_list = []
        for _ in range(samples):
            weights_1 = self.Dense_1._reparametrize()
            kernel_mu, kernel_rho = self.Dense_1.kernel_mu, self.Dense_1.kernel_rho
            pw = self.log_prior(weights_1)
            qw = self.log_posterior(weights_1, kernel_mu, kernel_rho)

            weights_2 = self.Dense_2._reparametrize()
            kernel_mu, kernel_rho = self.Dense_2.kernel_mu, self.Dense_2.kernel_rho
            pw += self.log_prior(weights_2)
            qw += self.log_posterior(weights_2, kernel_mu, kernel_rho)

            weights_3 = self.Output._reparametrize()
            kernel_mu, kernel_rho = self.Output.kernel_mu, self.Output.kernel_rho
            pw += self.log_prior(weights_3)
            qw += self.log_posterior(weights_3, kernel_mu, kernel_rho)

            # print(weights_1, weights_2, weights_3)

            outputs = self.run(inputs, weights_1, weights_2, weights_3)
            outputs_list.append(outputs)
            cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=targets))
            if inference:
                loss += cross_entropy
            else:
                loss += (qw - pw)*weight + cross_entropy

        if inference:
            return outputs_list, loss/samples
        return loss/samples

    def compute_gradients(self, inputs, targets, weight):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.Dense_1.kernel_mu)
            tape.watch(self.Dense_2.kernel_mu)
            tape.watch(self.Output.kernel_mu)
            tape.watch(self.Dense_1.kernel_rho)
            tape.watch(self.Dense_2.kernel_rho)
            tape.watch(self.Output.kernel_rho)
            F = self.get_loss(inputs, targets, 10, weight)
            dF_dwmu1, dF_dwmu2, dF_dwmu3, \
            dF_dwrho1, dF_dwrho2, dF_dwrho3, = \
            tape.gradient(F, [
            self.Dense_1.kernel_mu,
            self.Dense_2.kernel_mu,
            self.Output.kernel_mu,
            self.Dense_1.kernel_rho,
            self.Dense_2.kernel_rho,
            self.Output.kernel_rho])

        return [dF_dwmu1, dF_dwmu2, dF_dwmu3,
                dF_dwrho1, dF_dwrho2, dF_dwrho3]

    def learn(self, inputs, targets, weight=1.):
        grads = self.compute_gradients(inputs, targets, weight)
        self.Dense_1.kernel_mu = tf.math.subtract(self.Dense_1.kernel_mu, tf.scalar_mul(0.001, grads[0]))
        self.Dense_2.kernel_mu = tf.math.subtract(self.Dense_2.kernel_mu, tf.scalar_mul(0.001, grads[1]))
        self.Output.kernel_mu = tf.math.subtract(self.Output.kernel_mu, tf.scalar_mul(0.001, grads[2]))
        self.Dense_1.kernel_rho = tf.math.subtract(self.Dense_1.kernel_rho, tf.scalar_mul(0.001, grads[3]))
        self.Dense_2.kernel_rho = tf.math.subtract(self.Dense_2.kernel_rho, tf.scalar_mul(0.001, grads[4]))
        self.Output.kernel_rho = tf.math.subtract(self.Output.kernel_rho, tf.scalar_mul(0.001, grads[5]))
