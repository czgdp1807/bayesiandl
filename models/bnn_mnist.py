import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.activations import relu as Relu, softmax as Softmax
import math

class BNNLayer(tf.keras.layers.Layer):

    def __init__(self, num_outputs, activation):
        super(BNNLayer, self).__init__()
        self.num_outputs = num_outputs
        self.activation = activation

    def build(self, input_shape):
        self.kernel_mu = self.add_variable("kernel_mu",
                                           shape=[int(input_shape[-1]),
                                                   self.num_outputs],
                                           initializer=tf.keras.initializers.glorot_uniform())
        self.kernel_rho = self.add_variable("kernel_sigma",
                                            shape=[int(input_shape[-1]),
                                                    self.num_outputs],
                                            initializer=tf.keras.initializers.glorot_uniform())
        self.bias_mu = self.add_variable("bias_mu",
                                         shape=[self.num_outputs],
                                         initializer=tf.keras.initializers.glorot_uniform())
        self.bias_rho = self.add_variable("bias_rho",
                                          shape=[self.num_outputs],
                                          initializer=tf.keras.initializers.glorot_uniform())
        self._weights = self.add_variable("weights",
                                          shape=[int(input_shape[-1]),
                                                 self.num_outputs])
        self._bias = self.add_variable("bias",
                                       shape=[int(input_shape[-1]),
                                              self.num_outputs])

    def _reparametrize(self):
        return None

    def call(self, input):
        self._reparametrize()
        return self.activation(tf.matmul(input, self._weights) + self._bias)

class BNNLayer_Normal_Normal(BNNLayer):

    def _reparametrize(self):
        eps_shape = self.kernel_mu.shape
        eps_w = tfp.distributions.Normal(0, 1).sample(eps_shape)
        eps_b = tfp.distributions.Normal(0, 1).sample(eps_shape)
        term_w = tf.math.multiply(eps_w, tf.math.log(
                                         tf.constant(1, shape=eps_shape) +
                                         tf.math.exp(self.kernel_rho)))
        term_b = tf.math.multiply(eps_b, tf.math.log(
                                         tf.constant(1, shape=eps_shape) +
                                         tf.math.exp(self.bias_rho)))

        self._weights.assign(self.kernel_mu + term_w)
        self._bias.assign(self.bias_mu + term_b)
        self._eps_w, self._eps_b = eps_w, eps_b
        return weights, bias

class BNN_Normal_Normal(tf.keras.Model):
    def __init__(self, input_shape=None):
        super(BNN, self).__init__()
        init = tf.keras.initializers.glorot_uniform()
        self.InputLayer = tf.keras.layers.InputLayer(input_shape=input_shape)
        self.Dense_1 = BNNLayer_Normal_Normal(200, activation=Relu)
        self.Dense_2 = BNNLayer_Normal_Normal(200, activation=Relu)
        self.Output = BNNLayer_Normal_Normal(10, activation=Softmax)

    def run(self, inputs):
        layer_output_1 = self.InputLayer(inputs)
        layer_output_2 = self.Dense_1(layer_output_1)
        layer_output_3 = self.Dense_2(layer_output_2)
        return self.Output(layer_output_3)

    def prior(self, weights, pi, sigma_1, sigma_2):
        pdf = lambda weights, sigma: tf.scalar_mul(
                1/(((2*math.pi)**0.5)*sigma),
                tf.math.exp(tf.math.scalar_mul(-0.5,
                tf.math.scalar_mul(1/sigma**2, tf.math.square(weights))))
              )
        return tf.add(tf.scalar_mul(pi, pdf(weights, sigma_1)),
                      tf.scalar_mul(1 - pi, pdf(weights, sigma_2)))

    def posterior(self, weights, mu, rho, eps):
        shape = mu.shape
        sigma = tf.math.multiply(eps, tf.math.log(
                                 tf.constant(1, shape=shape) +
                                 tf.math.exp(rho)))
        pdf = lambda weights, mu, sigma: tf.multiply(
                tf.reciprocal(tf.scalar_mul((2*math.pi)**0.5, sigma)),
                tf.math.exp(tf.scalar_mul(-0.5,
                            tf.math.xdivy(tf.math.squared_difference(weights, mu),
                                          tf.math.square(sigma)))
                           )
                )
        return pdf(weights, mu, sigma)


    def get_loss(self, inputs, targets):
        weights_1 = self.Dense_1._weights
        kernel_mu, kernel_rho = self.Dense_1.kernel_mu, self.Dense_1.kernel_rho
        bias_mu, bias_rho = self.Dense_1.bias_mu, self.Dense_1.bias_rho
        eps_w, eps_b = self.Dense_1._eps_w, self.Dense_1._eps_b
        pw = tf.math.log(self.prior(weights_1, 0.5, 0.01, 0.001))
        qw = tf.math.log(self.posterior(weights_1, kernel_mu, kernel_rho, eps_w))
        b_1 = self.Dense_1._bias
        pb = tf.math.log(self.prior(b_1, 0.5, 0.01, 0.001))
        qb = tf.math.log(self.posterior(b_1, bias_mu, bias_rho, eps_b))

        kernel_mu, kernel_rho = self.Dense_2.kernel_mu, self.Dense_2.kernel_rho
        bias_mu, bias_rho = self.Dense_2.bias_mu, self.Dense_2.bias_rho
        eps_w, eps_b = self.Dense_2._eps_w, self.Dense_2._eps_b
        weights_2 = self.Dense_2._weights
        pw = tf.math.add(pw, tf.math.log(self.prior(weights_2, 0.5, 0.01, 0.001)))
        qw = tf.math.add(qw, tf.math.log(self.posterior(weights_2, kernel_mu, kernel_rho, eps_w)))
        b_2 = self.Dense_2._bias
        pb = tf.math.add(pb, tf.math.log(self.prior(b_2, 0.5, 0.01, 0.001)))
        qb = tf.math.add(qb, tf.math.log(self.posterior(b_2, bias_mu, bias_rho, eps_b)))

        kernel_mu, kernel_rho = self.Output.kernel_mu, self.Output.kernel_rho
        bias_mu, bias_rho = self.Output.bias_mu, self.Output.bias_rho
        eps_w, eps_b = self.Output._eps_w, self.Output._eps_b
        weights_3 = self.Output._weights
        pw = tf.math.add(pw, tf.math.log(self.prior(weights_3, 0.5, 0.01, 0.001)))
        qw = tf.math.add(qw, tf.math.log(self.posterior(weights_3, kernel_mu, kernel_rho, eps_w)))
        b_3 = self.Output._bias
        pb = tf.math.add(pb, tf.math.log(self.prior(b_3, 0.5, 0.01, 0.001)))
        qb = tf.math.add(qb, tf.math.log(self.posterior(b_3, bias_mu, bias_rho, eps_b)))

        prior = tf.math.add(pw, pb)
        posterior = tf.math.add(qw, qb)

        outputs = self.run(inputs)
        return tf.math.reduce_sum(posterior) - tf.math.reduce_sum(prior) - \
               tf.math.reduce_sum(tf.keras.losses.categorical_crossentropy(targets, outputs))

    def compute_gradients(self, inputs, targets):
        with tf.GradientTape() as tape:
            tape.watch(self.Dense_1._weights)
            tape.watch(self.Dense_2._weights)
            tape.watch(self.Output._weights)
            tape.watch(self.Dense_1._bias)
            tape.watch(self.Dense_2._bias)
            tape.watch(self.Output._bias)
            tape.watch(self.Dense_1.kernel_mu)
            tape.watch(self.Dense_2.kernel_mu)
            tape.watch(self.Output.kernel_mu)
            tape.watch(self.Dense_1.kernel_rho)
            tape.watch(self.Dense_2.kernel_rho)
            tape.watch(self.Output.kernel_rho)
            tape.watch(self.Dense_1.bias_mu)
            tape.watch(self.Dense_2.bias_mu)
            tape.watch(self.Output.bias_mu)
            tape.watch(self.Dense_1.bias_rho)
            tape.watch(self.Dense_2.bias_rho)
            tape.watch(self.Output.bias_rho)
            F = self.get_loss(inputs, targets)
            dF_dw1 = tape.gradient(F, self.Dense_1._weights)
            dF_dw2 = tape.gradient(F, self.Dense_2._weights)
            dF_dw3 = tape.gradient(F, self.Output._weights)
            dF_db1 = tape.gradient(F, self.Dense_1._bias)
            dF_db2 = tape.gradient(F, self.Dense_2._bias)
            dF_db3 = tape.gradient(F, self.Output._bias)
            dF_dwmu1 = tape.gradient(F, self.Dense_1.kernel_mu)
            dF_dwmu2 = tape.gradient(F, self.Dense_2.kernel_mu)
            dF_dwmu3 = tape.gradient(F, self.Output.kernel_mu)
            dF_dwrho1 = tape.gradient(F, self.Dense_1.kernel_rho)
            dF_dwrho2 = tape.gradient(F, self.Dense_2.kernel_rho)
            dF_dwrho3 = tape.gradient(F, self.Output.kernel_rho)
            dF_dbmu1 = tape.gradient(F, self.Dense_1.bias_mu)
            dF_dbmu2 = tape.gradient(F, self.Dense_2.bias_mu)
            dF_dbmu3 = tape.gradient(F, self.Output.bias_mu)
            dF_dbrho1 = tape.gradient(F, self.Dense_1.bias_rho)
            dF_dbrho2 = tape.gradient(F, self.Dense_2.bias_rho)
            dF_dbrho3 = tape.gradient(F, self.Output.bias_rho)

        kernel_grad_mu_1 = dF_dw1 + dF_dwmu1
        kernel_grad_mu_2 = dF_dw2 + dF_dwmu2
        kernel_grad_mu_3 = dF_dw3 + dF_dwmu3
        bias_grad_mu_1 = dF_db1 + dF_dbmu1
        bias_grad_mu_2 = dF_db2 + dF_dbmu2
        bias_grad_mu_3 = dF_db3 + dF_dbmu3
        factor = lambda eps, rho: tf.math.xdivy(eps,
                                  tf.add(
                                  tf.constant(1., rho),
                                  tf.math.exp(tf.math.scalar_mul(-1., rho)))
                                  )
        kernel_grad_rho_1 = dF_dw1*factor(self.Dense_1._eps_w, self.Dense_1.kernel_rho) + dF_dwrho1
        kernel_grad_rho_2 = dF_dw2*factor(self.Dense_2._eps_w, self.Dense_2.kernel_rho) + dF_dwrho2
        kernel_grad_rho_3 = dF_dw3*factor(self.Output._eps_w, self.Output.kernel_rho) + dF_dwrho3
        bias_grad_rho_1 = dF_db1*factor(self.Dense_1._eps_b, self.Dense_1.bias_rho) + dF_dbrho1
        bias_grad_rho_2 = dF_db2*factor(self.Dense_2._eps_b, self.Dense_2.bias_rho) + dF_dbrho2
        bias_grad_rho_3 = dF_db3*factor(self.Output._eps_b, self.Output.bias_rho) + dF_dbrho3
        return [kernel_grad_mu_1, kernel_grad_mu_2, kernel_grad_mu_3,
                bias_grad_mu_1, bias_grad_mu_2, bias_grad_mu_3,
                kernel_grad_rho_1, kernel_grad_rho_2, kernel_grad_rho_3,
                bias_grad_rho_1, bias_grad_rho_2, bias_grad_rho_3]

    def update(self, inputs, targets):
        grads = self.compute_gradients(inputs, targets)
        self.train_op.apply_gradients(zip(grads,
        [self.Dense_1.kernel_mu, self.Dense_2.kernel_mu, self.Output.kernel_mu,
         self.Dense_1.bias_mu, self.Dense_2.bias_mu, self.Output.bias_mu,
         self.Dense_1.kernel_rho, self.Dense_2.kernel_rho, self.Output.kernel_rho,
         self.Dense_1.bias_rho, self.Dense_2.bias_rho, self.Output.bias_rho
        ]))
