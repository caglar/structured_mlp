import numpy
import theano
from theano import tensor as T

class Layer(object):
    """
     A general base layer class for neural networks.
    """
    def __init__(self,
        input,
        n_in=None,
        n_out=None,
        activation=theano.tensor.nnet.sigmoid,
        rng=None,
        layer_shape=None):

        self.input = input
        self.n_in = n_in
        self.n_out = n_out
        self.rng = rng
        self.W = None
        self.b = None
        self.activation = activation
        self.layer_shape = layer_shape

    def reset_layer(self, check_flag=False):
        if self.n_in is not None and self.n_out is not None:
            if self.W is None or check_flag:
                if self.activation == T.tanh:
                    low = -numpy.sqrt(6. / (self.n_in + self.n_out))
                    high = numpy.sqrt(6. / (self.n_in + self.n_out))
                elif self.activation == T.nnet.sigmoid:
                    low = -4*numpy.sqrt(6. / (self.n_in + self.n_out))
                    high = 4*numpy.sqrt(6. / (self.n_in + self.n_out))
                else:
                    low = -numpy.sqrt(1. / (self.n_in + self.n_out))
                    high = numpy.sqrt(1. / (self.n_in + self.n_out))

                W_values = numpy.asarray(self.rng.uniform(
                    low=low,
                    high=high,
                    size=(self.n_in, self.n_out)),
                    dtype=theano.config.floatX)
                if self.activation == theano.tensor.nnet.sigmoid:
                    W_values *= 4
                self.W = theano.shared(value=W_values, name='W')

            if self.b is None or check_flag:
                b_values = numpy.zeros((self.n_out), dtype=theano.config.floatX)
                self.b = theano.shared(value=b_values, name='b')
            # parameters of the model
            self.params = [self.W, self.b]

class PretrainingLayers(Layer):

    def __init__(self,
        input,
        activation=T.tanh,
        out_activation=T.nnet.softmax,
        layer_shape=(64, 64),
        n_classes=11,
        rng=None):
        if rng is None:
            rng = numpy.random.RandomState()

        self.n_classes = n_classes
        self.out_activation = out_activation
        super(PretrainingLayers, self).__init__(input,
        activation=activation,
        rng=rng,
        layer_shape=layer_shape)
        self.reset_local_layer()

    def set_input(self, input):
        self.input = input

    def reset_local_layer(self, check_flag=False):
        # Weight shape should be:
        # (No of patches, No_of_input_pixels_per_patch, No of hidden unit per
        # patch)
        weight_shape = [self.layer_shape[0], self.layer_shape[1], self.layer_shape[3]]
        self.weight_shape = weight_shape
        #No of patches, no of first hidden layer units per patch, no of current layers' hiddens
        weight_out_shape = [self.layer_shape[0], self.layer_shape[3], self.n_classes]
        self.weight_out_shape = weight_out_shape
        if self.W is None or check_flag:
            self.fan_in = self.layer_shape[1]
            self.fan_out = self.layer_shape[3]
            self.fan_in_2 = self.layer_shape[3]
            self.fan_out_2 = self.n_classes

        if self.W is None or check_flag:
            #Define the lower and upper bound for the intervals that the
            #weights will be sampled from.
            if self.activation == T.tanh:
                low = -numpy.sqrt(6. / (self.fan_in + self.fan_out))
                high = numpy.sqrt(6. / (self.fan_in + self.fan_out))
                low_out = -numpy.sqrt(6. / (self.fan_in_2 + self.fan_out_2))
                high_out = numpy.sqrt(6. / (self.fan_in_2 + self.fan_out_2))
            elif self.activation == T.nnet.sigmoid:
                low = -4*numpy.sqrt(6. / (self.fan_in + self.fan_out))
                high = 4*numpy.sqrt(6. / (self.fan_in + self.fan_out))
                low_out = -4*numpy.sqrt(6. / (self.fan_in_2 + self.fan_out_2))
                high_out = 4*numpy.sqrt(6. / (self.fan_in_2 + self.fan_out_2))
            else:
                low = -numpy.sqrt(1. / (self.fan_in + self.fan_out))
                high = numpy.sqrt(1. / (self.fan_in + self.fan_out))
                low_out = -numpy.sqrt(1. / (self.fan_in_2 + self.fan_out_2))
                high_out = numpy.sqrt(1. / (self.fan_in_2 + self.fan_out_2))

            W_values = numpy.asarray(self.rng.uniform(
                low=low,
                high=high,
                size=(self.fan_in, self.fan_out)),
                dtype=theano.config.floatX)

            W_out_values = numpy.asarray(self.rng.uniform(
                low=low_out,
                high=high_out,
                size=(self.fan_in_2, self.fan_out_2)),
                dtype=theano.config.floatX)

            self.W = theano.shared(value=W_values, name='W_local')
            self.W_out = theano.shared(value=W_out_values, name="W_out")

        b_values = numpy.zeros((self.fan_out), dtype=theano.config.floatX)
        b_values_out = numpy.zeros((self.n_classes), dtype=theano.config.floatX)

        self.b = theano.shared(value=b_values, name='b_local')
        self.b_out = theano.shared(value=b_values_out, name='b_out')

        # parameters of the model
        self.params = [self.W, self.b, self.W_out, self.b_out]

    def standardize_outputs(self, output):
        mu = T.mean(output, axis=0)
        sigma = T.std(output, axis=0)
        return (output - mu)/sigma

    def get_local_activation(self, input, w, w_out, b, b_out):
        lin_output = T.dot(input, w) + b
        activation_in = (lin_output if self.activation is None
                else self.activation(lin_output))
        activation_out = self.out_activation(T.dot(activation_in, w_out) + b_out)
        return activation_out

    def fprop(self, input):
        hidden_acts = self.hidden_activations(self.input)
        out_acts = self.output_activations(hidden_acts)
        return out_acts

    def output_activations(self, input):
        lin_act = T.dot(input, self.W_out) + self.b_out
        activation_out = self.out_activation(lin_act)
        return activation_out

    def hidden_activations(self, input):
        lin_act = T.dot(input, self.W) + self.b
        self.output = self.activation(lin_act)
        return self.output

class HiddenLayer(Layer):

    def __init__(self, input, n_in, n_out, W=None, b=None, activation=T.tanh, rng=None):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int

        :param n_in: dimensionality of input
        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden layer.
        """
        if rng is None:
            rng = numpy.random.RandomState()

        super(HiddenLayer, self).__init__(input, n_in, n_out, activation=activation, rng=rng)

        self.reset_layer()

        if W is not None:
            self.W = W
        if b is not None:
            self.b = b

        self.setup_outputs(input)

    def setup_outputs(self, input):
        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if self.activation is None
                else self.activation(lin_output))

    def get_outputs(self, input):
        self.setup_outputs(input)
        return self.output

class LogisticRegressionLayer(Layer):
    """Multi-class Logistic Regression Class
    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """
    def __init__(self, input, n_in, n_out, is_binary=False, threshold=0.5, rng=None):
        """ Initialize the parameters of the logistic regression
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the architecture
        (one minibatch)
        :type n_in: int
        :param n_in: number of input units, the dimension of the space in which
        the datapoints lie
        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie
        """
        self.activation = T.nnet.sigmoid
        super(LogisticRegressionLayer, self).__init__(input,
        n_in, n_out, self.activation, rng)

        self.reset_layer()

        self.threshold = threshold

        self.is_binary = is_binary
        if n_out == 1:
            self.is_binary = True
        # The number of classes seen
        self.n_classes_seen = numpy.zeros(n_out)
        # The number of wrong classification made for class i
        self.n_wrong_clasif_made = numpy.zeros(n_out)

        self.reset_conf_mat()
        #
        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = self.get_class_memberships(self.input)

        if not self.is_binary:
            # compute prediction as class whose probability is maximal in
            # symbolic form
            self.y_decision = T.argmax(self.p_y_given_x, axis=1)
        else:
            #If the probability is greater than 0.5 assign to the class 1
            # otherwise it is 0. Which can also be interpreted as check if
            # p(y=1|x)>threshold.
            self.y_decision = T.gt(T.flatten(self.p_y_given_x), threshold)

        # parameters of the model
        self.params = [self.W, self.b]

    def reset_conf_mat(self):
        """
        Reset the confusion matrix.
        """
        self.conf_mat = numpy.zeros(shape=(self.n_out, self.n_out), dtype=numpy.dtype(int))

    def negative_log_likelihood(self, y):
        """ Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::
            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                    \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """

        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        if self.is_binary:
            -T.mean(T.log(self.p_y_given_x))
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def crossentropy_categorical(self, y):
        """
        Find the categorical crossentropy.
        """
        return T.mean(T.nnet.categorical_crossentropy(self.p_y_given_x, y))

    def crossentropy(self, y):
        """
        use the theano nnet cross entropy function. Return the mean.
        Note: self.p_y_given_x is (batch_size, 1) but y is is (batch_size,)
        in order to establish the compliance, we should flatten the p_y_given_x.
        """
        return T.mean(T.nnet.binary_crossentropy(T.flatten(self.p_y_given_x), y))

    def get_class_memberships(self, x):
        lin_activation = T.dot(x, self.W) + self.b
        if self.is_binary:
            """If it is binary return the sigmoid."""
            return T.nnet.sigmoid(lin_activation)
        """
            Else return the softmax class memberships.
        """
        return T.nnet.softmax(lin_activation)

    def update_conf_mat(self, y, p_y_given_x):
        """
        Update the confusion matrix with the given true labels and estimated 
        labels.
        """
        if self.n_out == 1:
            y_decision = (p_y_given_x > 0.5)
            y_decision = y_decision.astype(int)
        else:
            y_decision = numpy.argmax(p_y_given_x, axis=1)
        for i in xrange(y.shape[0]):
            self.conf_mat[y[i]][y_decision[i]] +=1

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # check if y has same dimension of y_decision
        if y.ndim != self.y_decision.ndim:
            raise TypeError('y should have the same shape as self.y_decision',
                    ('y', y.type, 'y_decision', self.y_decision.type))
            # check if y is of the correct datatype
        if y.dtype.startswith('int') or y.dtype.startswith('uint'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_decision, y))
        else:
            raise NotImplementedError()

    def raw_prediction_errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # check if y has same dimension of y_decision
        if y.ndim != self.y_decision.ndim:
            raise TypeError('y should have the same shape as self.y_decision',
                    ('y', y.type, 'y_decision', self.y_decision.type))
            # check if y is of the correct datatype
        if y.dtype.startswith('int') or y.dtype.startswith('uint'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.neq(self.y_decision, y)
        else:
            raise NotImplementedError()

    def error_per_classes(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # check if y has same dimension of y_decision
        if y.ndim != self.y_decision.ndim:
            raise TypeError('y should have the same shape as self.y_decision',
                    ('y', y.type, 'y_decision', self.y_decision.type))
            # check if y is of the correct datatype
        if y.dtype.startswith('int') or y.dtype.startswith('uint'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            y_decision_res = T.neq(self.y_decision, y)
            for (i, y_decision_r) in enumerate(y_decision_res):
                self.n_classes_seen[y[i]] += 1
                if y_decision_r:
                    self.n_wrong_clasif_made[y[i]] += 1
            pred_per_class = self.n_wrong_clasif_made / self.n_classes_seen
            return T.mean(y_decision_res), pred_per_class
        else:
            raise NotImplementedError()
