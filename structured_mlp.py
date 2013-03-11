from __future__ import division

import numpy
import theano

from theano import tensor as T

from collections import OrderedDict

import cPickle as pkl

from arcade_experiments.pretrained_mlp.structured_mlp.utils import as_floatX, safe_update, shared_dataset

from arcade_experiments.monitor import Monitor
from arcade_experiments.pretrained_mlp.structured_mlp.layer import LogisticRegressionLayer, HiddenLayer, PretrainingLayers
from arcade_experiments.pretrained_mlp.structured_mlp.dataset import *


DEBUGGING = False

class Costs:
    Crossentropy = "crossentropy"
    NegativeLikelihood = "negativelikelihood"


class NeuralActivations:
    Tanh = "tanh"
    Logistic = "sigmoid"
    Rectifier = "rectifier"


class StructuredMLP(object):

    """
    Multi-Layer Perceptron Class with multiple hidden layers. This class is
    used for pretraining the second phase neural network. Intermediate layers
    have tanh activation function or the sigmoid function (defined here by a
    ``SigmoidalLayer`` class)  while the top layer is a softmax layer (defined
    here by a ``LogisticRegression`` class).
    """
    def __init__(self,
            input,
            in_layer_shape,
            layer2_in = 1000,
            n_out = 11,
            use_adagrad = True,
            patch_size=64,
            activation=NeuralActivations.Rectifier,
            layer1_nout=11,
            exp_id=1,
            quiet=False,
            n_classes=11,
            save_file=None,
            mem_alloc="CPU",
            momentum=1.,
            enable_standardization=False,
            rng=None):
        """
        Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType

        :param input: symbolic variable that describes the input of the architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in which the datapoints lie.

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type in_layer_shape: list
        :param in_layer_shape: the shape of the first layer - format is :
            (no of patches, no of pixels per patch, no of batches, number of
            hidden units for locally connected hidden layer 1)

        :type layer2_in: list
        :param layer2_in: No of hidden units in the second hidden layer.

        :type shared_weights: use shared weights across the image
        :param shared_weights: boolean parameter to enable/disable the usage of
        shared weights

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in which
        the labels lie.
        """

        self.input = input

        if rng == None:
            rng = numpy.random.RandomState(1234)

        self.monitor = Monitor()

        self.learning_rate = 0.001

        self.exp_id = exp_id

        self.y = T.bvector('y')  # the labels are presented as 1D vector of int32
        self.mem_alloc = mem_alloc

        self.rng = rng
        self.ds = Dataset()
        self.n_out = n_out
        self.momentum = momentum
        self.save_file = save_file

        self.in_layer_shape = in_layer_shape
        self.layer2_in = layer2_in
        self.patch_size = patch_size
        self.layer1_nout = layer1_nout

        self.locally_connected_layer = None
        self.fully_connected_layer = None
        self.activation = activation
        self.n_hiddens_layer2 = (self.layer1_nout * in_layer_shape[0], layer2_in)
        self.n_classes = n_classes
        self.state = "train"
        self.enable_standardization = enable_standardization

        self.out_dir = "out/"
        self.grads = []
        self.test_scores = []

        #Whether to turn on or off the messages.
        self.quiet = quiet
        self.test_set_x = None
        self.valid_set_x = None
        self.test_set_y = None
        self.valid_set_y = None

        self.setup_hidden_layers(activation, in_layer_shape,
        self.n_hiddens_layer2, n_out)

        self.use_adagrad = use_adagrad

        #Error for patches with object in it:
        self.obj_patch_error_percent = 0

    def set_validation_data(self, data, labels, patch_mode=True):
        if patch_mode:
            self.valid_data = get_dataset_patches(data)
        else:
            self.valid_data = data
        self.valid_labels = labels
        self.valid_set_x = shared_dataset(self.valid_data, name="valid_set_x", mode=self.mem_alloc)
        self.valid_set_y = shared_dataset(labels, name="valid_labels", mode=self.mem_alloc)
        self.valid_set_y = T.cast(self.valid_set_y, "int8")

    def set_test_data(self, data, labels, patch_mode=True):
        if patch_mode:
            self.test_data = get_dataset_patches(data)
        else:
            self.test_data = data
        self.test_labels = labels
        self.test_set_x = shared_dataset(self.test_data, name="test_set_x", mode=self.mem_alloc)
        self.test_set_y = shared_dataset(labels, name="test_labels",  mode=self.mem_alloc)
        self.test_set_y = T.cast(self.test_set_y, "int8")

    def setup_hidden_layers(self,
        activation,
        in_layer_shape,
        n_hiddens_layer2,
        n_out):
        """
        Setup the hidden layers with the specified number of hidden units.
        """
        act_fn = T.tanh
        if activation == NeuralActivations.Rectifier:
            act_fn = self.rectifier_act

        if not in_layer_shape:
            in_layer_shape = self.in_layer_shape
        else:
            self.in_layer_shape = in_layer_shape

        if n_hiddens_layer2 == 0:
            n_hiddens_layer2 = n_hiddens_layer2

        if n_out == 0:
            n_out = self.n_out

        ml_in = T.fmatrix("mlp_input")
        self.locally_connected_layer = PretrainingLayers(ml_in,
            activation=act_fn,
            out_activation=act_fn,
            layer_shape=self.in_layer_shape,
            n_classes=self.layer1_nout,
            rng=self.rng)

        input = self.input
        input = input.dimshuffle(1, 0, 2)

        in_patches = [input[i] for i in xrange(self.patch_size)]

        #for p_i in in_patches:
        self.locally_connected_layer.set_input(in_patches[0])

        h1_repr = [self.locally_connected_layer.fprop(p_i) for p_i in in_patches]
        l_connected_out = T.concatenate(h1_repr, axis=1)

        if self.enable_standardization:
            std_val = T.std(l_connected_out, axis=0)
            z_val = (l_connected_out - T.mean(l_connected_out, axis=0)) / std_val
            l_connected_out = T.switch(T.eq(std_val, 0), l_connected_out, z_val)

        self.fully_connected_layer = HiddenLayer(rng=self.rng,
                                       input=l_connected_out,
                                       n_in=n_hiddens_layer2[0],
                                       n_out=n_hiddens_layer2[1],
                                       activation=act_fn)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegressionLayer(
                input=self.fully_connected_layer.output,
                n_in=n_hiddens_layer2[1],
                n_out=n_out,
                rng=self.rng)

        self.initialize_regularization()

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood

        # negative log likelihood of the MLP is given by the
        # crossentropy of the output of the model, computed in the
        # logistic regression layer
        self.crossentropy = self.logRegressionLayer.crossentropy
        self.crossentropy_categorical = self.logRegressionLayer.crossentropy_categorical

        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        self.raw_prediction_errors =\
        self.logRegressionLayer.raw_prediction_errors

        self.p_y_given_x = self.logRegressionLayer.p_y_given_x

        hidden_outputs =\
            self.fully_connected_layer.get_outputs(l_connected_out)

        self.class_memberships = self.logRegressionLayer.get_class_memberships(hidden_outputs)
        self.initialize_params()
        self.cost = None
        self.updates = None

    def initialize_params(self):
        # the parameters of the model are the parameters of the all hidden
        # and logistic regression layers it is made out of
        self.params = self.locally_connected_layer.params
        self.params += self.fully_connected_layer.params
        self.params += self.logRegressionLayer.params

    def initialize_regularization(self):
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be smalla
        self.L1 = abs(self.fully_connected_layer.W).sum() +\
            abs(self.locally_connected_layer.W).sum()
        self.L1 += abs(self.logRegressionLayer.W).sum()

        # square of L2 norm;
        # one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.fully_connected_layer.W ** 2).sum() +\
            abs(self.locally_connected_layer.W ** 2).sum()
        self.L2_sqr += (self.logRegressionLayer.W ** 2).sum()

    def rectifier_act(self, x, mask=False):
        """
        Activation function for rectifier hidden units. RELU, rectifier
        linear units.
        """
        if mask:
            activation = numpy.asarray(numpy.sign(numpy.random.uniform(low=-1,
                high=1, size=(10,))), dtype=theano.config.floatX) * T.maximum(x, 0)
        else:
            activation = T.maximum(x, 0)
        return activation

    def sgd_updates(self,
                    cost,
                    learning_rate):
        """
        Using this function, specify how to update the parameters of the model as a dictionary
        """
        updates = OrderedDict({})
        if self.use_adagrad:
                updates = self.sgd_updates_adagrad(cost, learning_rate)
        else:
                self.grads = T.grad(cost, self.params)
                # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
                # same length, zip generates a list C of same size, where each element
                # is a pair formed from the two lists :
                #    C = [(a1, b1), (a2, b2), (a3, b3) , (a4, b4)]
                for param, grad in zip(self.params, self.grads):
                    updates[param] = self.momentum * param - learning_rate * grad
        return updates

    def sgd_updates_adagrad(self,
                    cost,
                    learning_rate):
        """
        Return the dictionary of parameter specific learning rate updates using adagrad algorithm.
        """
        #Initialize the variables
        accumulators = OrderedDict({})
        e0s = OrderedDict({})
        learn_rates = []
        ups = OrderedDict({})

        #initialize the accumulator and the epsilon_0
        for param in self.params:
                accumulators[param] = theano.shared(value=as_floatX(0.), name="acc_%s" % param.name)
                e0s[param] = as_floatX(learning_rate)

        self.grads = [T.grad(cost, p) for p in self.params]

        #Compute the learning rates
        for param, gp in zip(self.params, self.grads):
                acc = accumulators[param]
                ups[acc] = T.sqrt((gp ** 2).sum())
                learn_rates.append(e0s[param] / ups[acc])

        #Find the updates based on the parameters
        updates = [(p, p - step * gp) for (step, p, gp) in zip(learn_rates,
        self.params, self.grads)]
        p_up = dict(updates)
        safe_update(ups, p_up)
        return ups

    def dropout(self, prob=0.5):
        """
        Randomly dropout the hidden units wrt given probability as a binomial
        prob.
        """
        for param in self.params:
            if param.name == "W":
                val = param.get_value(borrow=True)
                #throw a coin for weight:
                dropouts = numpy.random.binomial(1, prob, (val.shape[0], val.shape[1]))
                new_param = param * dropouts
                param.set_value(val)

    def get_cost_function(self,
        cost_type,
        y,
        L1_reg,
        L2_reg):

        if cost_type == Costs.Crossentropy:
            if self.n_out == 1:
                cost = self.crossentropy(y)
                if L1_reg is not None and L2_reg is not None:
                    cost += L1_reg * self.L1 \
                        + L2_reg * self.L2_sqr
            else:
                cost = self.crossentropy_categorical(y)
                if L1_reg is not None and L2_reg is not None:
                    cost += L1_reg * self.L1 \
                        + L2_reg * self.L2_sqr
        elif cost_type == Costs.NegativeLikelihood:
            cost = self.negative_log_likelihood(y)
            if L1_reg is not None and L2_reg is not None:
                    cost += L1_reg * self.L1 \
                    + L2_reg * self.L2_sqr
        return cost

    def reset_learning(self):
        self.setup_hidden_layers(self.activation, self.in_layer_shape, self.n_hiddens_layer2, self.n_out)

    def train(self,
              data=None,
              labels=None,
              **kwargs):

        randomize_batches = False

        if "randomize_mb" in kwargs:
            if kwargs["randomize_mb"]:
                randomize_batches = True

        if "L1_reg" in kwargs:
            L1_reg = kwargs["L1_reg"]
        else:
            L1_reg = None

        if "L2_reg" in kwargs:
           L2_reg = kwargs["L2_reg"]
        else:
            L2_reg = None

        learning_rate = kwargs["learning_rate"]
        n_epochs = kwargs["nepochs"]
        cost_type = kwargs["cost_type"]
        save_exp_data = kwargs["save_exp_data"]
        batch_size = kwargs["batch_size"]
        enable_dropout = kwargs["enable_dropout"]
        batch_size = self.in_layer_shape[2]

        test_args = {
            "batch_size": None,
            "test_mode": None
        }

        if data is None:
            raise Exception("Post-training can't start without pretraining class membership probabilities.")

        if labels is None:
            raise Exception("Post-training can not start without posttraining class labels.")

        data = numpy.asarray(data.tolist(), dtype=theano.config.floatX)
        labels = numpy.asarray(labels.tolist(), dtype="int8")
        self.state = "train"

        self.learning_rate = learning_rate

        train_set_x = shared_dataset(data, name="training_set_x", mode=self.mem_alloc)
        train_set_y = shared_dataset(labels, name="labels", mode=self.mem_alloc)
        train_set_y = T.cast(train_set_y, "int8")

        # compute number of minibatches for training
        n_examples = data.shape[0]
        n_train_batches = int(math.ceil(n_examples / batch_size))

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '...postraining the model'
        # allocate symbolic variables for the data
        index = T.lscalar('index')    # index to a [mini]batch

        mode = "FAST_RUN"
        if DEBUGGING:
            index.tag.test_value = 0
            self.y.tag.test_value = numpy.ones(n_examples)
            mode = "DEBUG_MODE"

        # the cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed
        # here symbolically.

        if self.cost is None:
            self.cost = self.get_cost_function(cost_type, self.y, L1_reg, L2_reg)

        if self.updates is None:
            self.updates = self.sgd_updates(self.cost, learning_rate)

        # compiling a Theano function `train_model` that returns the cost, butx
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`
        # p_y_given_x = self.class_memberships
        train_model = theano.function(inputs=[index],
            outputs=self.cost,
            updates=self.updates,
            givens = {
                self.input: train_set_x[index * batch_size:(index + 1) * batch_size],
                self.y: train_set_y[index * batch_size: (index + 1) * batch_size]
            },
            mode=mode)

        if DEBUGGING:
            theano.printing.debugprint(train_model)

        epoch = 0
        costs = []
        Ws = []

        while (epoch < n_epochs):
            cost_per_epoch = []
            print "In the epoch %d" % (epoch)
            minibatches = xrange(n_train_batches)
            if randomize_batches:
                minibatches = self.rng.permutation(n_train_batches)

            for minibatch_index in minibatches:
                if not self.quiet:
                    print "Postraining in Minibatch %i " % (minibatch_index)
                minibatch_avg_cost = train_model(minibatch_index)
                if enable_dropout:
                    self.dropout()

                cost_per_epoch.append(float(minibatch_avg_cost))
                costs.append(float(minibatch_avg_cost))

            print "Loss on the training dataset is %f" % (numpy.mean(cost_per_epoch))
            print "Testing on the training dataset."
            test_args["batch_size"] = batch_size
            test_args["test_mode"] = "train"
            self.test(train_set_x, train_set_y, numpy.mean(cost_per_epoch), **test_args)

            print "Testing on the test dataset."
            test_args["batch_size"] = self.test_set_x.get_value().shape[0]
            test_args["test_mode"] = "test"
            self.test(None, None, **test_args)

            epoch +=1
            print "Loss is ", numpy.mean(costs)
            self.monitor.newEpoch(batch_size)
            self.dump_contents()

        return costs

    def test(self,
             data=None,
             labels=None,
             training_loss=None,
             **kwargs):

        batch_size = kwargs["batch_size"]
        test_mode = kwargs["test_mode"]

        if data is not None and labels is not None:
            test_set_x = data
            test_set_y = labels
        else:
            test_set_x = self.test_set_x
            test_set_y = self.test_set_y

        self.state = "test"

        # compute number of minibatches for training, validation and testing
        n_examples = test_set_x.get_value().shape[0]
        n_test_batches = int(math.ceil(n_examples / batch_size))

        print '...post-testing the model'

        # allocate symbolic variables for the data
        index = T.lscalar()    # index to a [mini]batch7

        mode = "FAST_RUN"
        if DEBUGGING:
            theano.config.compute_test_value = 'raise'
            index.tag.test_value = 0
            self.y.tag.test_value = numpy.ones(n_examples)
            mode = "DEBUG_MODE"

        # the cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed
        # here symbolically

        # compiling a Theano function `test_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`
        p_y_given_x = self.class_memberships

        test_model = theano.function(inputs=[index],
            outputs=[self.errors(self.y), p_y_given_x],
            givens={
                self.input: test_set_x[index * batch_size:(index + 1) * batch_size],
                self.y: test_set_y[index * batch_size: (index + 1) * batch_size]},
            mode=mode)

        ###############
        # TEST MODEL  #
        ###############
        test_losses = []

        for minibatch_index in xrange(n_test_batches):
            errors, y_pre = test_model(minibatch_index)
            targets = test_set_y.eval()[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            training_costs = errors
            if test_mode == "train":
                if training_loss:
                    training_costs = training_loss
                self.monitor.add_train_exps(training_costs, batch_size, targets, y_pre)
            elif test_mode == "test":
                self.monitor.add_test_exps(errors, batch_size, targets, y_pre)
            else:
                raise NameError("Unknown test mode!")
            test_score = numpy.mean(errors)
        print "The error on %s dataset is %f" % (test_mode, test_score)
        return test_score, test_losses

    def dump_contents(self):
        output = open(self.save_file, "wb")
        pkl.dump([self.params, self.monitor, self.rng], output, 2)
        output.close()

    def load(self):
        input = open(self.save_file, "rb")
        tmp_dict = pkl.load(input)
        self.__dict__.update(tmp_dict)

    def __getstate__(self):
        state = dict(self.__dict__)
        return state

    def __setstate__(self, d):
        self.__dict__.update(d)
