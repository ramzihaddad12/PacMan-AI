import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"

        score = nn.DotProduct(x, self.w)
        return score

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        score = self.run(x)
        scalarScore = nn.as_scalar(score)
        return -1 if scalarScore < 0 else 1 

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        stop = False
        while not stop:
            stop = True
            for x,y in dataset.iterate_once(1):
                prediction = self.get_prediction(x)
                trueLabel = nn.as_scalar(y)
                if trueLabel != prediction:
                    stop = False
                    nn.Parameter.update(self.w, x, trueLabel)

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        size = 100

        self.weight1 = nn.Parameter(1, size)
        self.weight2 = nn.Parameter(size, 1)
        self.bias1 = nn.Parameter(1, size)
        self.bias2 = nn.Parameter(1, 1)
        self.batch_size = 1
        self.learningRate = 0.01


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        return nn.AddBias(nn.Linear(nn.ReLU(nn.AddBias(nn.Linear(x, self.weight1), self.bias1)), self.weight2), self.bias2)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        
        maxLoss = 0.02

        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                gradient = nn.gradients(self.get_loss(x,y), [self.weight1, self.bias1, self.bias2, self.weight2])
                self.weight1.update(gradient[0], -1 * self.learningRate)
                self.bias1.update(gradient[1], -1 * self.learningRate)
                self.bias2.update(gradient[2], -1 * self.learningRate)
                self.weight2.update(gradient[3], -1 * self.learningRate)

            if maxLoss <= nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))):
                continue
            else:
                return

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        inputSize = 784
        numOfLabels = 10
        labelSize = numOfLabels * numOfLabels

        self.weight1 = nn.Parameter(inputSize, labelSize)
        self.weight2 = nn.Parameter(labelSize, numOfLabels)
        self.bias1 = nn.Parameter(1, labelSize)
        self.bias2 = nn.Parameter(1, numOfLabels)
        self.batch_size = 1
        self.learningRate = 0.01

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        return nn.AddBias(nn.Linear(nn.ReLU(nn.AddBias(nn.Linear(x, self.weight1), self.bias1)), self.weight2), self.bias2)


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        minAccuracy = 0.97

        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                gradient = nn.gradients(self.get_loss(x,y), [self.weight1, self.bias1, self.bias2, self.weight2])
                self.weight1.update(gradient[0], -1 * self.learningRate)
                self.bias1.update(gradient[1], -1 * self.learningRate)
                self.bias2.update(gradient[2], -1 * self.learningRate)
                self.weight2.update(gradient[3], -1 * self.learningRate)

            if minAccuracy >= dataset.get_validation_accuracy():
                continue
            else:
                return

class DeepQModel(object):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        "*** YOUR CODE HERE ***"
        length = 750
        self.weight1 = nn.Parameter(state_dim, length)
        self.weight2 = nn.Parameter(length, 100)
        self.weight3 = nn.Parameter(100, action_dim)
        self.bias1 = nn.Parameter(1, length)
        self.bias2 = nn.Parameter(1, 100)
        self.bias3 = nn.Parameter(1, action_dim)
        self.batch_size = 50
        self.learning_rate = -0.2
        self.numTrainingGames = 4500

        self.parameters = [self.weight1, self.bias1, self.weight2, self.bias2, self.weight3, self.bias3]


    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a node with shape (batch_size x state_dim)
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(states), Q_target)


    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a node with shape (batch_size x state_dim)
        Output:
            result: a node with shape (batch_size x num_actions) containing Q-value
                scores for each of the actions
        """
        "*** YOUR CODE HERE ***"

        return nn.AddBias(nn.Linear(nn.ReLU(nn.AddBias(nn.Linear(nn.ReLU(nn.AddBias(nn.Linear(states, self.weight1), self.bias1)), self.weight2), self.bias2)), self.weight3), self.bias3)


    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a node with shape (batch_size x state_dim)
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        "*** YOUR CODE HERE ***"

        gradient = nn.gradients(self.get_loss(states,Q_target), self.parameters)

        self.weight1.update(gradient[0],self.learning_rate)
        self.bias1.update(gradient[1], self.learning_rate)
        self.weight2.update(gradient[2], self.learning_rate)
        self.bias2.update(gradient[3], self.learning_rate)
        self.weight3.update(gradient[4], self.learning_rate)
        self.bias3.update(gradient[5], self.learning_rate)