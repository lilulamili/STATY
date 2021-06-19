# Wrapper for MLP Hyperparameter Tuning
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin


# Continuous response variable
class MLPWrapperCon_1Layer(BaseEstimator, RegressorMixin):
    def __init__(self, layer1 = 10, solver = "adam", max_iter = 200, activation = "relu", learning_rate_init = 0.001, alpha = 0.0001): #learning_rate = "constant", momentum = 0.9, alpha = 0.0001, epsilon = 1e-8):
        self.layer1 = layer1
        self.solver = solver
        self.max_iter = max_iter
        self.activation = activation
        self.learning_rate_init = learning_rate_init
        #self.learning_rate = learning_rate
        #self.momentum = momentum
        self.alpha = alpha
        #self.epsilon = epsilon

    def fit(self, X, y):
        model = MLPRegressor(
            hidden_layer_sizes=[self.layer1],
            solver = self.solver,
            max_iter = self.max_iter,
            activation = self.activation,
            learning_rate_init = self.learning_rate_init,
            #learning_rate = self.learning_rate,
            #momentum = self.momentum,
            alpha = self.alpha
            #epsilon = self.epsilon
        )
        model.fit(X, y)
        self.model = model
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

class MLPWrapperCon_2Layer(BaseEstimator, RegressorMixin):
    def __init__(self, layer1=10, layer2=10, solver = "adam", max_iter = 200, activation = "relu", learning_rate_init = 0.001, alpha = 0.0001): #learning_rate = "constant", momentum = 0.9, alpha = 0.0001, epsilon = 1e-8):
        self.layer1 = layer1
        self.layer2 = layer2
        self.solver = solver
        self.max_iter = max_iter
        self.activation = activation
        self.learning_rate_init = learning_rate_init
        #self.learning_rate = learning_rate
        #self.momentum = momentum
        self.alpha = alpha
        #self.epsilon = epsilon

    def fit(self, X, y):
        model = MLPRegressor(
            hidden_layer_sizes=[self.layer1, self.layer2],
            solver = self.solver,
            max_iter = self.max_iter,
            activation = self.activation,
            learning_rate_init = self.learning_rate_init,
            #learning_rate = self.learning_rate,
            #momentum = self.momentum,
            alpha = self.alpha
            #epsilon = self.epsilon
        )
        model.fit(X, y)
        self.model = model
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

class MLPWrapperCon_3Layer(BaseEstimator, RegressorMixin):
    def __init__(self, layer1=10, layer2=10, layer3=10, solver = "adam", max_iter = 200, activation = "relu", learning_rate_init = 0.001, alpha = 0.0001): #learning_rate = "constant", momentum = 0.9, alpha = 0.0001, epsilon = 1e-8):
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.solver = solver
        self.max_iter = max_iter
        self.activation = activation
        self.learning_rate_init = learning_rate_init
        #self.learning_rate = learning_rate
        #self.momentum = momentum
        self.alpha = alpha
        #self.epsilon = epsilon

    def fit(self, X, y):
        model = MLPRegressor(
            hidden_layer_sizes=[self.layer1, self.layer2, self.layer3],
            solver = self.solver,
            max_iter = self.max_iter,
            activation = self.activation,
            learning_rate_init = self.learning_rate_init,
            #learning_rate = self.learning_rate,
            #momentum = self.momentum,
            alpha = self.alpha
            #epsilon = self.epsilon
        )
        model.fit(X, y)
        self.model = model
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)



# Binary response variable
class MLPWrapperBin_1Layer(BaseEstimator, ClassifierMixin):
    def __init__(self, layer1=10, solver = "adam", max_iter = 200, activation = "relu", learning_rate_init = 0.001, alpha = 0.0001): #learning_rate = "constant", momentum = 0.9, alpha = 0.0001, epsilon = 1e-8):
        self.layer1 = layer1
        self.solver = solver
        self.max_iter = max_iter
        self.activation = activation
        self.learning_rate_init = learning_rate_init
        #self.learning_rate = learning_rate
        #self.momentum = momentum
        self.alpha = alpha
        #self.epsilon = epsilon

    def fit(self, X, y):
        model = MLPClassifier(
            hidden_layer_sizes=[self.layer1],
            solver = self.solver,
            max_iter = self.max_iter,
            activation = self.activation,
            learning_rate_init = self.learning_rate_init,
            #learning_rate = self.learning_rate,
            #momentum = self.momentum,
            alpha = self.alpha
            #epsilon = self.epsilon
        )
        model.fit(X, y)
        self.model = model
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        return self.model.score(X, y)

class MLPWrapperBin_2Layer(BaseEstimator, ClassifierMixin):
    def __init__(self, layer1=10, layer2=10, solver = "adam", max_iter = 200, activation = "relu", learning_rate_init = 0.001, alpha = 0.0001): #learning_rate = "constant", momentum = 0.9, alpha = 0.0001, epsilon = 1e-8):
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer1 = layer1
        self.solver = solver
        self.max_iter = max_iter
        self.activation = activation
        self.learning_rate_init = learning_rate_init
        #self.learning_rate = learning_rate
        #self.momentum = momentum
        self.alpha = alpha
        #self.epsilon = epsilon

    def fit(self, X, y):
        model = MLPClassifier(
            hidden_layer_sizes=[self.layer1, self.layer2],
            solver = self.solver,
            max_iter = self.max_iter,
            activation = self.activation,
            learning_rate_init = self.learning_rate_init,
            #learning_rate = self.learning_rate,
            #momentum = self.momentum,
            alpha = self.alpha
            #epsilon = self.epsilon
        )
        model.fit(X, y)
        self.model = model
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        return self.model.score(X, y)

class MLPWrapperBin_3Layer(BaseEstimator, ClassifierMixin):
    def __init__(self, layer1=10, layer2=10, layer3=10, solver = "adam", max_iter = 200, activation = "relu", learning_rate_init = 0.001, alpha = 0.0001): #learning_rate = "constant", momentum = 0.9, alpha = 0.0001, epsilon = 1e-8):
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer1 = layer1
        self.solver = solver
        self.max_iter = max_iter
        self.activation = activation
        self.learning_rate_init = learning_rate_init
        #self.learning_rate = learning_rate
        #self.momentum = momentum
        self.alpha = alpha
        #self.epsilon = epsilon

    def fit(self, X, y):
        model = MLPClassifier(
            hidden_layer_sizes=[self.layer1, self.layer2, self.layer3],
            solver = self.solver,
            max_iter = self.max_iter,
            activation = self.activation,
            learning_rate_init = self.learning_rate_init,
            #learning_rate = self.learning_rate,
            #momentum = self.momentum,
            alpha = self.alpha,
            #epsilon = self.epsilon
        )
        model.fit(X, y)
        self.model = model
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        return self.model.score(X, y)
