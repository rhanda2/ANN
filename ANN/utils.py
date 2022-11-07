import numpy as np
import math

"""
    GT = Ground Truth
    Pred = Current predictions
"""


class MSELoss:      # For Reference
    def __init__(self):
        # Buffers to store intermediate results.
        self.current_prediction = None
        self.current_gt = None
        pass

    def __call__(self, y_pred, y_gt):
        self.current_prediction = y_pred
        self.current_gt = y_gt

        # MSE = 0.5 x (GT - Prediction)^2
        loss = 0.5 * np.power((y_gt - y_pred), 2)
        return loss

    def grad(self):
        # Derived by calculating dL/dy_pred
        gradient = -1 * (self.current_gt - self.current_prediction)

        # We are creating and emptying buffers to emulate computation graphs in
        # Modern ML frameworks such as Tensorflow and Pytorch. It is not required.
        self.current_prediction = None
        self.current_gt = None

        return gradient


class CrossEntropyLoss:     # TODO: Make this work!!!
    def __init__(self):
        # Buffers to store intermediate results.
        self.current_prediction = None
        self.current_gt = None
        pass

    def __call__(self, y_pred, y_gt):
        # TODO: Calculate Loss Function
        self.current_prediction = y_pred
        self.current_gt = y_gt
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -y_gt * np.log(y_pred_clipped) - (1 - y_gt) * np.log(1 - y_pred_clipped) # added in binomial not multinomial
        return loss

    def grad(self):
        # TODO: Calculate Gradients for back propagation
        y_pred = self.current_prediction
        y_gt = self.current_gt
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        gradient = -(y_gt / y_pred_clipped) + (1 - y_gt) / (1 - y_pred_clipped)
        return gradient


class SoftmaxActivation:    # TODO: Make this work!!!
    def __init__(self):
        self.z = None
        self.y = None
        pass

    def __call__(self, z):
        # TODO: Calculate Activation Function
        self.z = z
        z = np.clip(z, 1e-15, 1 - 1e-15)
        sums = np.sum(np.exp(z), axis=1)
        y = np.exp(z)/np.sum(np.exp(z))
        for i in range(z.shape[0]):
            y[i] = np.exp(z[i])/sums[i]
        self.y = y
        return y

    def grad(self):
        # TODO: Calculate Gradients.. Remember this is calculated w.r.t. input to the function -> dy/dz
        s_y = self.y
        return s_y * (1 - s_y)


class SigmoidActivation:    # TODO: Make this work!!!
    def __init__(self):
        self.z = None

    def __call__(self, z):
        # TODO: Calculate Activation Function
        self.z = z 
        # z = np.clip(z, 1e-15, 1 - 1e-15)
        y =  1/(1 + np.exp(-1*z))
        return y

    def grad(self):
        # TODO: Calculate Gradients.. Remember this is calculated w.r.t. input to the function -> dy/dz
        f_x = self(self.z)
        f_dash = f_x*(1 - f_x)
        return f_dash


class ReLUActivation:
    def __init__(self):
        self.z = None
        pass

    def __call__(self, z):
        # y = f(z) = max(z, 0) -> Refer to the computational model of an Artificial Neuron
        self.z = z
        y = np.maximum(z, 0)
        return y

    def __grad__(self):
        # dy/dz = 1 if z was > 0 or dy/dz = 0 if z was <= 0
        gradient = np.where(self.z > 0, 1, 0)
        return gradient


def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)
    accuracy = np.sum(y_true == y_pred) / y_true.shape[0]
    return accuracy

if __name__ == '__main__':
    Sig = SigmoidActivation()
    print(Sig(np.array([[1,1],[1,1]])))