import torch
import torch.nn as nn

from torch.autograd import Variable

class NeuralNet(nn.Module):

    def __init__(self,layers,forward):
        assert isinstance(layers,dict)
        super(NeuralNet,self).__init__()

        for layer in layers:
            setattr(self,layer,layers[layer])

        self.forward_pass = forward

    def forward(self,x):
        return self.forward_pass(self,x)
    
mlp_1_0_layers = {
    "fc1":nn.Linear(5,5),
    "fc1_actv_fn":nn.LeakyReLU(),
    "fc2":nn.Linear(5,3),
    "fc2_actv_fn":nn.LeakyReLU(),
    "out":nn.Linear(3,3),
    "out_actv_fn":nn.LeakyReLU(),
    "softmax":nn.Softmax()
}
def mlp_1_0_forward(neural_net,x):
    if ~isinstance(x,torch.Tensor):
        if isinstance(x,(float,list,tuple)):
            x = torch.tensor(x)
    x = neural_net.fc1(x)
    x = neural_net.fc1_actv_fn(x)
    x = neural_net.fc2(x)
    x = neural_net.fc2_actv_fn(x)
    x = neural_net.out(x)
    x = neural_net.out_actv_fn(x)
    x = neural_net.softmax(x)
    return x
mlp_1_0 = NeuralNet(mlp_1_0_layers,mlp_1_0_forward)