import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.manual_seed(42)

class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, predictions, targets):
        error = abs(predictions-targets)
        clipped_error = torch.clamp(error,0.0,1.0)
        linear_error = 2*(error-clipped_error)
        return torch.mean(torch.square(clipped_error)+linear_error)

class NeuralNet(nn.Module):

    def __init__(self,layers,forward):
        assert isinstance(layers,dict)
        super(NeuralNet,self).__init__()

        for layer in layers:
            setattr(self,layer,layers[layer])

        self.forward_pass = forward

    def forward(self,x):
        return self.forward_pass(self,x)
    
class mlp_1_0(nn.Module):

    def __init__(self):
        super(mlp_1_0,self).__init__()
        self.fc1=nn.Linear(5,3000)
        self.fc1_actv_fn=nn.LeakyReLU()
        self.fc2=nn.Linear(3000,3000)
        self.fc2_actv_fn=nn.LeakyReLU()
        self.fc3=nn.Linear(3000,300)
        self.fc3_actv_fn=nn.LeakyReLU()
        self.out=nn.Linear(300,3)

    def forward(self,x):
        if ~isinstance(x,torch.Tensor):
            if isinstance(x,(float,list,tuple)):
                x = torch.tensor(x)
        x = self.fc1_actv_fn(self.fc1(x))
        x = self.fc2_actv_fn(self.fc2(x))
        x = self.fc3_actv_fn(self.fc3(x))
        x = self.out(x)
        return x