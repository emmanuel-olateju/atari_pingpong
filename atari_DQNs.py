import torch
import torch.nn as nn

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
        self.fc1=nn.Linear(5,500)
        self.fc1_actv_fn=nn.LeakyReLU()
        self.fc2=nn.Linear(500,300)
        self.fc2_actv_fn=nn.LeakyReLU()
        self.fc3=nn.Linear(300,150)
        self.fc3_actv_fn=nn.LeakyReLU()
        self.fc4=nn.Linear(150,75)
        self.fc4_actv_fn=nn.LeakyReLU()
        self.fc5=nn.Linear(75,25)
        self.fc5_actv_fn=nn.LeakyReLU()
        self.out=nn.Linear(25,3)
        self.softmax=nn.Softmax()

    def forward(self,x):
        if ~isinstance(x,torch.Tensor):
            if isinstance(x,(float,list,tuple)):
                x = torch.tensor(x)
        x = self.fc1_actv_fn(self.fc1(x))
        x = self.fc2_actv_fn(self.fc2(x))
        x = self.fc3_actv_fn(self.fc3(x))
        x = self.fc4_actv_fn(self.fc4(x))
        x = self.fc5_actv_fn(self.fc5(x))
        x = self.softmax(self.out(x))
        return x

class mlp_1_1(nn.Module):

    def __init__(self):
        super(mlp_1_1,self).__init__()
        self.fc1=nn.Linear(5,1000)
        self.fc1_actv_fn=nn.LeakyReLU()
        self.fc2=nn.Linear(1000,500)
        self.fc2_actv_fn=nn.LeakyReLU()
        self.fc3=nn.Linear(500,300)
        self.fc3_actv_fn=nn.LeakyReLU()
        self.fc4=nn.Linear(300,150)
        self.fc4_actv_fn=nn.LeakyReLU()
        self.fc5=nn.Linear(150,75)
        self.fc5_actv_fn=nn.LeakyReLU()
        self.out=nn.Linear(75,3)
        self.softmax=nn.Softmax()

    def forward(self,x):
        if ~isinstance(x,torch.Tensor):
            if isinstance(x,(float,list,tuple)):
                x = torch.tensor(x)
        x = self.fc1_actv_fn(self.fc1(x))
        x = self.fc2_actv_fn(self.fc2(x))
        x = self.fc3_actv_fn(self.fc3(x))
        x = self.fc4_actv_fn(self.fc4(x))
        x = self.fc5_actv_fn(self.fc5(x))
        x = self.softmax(self.out(x))
        return x
    
class mlp_1_2(nn.Module):

    def __init__(self):
        super(mlp_1_2,self).__init__()
        self.fc1=nn.Linear(5,1000)
        self.fc1_actv_fn=nn.LeakyReLU()
        self.fc2=nn.Linear(1000,500)
        self.fc2_actv_fn=nn.LeakyReLU()
        self.fc3=nn.Linear(500,300)
        self.fc3_actv_fn=nn.LeakyReLU()
        self.fc4=nn.Linear(300,150)
        self.fc4_actv_fn=nn.LeakyReLU()
        self.fc5=nn.Linear(150,75)
        self.fc5_actv_fn=nn.LeakyReLU()
        self.fc6=nn.Linear(75,38)
        self.fc6_actv_fn=nn.LeakyReLU()
        self.fc7=nn.Linear(38,19)
        self.fc7_actv_fn=nn.LeakyReLU()
        self.fc8=nn.Linear(19,9)
        self.fc8_actv_fn=nn.LeakyReLU()
        self.fc9=nn.Linear(9,6)
        self.fc9_actv_fn=nn.LeakyReLU()
        self.out=nn.Linear(6,3)
        self.softmax=nn.Softmax()

    def forward(self,x):
        if ~isinstance(x,torch.Tensor):
            if isinstance(x,(float,list,tuple)):
                x = torch.tensor(x)
        x = self.fc1_actv_fn(self.fc1(x))
        x = self.fc2_actv_fn(self.fc2(x))
        x = self.fc3_actv_fn(self.fc3(x))
        x = self.fc4_actv_fn(self.fc4(x))
        x = self.fc5_actv_fn(self.fc5(x))
        x = self.fc6_actv_fn(self.fc6(x))
        x = self.fc7_actv_fn(self.fc7(x))
        x = self.fc8_actv_fn(self.fc8(x))
        x = self.fc9_actv_fn(self.fc9(x))
        x = self.softmax(self.out(x))
        return x
    
# mlp_1_0_layers = {
#     "fc1":nn.Linear(5,5),
#     "fc1_actv_fn":nn.LeakyReLU(),
#     "fc2":nn.Linear(5,3),
#     "fc2_actv_fn":nn.LeakyReLU(),
#     "out":nn.Linear(3,3),
#     "out_actv_fn":nn.LeakyReLU(),
#     "softmax":nn.Softmax()
# }
# def mlp_1_0_forward(neural_net,x):
#     if ~isinstance(x,torch.Tensor):
#         if isinstance(x,(float,list,tuple)):
#             x = torch.tensor(x)
#     x = neural_net.fc1(x)
#     x = neural_net.fc1_actv_fn(x)
#     x = neural_net.fc2(x)
#     x = neural_net.fc2_actv_fn(x)
#     x = neural_net.out(x)
#     x = neural_net.out_actv_fn(x)
#     x = neural_net.softmax(x)
#     return x
# mlp_1_0 = NeuralNet(mlp_1_0_layers,mlp_1_0_forward)