import torch.nn as nn
import torch.nn.functional as F
__all__ = ['mlp']

# 定义模型
class MLP(nn.Module):
    def __init__(self, input_shape = 784, output_shape = 10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)  
        self.fc2 = nn.Linear(128, 64)  
        self.fc3 = nn.Linear(64, output_shape)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)  
        x = self.fc2(x)
        # x = F.relu(x)    
        x = self.fc3(x)
        # x = F.relu(x)  
        # x = self.fc4(x)
        # # x = F.relu(x)  
        # x = self.fc5(x)
        return x
    
def mlp(**kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = MLP(**kwargs)
    return model
