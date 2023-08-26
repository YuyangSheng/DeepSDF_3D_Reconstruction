import torch
import torch.nn as nn
import torch.nn.functional as F


class AD_SDF(nn.Module):
    def __init__(self, z_dim=256, dropout_prob=0.2):

        super(AD_SDF, self).__init__()

        self.dropout_prob = dropout_prob

        self.decoder_stage1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(z_dim+3, 512), name='weight'),
            nn.ReLU(True),
            nn.Dropout(self.dropout_prob),

            nn.utils.weight_norm(nn.Linear(512, 512), name='weight'),
            nn.ReLU(True),
            nn.Dropout(self.dropout_prob),

            nn.utils.weight_norm(nn.Linear(512, 512), name='weight'),
            nn.ReLU(True),
            nn.Dropout(self.dropout_prob),

            nn.utils.weight_norm(nn.Linear(512, 509-z_dim), name='weight'),
            nn.ReLU(True),
            nn.Dropout(self.dropout_prob))
        
        self.decoder_stage2 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(512, 512), name='weight'),
            nn.ReLU(True),
            nn.Dropout(self.dropout_prob),

            nn.utils.weight_norm(nn.Linear(512, 512), name='weight'),
            nn.ReLU(True),
            nn.Dropout(self.dropout_prob),

            nn.utils.weight_norm(nn.Linear(512, 512), name='weight'),
            nn.ReLU(True),
            nn.Dropout(self.dropout_prob),

            nn.utils.weight_norm(nn.Linear(512, 1), name='weight'),
            nn.Tanh(),
            nn.Dropout(self.dropout_prob))
    
    def forward(self, inputs):
        # TODO: should the latent code for each point be the same?
        # self.latent_vectors: N * z_dim
        # xyz: N * 3
        # inputs: N * (z_dim+3)
        # define outside the model: backward
        # define inside the model: backward when inference

        output_stage1 = self.decoder_stage1(inputs)

        input_stage2 = torch.cat((output_stage1, inputs), dim=1)
        output_stage2 = self.decoder_stage2(input_stage2)

        return output_stage2
