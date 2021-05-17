''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1_real = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_1_phase = nn.Conv1d(d_in, d_hid, 1) # position-wise
        

        self.w_2_real = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.w_2_phase = nn.Conv1d(d_hid, d_in, 1) # position-wise
        
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_real,x_phase):
        residual_real = x_real
        residual_phase = x_phase
        cnn_real = x_real.transpose(1, 2)
        cnn_phase = x_phase.transpose(1, 2)
        

        w1_real = F.relu(self.w_1_real(cnn_real)-self.w_1_phase(cnn_phase))
        w1_phase = F.relu(self.w_1_real(cnn_phase)+self.w_1_phase(cnn_real))


        output_real = self.w_2_real(w1_real)-self.w_2_phase(w1_phase)
        output_phase = self.w_2_real(w1_phase)+self.w_2_phase(w1_real)

        
        output_real = output_real.transpose(1, 2)
        output_phase = output_phase.transpose(1, 2)
        
        output_real = self.dropout(output_real)
        output_phase = self.dropout(output_phase)
        

        output_real = self.layer_norm(output_real + residual_real)
        output_phase = self.layer_norm(output_phase + residual_phase)
        return output_real,output_phase
