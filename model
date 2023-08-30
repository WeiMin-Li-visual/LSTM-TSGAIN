import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM

class Orange(nn.Module):
    def __init__(self, features_in, features_out, seq_num, pred_length, device, Dim, run_unit, view_num, latDim, hidden):
        super(Orange, self).__init__()
        self.features_in = features_in
        self.features_out = features_out
        self.pred_length = pred_length
        self.run_unit = run_unit
        self.view_num = view_num
        ...

        self.Generator = Generator(self.Dim, self.H_Dim1, self.H_Dim2, device)

        self.model_list = {"LSTM": LSTM(input_size=self.latDim + self.features_out, hidden_size=self.latDim, batch_first=False)}
        
        #Fusion
        self.MRI = nn.Linear(self.Dim, self.Dim)
        self.Demo = nn.Linear(self.features_out, self.features_out)

        self.Fusion = nn.ModuleList()
        self.Fusion.append(nn.Linear(self.features_out+self.Dim, self.latDim))
        self.Fusion.append(nn.Linear(self.latDim, self.latDim))
        self.Fusion.append(nn.Linear(self.latDim, self.latDim))

        # Encoder
        self.Fforward = nn.ModuleList()
        for i in range(self.mid_layer_num):
            self.Fforward.append(nn.Linear(self.latDim, self.latDim))
        if self.view_num == 2:
            
            self.Fforward.append(nn.Linear(self.latDim, self.latDim + self.features_out))
        else:
            self.Fforward.append(nn.Linear(self.Dim, self.Dim + self.features_out))

        self.classfier = nn.ModuleList()
        self.classfier.append(nn.Linear(self.latDim, self.hidden))
        self.classfier.append(nn.Linear(self.hidden, self.hidden))
        self.classfier.append(nn.Linear(self.hidden, self.lables))

        self.encoder_RNN = self.model_list[self.run_unit]

    
    def forward(self, X_score, X_missing_mask, New_X, M, X_demo):
        G_sample = self.Generator(New_X, M)
        Hat_New_X = New_X * M + G_sample * (1 - M)  
        
        MAI = F.relu(self.MRI(Hat_New_X)) + Hat_New_X 
        Demo = F.relu(self.Demo(X_demo)) + X_demo 
        concat = torch.cat((MAI, Demo), dim=2) 

        ...

        for i in range(1, self.seq_num):  
            missing_mask_temp = X_missing_mask[:,i].unsqueeze(dim=1) 
            input_temp = encoder_inputs[i,:,:].unsqueeze(dim=0) 
            input_temp = HardCalculateLayer(self.device)([encoder_single_output, input_temp, missing_mask_temp])  
            encoder_outputs, (encoder_state_h, encoder_state_c) = self.encoder_RNN(input_temp, states)  
           
            states = (encoder_state_h, encoder_state_c)

            for layer_num in range(self.mid_layer_num):
                encoder_outputs = F.relu(self.Fforward[layer_num](encoder_outputs))
            encoder_single_output = self.Fforward[-1](encoder_outputs) + input_temp  

            if i != self.seq_num - 1:   
                all_encoder_outputs.append(encoder_single_output)  

        encoder_outputs = torch.cat(all_encoder_outputs ,dim=0) 

        #Decoder
        decoder_inputs = encoder_single_output #(1,128,133)
        all_outputs = []
        inputs = decoder_inputs #(1,128,133)
        outputs = 0
        for _ in range(self.pred_length):
            decoder_outputs, (decoder_state_h, decoder_state_c) = self.encoder_RNN(inputs, states) 

            states = (decoder_state_h, decoder_state_c)

            for layer_num in range(self.mid_layer_num):
                decoder_outputs = F.relu(self.Fforward[layer_num](decoder_outputs))
            outputs = self.Fforward[-1](decoder_outputs) + inputs 
            all_outputs.append(outputs)
            inputs = outputs

        decoder_outputs_temp = torch.cat(all_outputs, dim=0)  
        if self.view_num == 2:  
            decoder_outputs_out = [decoder_outputs_temp[:, :, self.latDim + i].unsqueeze(dim=2) for i in range(self.features_out)]  
        else:
            decoder_outputs_out = [decoder_outputs_temp[:, :, self.Dim + i].unsqueeze(dim=2) for i in range(self.features_out)]

        outputs = outputs.squeeze()[:, :self.latDim] 
        class_out = torch.sigmoid(self.classfier[0](outputs))
        class_out = torch.sigmoid(self.classfier[1](class_out))
        class_out = self.classfier[2](class_out)

        return G_sample, Hat_New_X, latent_temp, encoder_outputs, decoder_outputs_out, class_out
