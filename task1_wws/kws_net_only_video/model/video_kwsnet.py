# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
from .extract_lip_embedding_resnet import VideoFrontend


class LSTM_Encoder(nn.Module):
    def __init__(self, feature_dim, hidden_size, num_layers):
        super(LSTM_Encoder, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.stack_rnn = nn.LSTM(input_size=self.feature_dim, hidden_size=self.hidden_size, batch_first=False,
                                 bidirectional=False, num_layers=1)

    def forward(self, cur_inputs, current_frame):
        packed_input = nn.utils.rnn.pack_padded_sequence(cur_inputs, current_frame)
        rnn_out, _ = self.stack_rnn(packed_input)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out)

        return rnn_out


class KWS_Net(nn.Module):
    def __init__(self, args):
        super(KWS_Net, self).__init__()
        self.lip_encoder = VideoFrontend()
        # self.conv_av1 = nn.Conv2d(256,256, kernel_size=(1, 5), stride=(1,2), padding=(0,2))
        # self.conv_av2 = nn.Conv2d(256,512, kernel_size=(1, 5), stride=(1,2), padding=(0,2))

        self.feature_dim = args.input_dim
        self.hidden_size = args.hidden_sizes
        self.num_layers = args.lstm_num_layers

        self.encoder = LSTM_Encoder(256, self.hidden_size, self.num_layers)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.conv2 = nn.Conv2d(32, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.max_pool1 = nn.MaxPool2d(2, stride=(1, 2))
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(5, 5), stride=(1, 2), padding=(2, 2))
        self.fc1 = nn.Linear(256, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, video_inputs, current_frame):
        video_inputs = video_inputs.permute(1, 0, 2, 3)

        lip_inputs = self.lip_encoder(video_inputs)  # (B, T, 256)
        lip_inputs = lip_inputs.permute(1, 0, 2)  # (T, B, 256)

        # #lstm layer
        encoder_output = self.encoder(lip_inputs, current_frame)
        encoder_output = encoder_output.permute(1, 2, 0)
        # print(encoder_output.shape) # (B,64,T)

        # #CNN detector and classifier 
        cnn_input = encoder_output.unsqueeze(1)
        cnn_out = self.conv1(cnn_input)
        cnn_out = self.conv2(cnn_out)
        cnn_out = self.max_pool1(cnn_out)
        cnn_out = ((self.conv3(cnn_out)).mean(-2)).permute(0, 2, 1)
        cnn_out = self.dropout(cnn_out)
        cnn_out = self.fc1(cnn_out)
        cnn_out = self.dropout(cnn_out)
        cnn_out = self.fc2(cnn_out)
        cnn_out = self.dropout(cnn_out)
        cnn_out = self.fc3(cnn_out)

        max_pool2 = nn.MaxPool2d((cnn_out.shape[1], 1))
        cnn_out = max_pool2(cnn_out)
        cnn_out = cnn_out.squeeze(-1)

        return cnn_out
