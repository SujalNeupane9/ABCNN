import torch
import torch.nn as nn
import torch.nn.functional as functional
from transformers import AutoTokenizer, AutoModel

class ABCNN3(nn.Module):
    def __init__(self, model_name, num_classes):
        super(ABCNN3, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedding = AutoModel.from_pretrained(model_name)
        self.rnn = nn.GRU(input_size=self.embedding.config.hidden_size, hidden_size=128, num_layers=1,
                bidirectional=True, batch_first=True)
        self.attention = nn.Linear(256, 1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, inputs):
        x1_encoded = self.embedding(inputs['input_ids'], attention_mask=inputs['attention_mask'], token_type_ids=inputs['token_type_ids']).last_hidden_state.squeeze(1)
        x2_encoded = self.embedding(inputs['input_ids'], attention_mask=inputs['attention_mask'], token_type_ids=inputs['token_type_ids']).last_hidden_state.squeeze(1)

        _, x1_h = self.rnn(x1_encoded)
        _, x2_h = self.rnn(x2_encoded)

        x1_h = torch.cat([x1_h[0], x1_h[1]], dim=1)
        x2_h = torch.cat([x2_h[0], x2_h[1]], dim=1)

        x1_att = self.attention(x1_h).unsqueeze(1)
        x2_att = self.attention(x2_h).unsqueeze(1)

        x1_weighted = torch.bmm(torch.transpose(x1_att, 1, 2), x1_h.unsqueeze(1)).squeeze()
        x2_weighted = torch.bmm(torch.transpose(x2_att, 1, 2), x2_h.unsqueeze(1)).squeeze()

        #print(x1_weighted.shape,x2_weighted.shape)
        x = torch.cat([x1_weighted, x2_weighted], dim=1)
        #print(x.shape)
        logits = self.fc(x)

        return logits
