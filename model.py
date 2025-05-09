import torch.nn as nn

class BERT_Arch(nn.Module):
    def __init__(self, bert_model, num_labels):
        super().__init__()
        self.bert    = bert_model
        self.dropout = nn.Dropout(0.2)
        self.relu    = nn.ReLU()
        self.fc1     = nn.Linear(768, 512)
        self.fc2     = nn.Linear(512, 256)
        self.fc3     = nn.Linear(256, num_labels)

    def forward(self, input_ids, attention_mask):
        cls_emb = self.bert(input_ids, attention_mask=attention_mask)[0][:,0]
        x = self.relu(self.fc1(cls_emb))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)
        return logits