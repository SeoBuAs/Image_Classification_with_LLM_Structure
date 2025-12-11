from transformers import T5Config, T5EncoderModel
import torch
import torch.nn as nn

class T5ForImageClassification(nn.Module):
    def __init__(self, num_labels, image_extractor, node_nums):
        super(T5ForImageClassification, self).__init__()
        self.config = T5Config.from_pretrained('t5-base')
        hidden_size = self.config.d_model
        self.config.num_labels = num_labels
        
        self.transformer = T5EncoderModel(self.config)
        self.image_extractor = image_extractor

        self.embedding_projection = nn.Linear(400, hidden_size)

        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.config.hidden_size, node_nums),  
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(node_nums, num_labels)
        )

        for param in self.transformer.parameters():
            param.requires_grad = False

    def forward(self, images, labels=None):
        image_features = self.image_extractor(images)
        image_features = self.embedding_projection(image_features)

        inputs_embeds = image_features.unsqueeze(1)
        batch_size = inputs_embeds.size(0)
        seq_length = inputs_embeds.size(1)
        attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=inputs_embeds.device)
        
        encoder_outputs = self.transformer(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        transformer_features = encoder_outputs.last_hidden_state[:, -1, :]
        logits = self.mlp(transformer_features)
        
        outputs = (logits,)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            outputs = (loss,) + outputs
        return outputs