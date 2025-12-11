from transformers import ElectraModel, ElectraConfig
import torch
import torch.nn as nn

class ElectraForImageClassification(nn.Module):
    def __init__(self, num_labels, image_extractor, node_nums):
        super(ElectraForImageClassification, self).__init__()
        self.config = ElectraConfig.from_pretrained('google/electra-base-discriminator')
        self.config.num_labels = num_labels
        
        self.transformer = ElectraModel(self.config)
        self.image_extractor = image_extractor
        
        self.embedding_projection = nn.Linear(400, self.config.hidden_size)
        
        for param in self.transformer.parameters():
            param.requires_grad = False
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.config.hidden_size, node_nums),  
            nn.GELU(),
            nn.Linear(node_nums, num_labels)
        )

    def forward(self, images, labels=None):
        image_features = self.image_extractor(images)
        image_features = self.embedding_projection(image_features)
        
        inputs_embeds = image_features.unsqueeze(1).repeat(1, self.config.max_position_embeddings, 1)
        
        transformer_output = self.transformer(inputs_embeds=inputs_embeds)
        transformer_features = transformer_output.last_hidden_state[:, -1, :]
        logits = self.mlp(transformer_features)
        
        outputs = (logits,)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            outputs = (loss,) + outputs
        return outputs
