from transformers import GPT2Model, GPT2Config
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=16):
        super(LoRALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.lora_A = nn.Parameter(torch.randn(out_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.scaling = self.alpha / self.r
        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
    
    def forward(self, x):
        return F.linear(x, self.weight) + self.scaling * F.linear(x, self.lora_A @ self.lora_B)
    
class LoRAMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, r=8):
        super(LoRAMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.r = r
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.query_A = nn.Parameter(torch.randn(embed_dim, r))
        self.query_B = nn.Parameter(torch.randn(r, embed_dim))
        self.key_A = nn.Parameter(torch.randn(embed_dim, r))
        self.key_B = nn.Parameter(torch.randn(r, embed_dim))
        self.value_A = nn.Parameter(torch.randn(embed_dim, r))
        self.value_B = nn.Parameter(torch.randn(r, embed_dim))

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        query = query @ self.query_A @ self.query_B
        key = key @ self.key_A @ self.key_B
        value = value @ self.value_A @ self.value_B
        return self.attention(query, key, value, attn_mask, key_padding_mask)

class GPT2ForImageClassification(nn.Module):
    def __init__(self, num_labels, image_extractor, node_nums):
        super(GPT2ForImageClassification, self).__init__()
        
        self.config = GPT2Config.from_pretrained('gpt2')
        self.config.num_labels = num_labels
        
        self.transformer = GPT2Model(self.config)
        self.image_extractor = image_extractor
        self.embedding_projection = nn.Linear(400, 768)
    
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.config.hidden_size, node_nums),  
            nn.GELU(),
            nn.Linear(node_nums, num_labels)
        )
        
        for param in self.transformer.parameters():
            param.requires_grad = False
            
        for name, module in self.transformer.named_modules():
            if isinstance(module, nn.Linear):
                setattr(self.transformer, name, LoRALinear(module.in_features, module.out_features, rank))
            elif isinstance(module, nn.MultiheadAttention):
                setattr(self.transformer, name, LoRAMultiheadAttention(module, rank))

    def forward(self, images, labels=None):
        image_features = self.image_extractor(images)
        image_features = self.embedding_projection(image_features)
        inputs_embeds = image_features.unsqueeze(1).repeat(1, self.config.n_positions, 1)

        transformer_output = self.transformer(inputs_embeds=inputs_embeds)
        transformer_features = transformer_output.last_hidden_state[:, -1, :]
        logits = self.mlp(transformer_features)
        
        outputs = (logits,)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            outputs = (loss,) + outputs
        return outputs