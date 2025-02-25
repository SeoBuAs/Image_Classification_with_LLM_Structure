from transformers import ElectraModel, ElectraConfig
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

# class ElectraForImageClassification(nn.Module):
#     def __init__(self, num_labels, image_extractor, node_nums):
#         super(ElectraForImageClassification, self).__init__()
#         self.config = ElectraConfig.from_pretrained('google/electra-base-discriminator')
#         self.config.num_labels = num_labels
        
#         self.transformer = ElectraModel(self.config)
#         self.image_extractor = image_extractor
        
#         self.embedding_projection = nn.Linear(400, self.config.hidden_size)
#         self.lora_modules = []
#         self.mlp = nn.Sequential(
#             nn.GELU(),
#             nn.Linear(self.config.hidden_size, node_nums),  
#             nn.GELU(),
#             nn.Linear(node_nums, num_labels)
#         )

#     #     self.replace_modules_with_lora()

#     # def replace_modules_with_lora(self):
#     #     for name, module in self.transformer.named_modules():
#     #         if isinstance(module, nn.Linear):
#     #             lora_module = LoRALinear(module.in_features, module.out_features)
#     #             self._replace_module(name, lora_module)
#     #         elif isinstance(module, nn.MultiheadAttention):
#     #             lora_module = LoRAMultiheadAttention(module.embed_dim, module.num_heads)
#     #             self._replace_module(name, lora_module)

#     # def _replace_module(self, module_name, new_module):
#     #     parts = module_name.split('.')
#     #     parent = self.transformer
#     #     for part in parts[:-1]:
#     #         if part.isdigit():
#     #             parent = parent[int(part)]
#     #         else:
#     #             parent = getattr(parent, part)
#     #     setattr(parent, parts[-1], new_module)

#     def forward(self, images, labels=None):
#         image_features = self.image_extractor(images)
#         image_features = self.embedding_projection(image_features)
        
#         inputs_embeds = image_features.unsqueeze(1).repeat(1, self.config.max_position_embeddings, 1)
        
#         transformer_output = self.transformer(inputs_embeds=inputs_embeds)
#         transformer_features = transformer_output.last_hidden_state[:, -1, :]
#         logits = self.mlp(transformer_features)
        
#         outputs = (logits,)
#         if labels is not None:
#             loss = nn.CrossEntropyLoss()(logits, labels)
#             outputs = (loss,) + outputs
#         return outputs


class ElectraForImageClassification(nn.Module):
    def __init__(self, num_labels, image_extractor, node_nums):
        super(ElectraForImageClassification, self).__init__()
        self.config = ElectraConfig.from_pretrained('google/electra-base-discriminator')
        self.config.num_labels = num_labels
        
        self.transformer = ElectraModel(self.config)
        self.image_extractor = image_extractor
        
        self.embedding_projection = nn.Linear(400, self.config.hidden_size)
        self.lora_modules = []
        for param in self.transformer.parameters():
            param.requires_grad = False
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.config.hidden_size, node_nums),  
            nn.GELU(),
            nn.Linear(node_nums, num_labels)
        )

    #     self.replace_modules_with_lora()

    # def replace_modules_with_lora(self):
    #     for name, module in self.transformer.named_modules():
    #         if isinstance(module, nn.Linear):
    #             lora_module = LoRALinear(module.in_features, module.out_features)
    #             self._replace_module(name, lora_module)
    #         elif isinstance(module, nn.MultiheadAttention):
    #             lora_module = LoRAMultiheadAttention(module.embed_dim, module.num_heads)
    #             self._replace_module(name, lora_module)

    # def _replace_module(self, module_name, new_module):
    #     parts = module_name.split('.')
    #     parent = self.transformer
    #     for part in parts[:-1]:
    #         if part.isdigit():
    #             parent = parent[int(part)]
    #         else:
    #             parent = getattr(parent, part)
    #     setattr(parent, parts[-1], new_module)

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
