import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torchvision import transforms, models, datasets
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import GPT2Model, GPT2Config
import torch.nn.functional as F


################################################################################
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
])

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 라벨을 부여할 폴더명과 라벨 매핑
        label_map = {
            'AGC': 0,
            'Dysplasia': 1,
            'EGC': 2,
            'Normal': 3
        }
        
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                # 폴더명이 매핑된 라벨을 부여
                if folder in label_map:
                    label = label_map[folder]
                    for image_name in os.listdir(folder_path):
                        image_path = os.path.join(folder_path, image_name)
                        self.image_paths.append(image_path)
                        self.labels.append(label)
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
####################################################################################
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=16):
        super(LoRALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r  # Rank
        self.alpha = alpha  # Scaling factor

        self.lora_A = nn.Parameter(torch.randn(out_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.scaling = self.alpha / self.r
        
        # 기존 가중치의 동결된 사본
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

from torchvision import models
from torchvision import models

class ConvNeXtFeatureExtractor(nn.Module):
    def __init__(self):
        super(ConvNeXtFeatureExtractor, self).__init__()
        # Pretrained ConvNeXt Base
        convnext = models.convnext_base(pretrained=True)
        
        # Feature Extraction Layer (Fully Connected Layer 제외)
        self.features = nn.Sequential(*list(convnext.children())[:-2])  # 마지막 FC 레이어 제거
        self.fc = nn.Linear(1024 * 14 * 14, 200)  # ConvNeXt Base의 출력 크기 7x7x1024
        
        # Grad-CAM 관련 변수
        self.activations = None
        self.gradients = None

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, x):
        # print(self.features)  # 여기에 출력하여 각 레이어를 확인
        print(self.features[0])
        print(len(self.features[0]))
        self.features[0][7].register_forward_hook(self.save_activations)  # 중간 레이어로 적절히 선택
        self.features[0][7].register_backward_hook(self.save_gradients)

        x = self.features(x)  # Feature Map 추출
        x = torch.flatten(x, 1)  # Flatten
        x = self.fc(x)  # FC Layer 통과
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F

class GPT2ForImageClassification(nn.Module):
    def __init__(self, num_labels):
        super(GPT2ForImageClassification, self).__init__()
        
        self.config = GPT2Config.from_pretrained('gpt2')
        self.config.num_labels = num_labels
        
        self.transformer = GPT2Model(self.config)
        self.image_extractor = ConvNeXtFeatureExtractor()
        self.embedding_projection = nn.Linear(200, 768)
    
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.config.n_positions, self.config.n_embd))  # Positional encoding

        # GELU 활성화 함수만 사용하는 모델
        self.mlp = nn.Sequential(
            nn.Linear(self.config.n_embd, 256),  # GPT2의 마지막 출력 차원에 맞추기
            nn.LeakyReLU(negative_slope=0.04),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.04),
            nn.Dropout(0.1),
            nn.Linear(128, 64),  # 중간 단계 추가
            nn.LeakyReLU(negative_slope=0.04),
            nn.Linear(64, 32),   # 점진적인 축소
            nn.LeakyReLU(negative_slope=0.04),
            nn.Linear(32, 4),     # 최종 출력
        )
        
        for name, module in self.transformer.named_modules():
            if isinstance(module, nn.Linear):
                setattr(self.transformer, name, LoRALinear(module.in_features, module.out_features, rank))
            elif isinstance(module, nn.MultiheadAttention):
                setattr(self.transformer, name, LoRAMultiheadAttention(module, rank))

    def forward(self, images, labels=None):
        image_features = self.image_extractor(images)
        
        image_features = self.embedding_projection(image_features)
        inputs_embeds = image_features.unsqueeze(1).repeat(1, self.config.n_positions, 1) + self.positional_encoding

        transformer_output = self.transformer(inputs_embeds=inputs_embeds)
        transformer_features = transformer_output.last_hidden_state[:, -1, :]
        logits = self.mlp(transformer_features)
        
        outputs = (logits,)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            outputs = (loss,) + outputs
        return outputs
####################################################################################
model = GPT2ForImageClassification(num_labels=4)  # num_labels는 모델에 맞게 설정
# 저장된 모델 불러오기
model.load_state_dict(torch.load('/mnt/block-storage/LLM_Classification/model/gpt2_REAKY_CONVNEXT.pth'))  # 모델 경로에 맞게 설정
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 이미지 로드
image_path = "/mnt/block-storage/LLM_Classification/data_v2/four class 강릉아산 1199/EGC/Anoy1_EGC LP_001.dcm.jpg"  # 이미지 경로에 맞게 설정
image = Image.open(image_path).convert("RGB")

# 모델에 맞는 전처리
transform = transforms.Compose([
    transforms.Resize((448, 448)),       # 모델에 맞게 크기 조정
    transforms.ToTensor(),               # 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 이미지 정규화
])

image_tensor = transform(image).to(device)  # 배치 차원 추가
####################################################################################
# import torch
# import torch.nn.functional as F
# import numpy as np
# import cv2
# from torchvision import transforms

# import torch
# import torch.nn.functional as F
# import numpy as np
# import cv2

# def generate_gradcam(model, image_tensor):
#     model.eval()
#     image_tensor.requires_grad = True

#     # 중간 레이어의 활성화 및 그래디언트 추적을 위한 hook 함수 설정
#     activations = []
#     gradients = []

#     def save_activation(module, input, output):
#         activations.append(output)
    
#     def save_gradient(module, grad_input, grad_output):
#         gradients.append(grad_output[0])
    
#     # 특정 레이어에 hook을 추가 (예: 마지막 convolutional layer)
#     target_layer = model.features[-1]  # 예시로 마지막 convolutional layer를 선택
#     target_layer.register_forward_hook(save_activation)
#     target_layer.register_backward_hook(save_gradient)
    
#     # Forward pass
#     output = model(image_tensor)

#     # 최댓값을 갖는 클래스를 선택 (target_class_idx가 올바르게 추출되도록)
#     target_class_idx = output.argmax(dim=1)  # 최댓값을 갖는 클래스 인덱스 선택
#     target_class_idx = target_class_idx.item()  # 텐서에서 값을 추출하여 정수로 변환

#     # Backward pass
#     model.zero_grad()
#     output[0, target_class_idx].backward()

#     # Grad-CAM 계산
#     grad = gradients[0]  # 첫 번째 샘플에 대한 그래디언트
#     activation = activations[0]  # 첫 번째 샘플에 대한 활성화

#     # 그래디언트의 평균을 가중치로 사용
#     weights = grad.mean(dim=(2, 3), keepdim=True)  # (C, H, W)에서 평균을 구함
#     grad_cam_map = (weights * activation).sum(dim=1).squeeze()  # 채널 차원에서 합산

#     # ReLU 함수로 음수 값을 제거
#     grad_cam_map = torch.relu(grad_cam_map)

#     # numpy 배열로 변환
#     grad_cam_map = grad_cam_map.cpu().detach().numpy()

#     return grad_cam_map

# # Example of using Grad-CAM
# grad_cam_map = generate_gradcam(model.image_extractor, image_tensor)

# # Grad-CAM 시각화
# import matplotlib.pyplot as plt

# plt.imshow(grad_cam_map, cmap='jet')
# plt.colorbar()
# plt.savefig('/mnt/block-storage/LLM_Classification/grad_cam_output.jpg')  # Grad-CAM 이미지를 파일로 저장
# plt.show()
####################################################################################
# import innvestigate
# import innvestigate.utils as iutils
# import numpy as np

# # 모델을 LRP 호환 모드로 변환
# analyzer = innvestigate.create_analyzer("lrp.epsilon", model)
# analysis = analyzer.analyze(image_tensor.cpu().numpy())

# # 결과 시각화
# plt.imshow(analysis[0, :, :], cmap='jet')
# plt.colorbar()
# plt.savefig('/mnt/block-storage/LLM_Classification/lrp_output.jpg')  # LRP 이미지를 파일로 저장
# plt.show()
####################################################################################
# import torch
# import matplotlib.pyplot as plt

# def get_attention_maps(model, image):
#     # 모델의 Attention Layer에서 attention map을 추출
#     attention_maps = []

#     # Hook을 이용해 attention map 추출
#     def hook_fn(module, input, output):
#         attention_maps.append(output[1])  # Attention map 추출

#     hooks = []
#     for layer in model.transformer.h:
#         hook = layer.attn.attn.register_forward_hook(hook_fn)
#         hooks.append(hook)

#     # 모델에 입력을 전달하여 attention map을 받음
#     image = image.unsqueeze(0)  # 배치 차원 추가
#     output = model(image)

#     # 시각화
#     attention_map = attention_maps[-1].squeeze().cpu().detach().numpy()  # 마지막 레이어의 attention map
#     plt.imshow(attention_map[0], cmap='jet')  # 첫 번째 헤드의 Attention Map
#     plt.colorbar()
#     plt.savefig('/mnt/block-storage/LLM_Classification/attention_map_output.jpg')  # Attention Map 이미지를 파일로 저장
#     plt.show()

#     # Hook 제거
#     for hook in hooks:
#         hook.remove()

# get_attention_maps(model, image_tensor)




import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def generate_gradcam(self, image_tensor):
        self.model.eval()
        output = self.model(image_tensor)

        self.model.zero_grad()
        class_idx = torch.argmax(output, dim=1).item() 
        output[0, class_idx].backward(retain_graph=True)

        gradients = self.model.gradients
        activations = self.model.activations

        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3], keepdim=True)

        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[:, i, :, :]

        grad_cam_map = torch.mean(activations, dim=1).squeeze()
        grad_cam_map = F.relu(grad_cam_map)
        grad_cam_map = grad_cam_map.cpu().detach().numpy()
        grad_cam_map = cv2.resize(grad_cam_map, (image_tensor.shape[2], image_tensor.shape[3]))
        grad_cam_map -= np.min(grad_cam_map)
        grad_cam_map /= np.max(grad_cam_map)

        return grad_cam_map

    def overlay_gradcam(self, image_tensor, gradcam_map):
        gradcam_map = cv2.applyColorMap(np.uint8(255 * gradcam_map), cv2.COLORMAP_JET)
        gradcam_map = np.float32(gradcam_map) / 255 
        image = image_tensor.squeeze().cpu().detach().numpy()
        image = np.transpose(image, (1, 2, 0)) 
        image = (image - image.min()) / (image.max() - image.min())  # 스케일 조정 (0~1 사이 값으로)
        image = np.float32(image)
        overlay = cv2.addWeighted(image, 0.6, gradcam_map, 0.4, 0)  # 두 이미지를 합성
        overlay = np.uint8(255 * overlay) 
        return overlay


grad_cam = GradCAM(model.image_extractor, model.image_extractor.features[0][7])  # 타겟 레이어 설정

image_tensor = transform(image).unsqueeze(0).to(device)

grad_cam_map = grad_cam.generate_gradcam(image_tensor)

overlay_image = grad_cam.overlay_gradcam(image_tensor, grad_cam_map)

plt.imshow(overlay_image)
plt.savefig('/mnt/block-storage/LLM_Classification/grad_cam_output.jpg')  # Grad-CAM 이미지를 파일로 저장
plt.show()


import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from captum.attr import Saliency
from captum.attr import DeepLift, visualization as viz
import shap

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def generate_gradcam(self, image_tensor):
        # Forward Pass
        self.model.eval()
        output = self.model(image_tensor)

        # Backward Pass
        self.model.zero_grad()
        class_idx = torch.argmax(output, dim=1).item()  # 예측된 클래스 인덱스
        output[0, class_idx].backward(retain_graph=True)  # 예측 클래스에 대한 그라디언트 계산

        # Grad-CAM 생성
        gradients = self.model.gradients
        activations = self.model.activations

        # 클래스별 중요도 계산 (채널별 평균 그라디언트)
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3], keepdim=True)

        # 활성화 맵에 중요도 반영
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[:, i, :, :]

        # 활성화 맵을 채널별로 합침
        grad_cam_map = torch.mean(activations, dim=1).squeeze()

        # ReLU 활성화 함수 (음수값 제거)
        grad_cam_map = F.relu(grad_cam_map)

        # Grad-CAM 맵 크기 조정 (원본 이미지 크기에 맞게)
        grad_cam_map = grad_cam_map.cpu().detach().numpy()
        grad_cam_map = cv2.resize(grad_cam_map, (image_tensor.shape[2], image_tensor.shape[3]))
        grad_cam_map -= np.min(grad_cam_map)
        grad_cam_map /= np.max(grad_cam_map)

        return grad_cam_map

    def overlay_gradcam(self, image_tensor, gradcam_map):
        # Grad-CAM 맵을 이미지에 덧씌우기 (컬러맵 적용)
        gradcam_map = cv2.applyColorMap(np.uint8(255 * gradcam_map), cv2.COLORMAP_JET)
        gradcam_map = np.float32(gradcam_map) / 255  # float32로 변환

        # 이미지 텐서를 numpy 배열로 변환 및 스케일 조정
        image = image_tensor.squeeze().cpu().detach().numpy()
        image = np.transpose(image, (1, 2, 0))  # 채널 순서 변경 (C, H, W -> H, W, C)
        image = (image - image.min()) / (image.max() - image.min())  # 스케일 조정 (0~1 사이 값으로)
        image = np.float32(image)  # float32로 변환

        # 합성
        overlay = cv2.addWeighted(image, 0.6, gradcam_map, 0.4, 0)  # 두 이미지를 합성
        overlay = np.uint8(255 * overlay)  # 결과를 uint8로 변환
        return overlay

class SaliencyMap:
    def __init__(self, model):
        self.model = model
        self.saliency = Saliency(self.model)

    def generate_saliency(self, image_tensor, target=None):
        self.model.eval()
        saliency_map = self.saliency.attribute(image_tensor, target=target)
        saliency_map = saliency_map.squeeze().cpu().detach().numpy()
        saliency_map = np.abs(saliency_map).max(axis=0)
        saliency_map -= np.min(saliency_map)
        saliency_map /= np.max(saliency_map)
        return saliency_map

class SHAPExplainer:
    def __init__(self, model):
        self.model = model

    def generate_shap(self, image_tensor, background_tensor):
        self.model.eval()
        explainer = shap.DeepExplainer(self.model, background_tensor)
        shap_values = explainer.shap_values(image_tensor)
        shap_map = np.mean(np.abs(shap_values[0]), axis=1).squeeze()
        shap_map -= np.min(shap_map)
        shap_map /= np.max(shap_map)
        return shap_map

# Example usage

# Grad-CAM
grad_cam = GradCAM(model.image_extractor, model.image_extractor.features[0][7])  # 타겟 레이어 설정
image_tensor = transform(image).unsqueeze(0).to(device)
grad_cam_map = grad_cam.generate_gradcam(image_tensor)
overlay_image = grad_cam.overlay_gradcam(image_tensor, grad_cam_map)
plt.imshow(overlay_image)
plt.savefig('/mnt/block-storage/LLM_Classification/grad_cam_output.jpg')  # Grad-CAM 이미지를 파일로 저장
plt.show()

# Saliency Map
saliency_map_extractor = SaliencyMap(model.image_extractor)
saliency_map = saliency_map_extractor.generate_saliency(image_tensor)
plt.imshow(saliency_map, cmap='hot')
plt.colorbar()
plt.savefig('/mnt/block-storage/LLM_Classification/saliency_map_output.jpg')  # Saliency 맵 저장
plt.show()

# SHAP
background_tensor = torch.cat([image_tensor] * 5, dim=0)  # 배경 데이터 생성
shap_explainer = SHAPExplainer(model.image_extractor)
shap_map = shap_explainer.generate_shap(image_tensor, background_tensor)
plt.imshow(shap_map, cmap='coolwarm')
plt.colorbar()
plt.savefig('/mnt/block-storage/LLM_Classification/shap_output.jpg')  # SHAP 맵 저장
plt.show()


import matplotlib.pyplot as plt

# 예시 코드: 이미지, 예측값, 실제값 비교 및 평가
num_rows = 2
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))

for i in range(num_images):
    plt.subplot(num_rows, num_cols, 2 * i + 1)
    plt.imshow(test_images[i])
    plt.title(f"True: {test_labels[i]}, Pred: {predictions[i]}")
    
    # 잘못된 예측 강조 (빨간색으로)
    if predictions[i] != test_labels[i]:
        plt.text(0.5, 0.5, 'Wrong', color='red', ha='center', va='center', fontsize=12)
    
    plt.subplot(num_rows, num_cols, 2 * i + 2)
    plt.bar(range(len(predictions[i])), predictions[i], color='blue', alpha=0.6, label="Predictions")
    plt.bar(range(len(test_labels[i])), test_labels[i], color='red', alpha=0.6, label="True Labels")
    plt.legend(loc='upper right')

plt.tight_layout()
plt.show()



import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from captum.attr import LayerConductance

class LRP:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def generate_lrp(self, image_tensor):
        # 모델을 evaluation 모드로 설정
        self.model.eval()
        
        # Forward Pass
        output = self.model(image_tensor)

        # 예측 클래스의 인덱스 얻기
        class_idx = torch.argmax(output, dim=1).item()

        # Backward Pass
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)  # 예측 클래스에 대한 그라디언트 계산

        # LRP 계산을 위한 활성화 및 그라디언트 추출
        activations = self.target_layer.output
        gradients = self.target_layer.gradients

        # 클래스별 중요도 계산 (채널별 평균 그라디언트)
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3], keepdim=True)

        # 활성화 맵에 중요도 반영
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[:, i, :, :]

        # 활성화 맵을 채널별로 합침
        lrp_map = torch.mean(activations, dim=1).squeeze()

        # ReLU 활성화 함수 (음수값 제거)
        lrp_map = F.relu(lrp_map)

        # LRP 맵 크기 조정 (원본 이미지 크기에 맞게)
        lrp_map = lrp_map.cpu().detach().numpy()
        lrp_map = cv2.resize(lrp_map, (image_tensor.shape[2], image_tensor.shape[3]))
        lrp_map -= np.min(lrp_map)
        lrp_map /= np.max(lrp_map)

        return lrp_map

    def overlay_lrp(self, image_tensor, lrp_map):
        # LRP 맵을 이미지에 덧씌우기 (컬러맵 적용)
        lrp_map = cv2.applyColorMap(np.uint8(255 * lrp_map), cv2.COLORMAP_JET)
        lrp_map = np.float32(lrp_map) / 255  # float32로 변환

        # 이미지 텐서를 numpy 배열로 변환 및 스케일 조정
        image = image_tensor.squeeze().cpu().detach().numpy()
        image = np.transpose(image, (1, 2, 0))  # 채널 순서 변경 (C, H, W -> H, W, C)
        image = (image - image.min()) / (image.max() - image.min())  # 스케일 조정 (0~1 사이 값으로)
        image = np.float32(image)  # float32로 변환

        # 합성
        overlay = cv2.addWeighted(image, 0.6, lrp_map, 0.4, 0)  # 두 이미지를 합성
        overlay = np.uint8(255 * overlay)  # 결과를 uint8로 변환
        return overlay
# LRP
lrp_explainer = LRP(model.image_extractor, model.image_extractor.features[0][7])  # 타겟 레이어 설정
image_tensor = transform(image).unsqueeze(0).to(device)

# LRP 맵 생성
lrp_map = lrp_explainer.generate_lrp(image_tensor)

# LRP 맵을 이미지에 덧씌우기
overlay_image = lrp_explainer.overlay_lrp(image_tensor, lrp_map)

# 결과 출력 및 저장
plt.imshow(overlay_image)
plt.savefig('/mnt/block-storage/LLM_Classification/lrp_output.jpg')  # LRP 이미지를 파일로 저장
plt.show()
