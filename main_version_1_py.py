import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import timm
from timm.data import create_transform
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
import numpy as np
import requests
from io import BytesIO
from fastapi import FastAPI, File, UploadFile
import imageio
from fastapi.responses import JSONResponse

app = FastAPI()


transform = transforms.Compose([
    transforms.Resize((512, 512)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_res = create_transform(
    input_size=(224, 224),
    is_training=True
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r'/app/models/effnet.pth'
model_classification = models.efficientnet_b1(pretrained=True)
num_classes = 5
model_classification.classifier[1] = nn.Linear(model_classification.classifier[1].in_features, num_classes)
model_classification.load_state_dict(torch.load(model_path,  map_location=device))
model_classification.eval()
model_classification.to(device)

model_name = 'efficientnet_b0'
resnet_model = timm.create_model(model_name, pretrained=False, num_classes=2)
model_path = r'/app/models/best_model_dirty_eff18.pth'
resnet_model.load_state_dict(torch.load(model_path,  map_location=device))
resnet_model = resnet_model.to(device)
resnet_model.eval()

class_names = {
    0: "concrete",
    1: "brick",
    3: "wood",
    4: "ground",
    2: "tent"
}

@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    contents = await file.read()
    video_path = "temp_video.mp4"
    number_car = file.filename.split('.')[0]

    with open(video_path, "wb") as f:
        f.write(contents)

    closest_box_info = extract_frames(video_path, frame_interval=24, start_time=110, end_time=140)
    frame = extract_single_frame(video_path)

    response_data = {
        "number_car": number_car,
        "is_awning": None,
        "predicted_class_resnet": None
    }

    if frame is not None:
        probabilities_resnet, predicted_class_resnet, max_prob_resnet = classify_image_resnet(frame, resnet_model, transform_res, device)
        response_data["predicted_class_resnet"] = predicted_class_resnet == 1
        response_data["max_prob_resnet"] = max_prob_resnet
        print("max_prob_resnet:", max_prob_resnet)
    
    if closest_box_info is not None and closest_box_info[0] is not None:
        probabilities_efficientnet, predicted_class_efficientnet, max_prob_efficientnet = classify_image(closest_box_info[0], model_classification, transform, device)
        response_data["predicted_class_efficientnet"] = class_names.get(predicted_class_efficientnet, f"Class_{predicted_class_efficientnet}")

        class_probabilities = {}
        max_probability = 0
        max_class_name = None
        total_prob_without_tent = 0

        for i, prob in enumerate(probabilities_efficientnet[0]):
            class_name = class_names.get(i, f"Class_{i}")
            prob_value = prob.item()

            if i == 2:  
                response_data["is_awning"] = prob_value > 0.5
                continue

            class_probabilities[class_name] = prob_value
            total_prob_without_tent += prob_value

            if prob_value > max_probability:
                max_probability = prob_value
                max_class_name = class_name

        if max_class_name is None and class_probabilities:
            max_class_name = max(class_probabilities, key=class_probabilities.get)

        response_data["max_class_name"] = max_class_name

        for class_name in class_probabilities:
            if class_name != "tent":
                class_probabilities[class_name] = round((class_probabilities[class_name] / total_prob_without_tent) * 100)

        closest_box_cls_name = response_data.get("closest_box_cls")
        average_confidence = (response_data.get("closest_box_conf", 0) + response_data.get("max_prob_resnet", 0)) / 2
        is_same_class = closest_box_cls_name == response_data.get("max_class_name")
        is_high_confidence = average_confidence >= 0.75
        is_very_high_resnet_confidence = response_data.get("max_prob_resnet", 0) >= 0.85
        response_data["is_exactly"] = is_same_class and is_high_confidence or (not is_same_class and is_very_high_resnet_confidence)

        final_response = {
            "number_car": response_data["number_car"],
            "is_exactly": response_data["is_exactly"],
            "is_awning": response_data.get("is_awning", False),
            "wood": class_probabilities.get("wood", 0),
            "dirt": class_probabilities.get("ground", 0),
            "concrete": class_probabilities.get("concrete", 0),
            "brick": class_probabilities.get("brick", 0),
            "statement": response_data["max_class_name"],
            "is_visibility": response_data["predicted_class_resnet"]
        }

        image_to_send = Image.fromarray(closest_box_info[0])
        buffered = BytesIO()
        image_to_send.save(buffered, format="JPEG")
        buffered.seek(0)

        url = 'http://example.com/api/upload'
        data = {'json_data': final_response}
        files = {'image': ('image.jpg', buffered, 'image/jpeg')}

        response = requests.post(url, data=data, files=files)

        if response.status_code == 200:
            print("Запрос успешно отправлен")
        else:
            print("Ошибка при отправке запроса:", response.status_code)

    return JSONResponse(content=final_response)

def extract_frames(video_path, frame_interval=1, start_time=0, end_time=None):
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return

    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()['fps']
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time is not None else None

    model_detection = YOLO(r"/app/models/best.pt")

    closest_box_image = None
    highest_confidence = 0
    closest_box_cls = None
    closest_box_conf = None

    for i, frame in enumerate(reader):
        if i < start_frame or (end_frame is not None and i > end_frame):
            continue
        if (i - start_frame) % frame_interval != 0:
            continue

        result = model_detection.predict(frame)
        if not result[0].boxes.xywh.numel():
            continue

        image_center = np.array(frame.shape[1::-1]) / 2
        closest_distance = float('inf')

        for j, box in enumerate(result[0].boxes.xywh):
            x_center, y_center, width, height = box
            x_min = int(x_center.item() - width.item() / 2)
            y_min = int(y_center.item() - height.item() / 2)
            box_center = np.array([x_center.item(), y_center.item()])
            distance = np.linalg.norm(box_center - image_center)

            conf = result[0].boxes.conf[j].item()
            if conf > highest_confidence and distance < closest_distance:
                closest_box_image = frame[y_min:y_min + int(height.item()), x_min:x_min + int(width.item())]
                closest_box_cls = result[0].boxes.cls[j].item()
                closest_box_conf = conf
                highest_confidence = conf
                closest_distance = distance

    return closest_box_image, closest_box_cls, closest_box_conf

def classify_image(image_np, model, transform, device):
    image_pil = Image.fromarray(image_np)
    
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        max_prob, predicted_class = torch.max(probabilities, 1)

    return probabilities.cpu().numpy(), predicted_class.cpu().item(), max_prob.cpu().item()


def extract_single_frame(video_path, frame_number=140):
    reader = imageio.get_reader(video_path)
    for i, frame in enumerate(reader):
        if i == frame_number:
            return frame
    return None

def classify_image_resnet(image_np, model, transform, device):
    image_pil = Image.fromarray(image_np)
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        max_prob, predicted_class = torch.max(probabilities, 1)

    return probabilities.cpu().numpy(), predicted_class.cpu().item(), max_prob.cpu().item()
