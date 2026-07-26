import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from deepfake_model import Model
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import uuid

# Define global variables
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
inv_normalize = transforms.Normalize(mean=-1 * np.divide(mean, std), std=np.divide([1, 1, 1], std))


def predict(model, img):
    with torch.no_grad():
        fmap, logits = model(img)
        weight_softmax = model.linear1.weight.detach().cpu().numpy()
        logits = torch.softmax(logits, dim=1)
        _, prediction = torch.max(logits, 1)
        confidence = logits[:, int(prediction.item())].item() * 100
        print('confidence of prediction:', logits[:, int(prediction.item())].item() * 100)
        idx = np.argmax(logits.detach().cpu().numpy())
        bz, nc, h, w = fmap.shape
        out = np.dot(fmap[-1].detach().cpu().numpy().reshape((nc, h * w)).T, weight_softmax[idx, :].T)
        predict = out.reshape(h, w)
        predict = predict - np.min(predict)
        predict_img = predict / np.max(predict)
        predict_img = np.uint8(255 * predict_img)
        out = cv2.resize(predict_img, (im_size, im_size))
        img = im_convert(img[:, -1, :, :, :])
        result = img * 0.8 * 255
        result1 = img * 0.8
        return [int(prediction.item()), confidence, result1]


# Define the image conversion function
def im_convert(tensor):
    """ Convert a tensor to a numpy array image. """
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    return (image * 255).astype(np.uint8)


# Inside the process_videos function
def process_videos(path_to_videos):
    predictions = []
    for video_path in path_to_videos:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sequence_length = frame_count
        batch_size = 10  # Adjust batch size as needed

        for start in range(0, frame_count, batch_size):
            frames = []
            for i in range(start, min(start + batch_size, frame_count)):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (im_size, im_size))  # Resize frame
                frames.append(frame)
            if frames:
                frames = np.stack(frames)
                frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
                frames = frames.unsqueeze(0).cpu()  # Add batch dimension and move to CPU
                predictions += [predict(model, frames)]  # Append prediction as tuple
    return predictions



# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model
model = Model(2)  # Assuming the model architecture is defined and loaded properly
path_to_model = r"C:\Users\shubh\Downloads\model_97_acc_100_frames_FF_data.pt"
model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
model.eval()

# Define the route for the home page
@app.route('/')
def home():
    return render_template('tindex.html')

# Inside the upload_video route
# Inside the upload_video route
@app.route('/upload_video', methods=['POST'])
def upload_video():
    # Check if a video file was uploaded
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    # Check if the file is empty
    if file.filename == '':
        return redirect(request.url)
    # Check if the file is a valid video file
    if file and file.filename.endswith('.mp4'):
        # Save the uploaded video to a temporary directory
        video_path = secure_filename(file.filename)
        file.save(video_path)
        # Process the uploaded video for deepfake detection
        predictions = process_videos([video_path])
        # Determine if the video contains deepfakes
        is_deepfake = any(prediction[0] == 1 for prediction in predictions)
        if is_deepfake:
            # If deepfakes are detected, handle accordingly
            return render_template('deepfake_results.html')
        else:
            # If no deepfakes are detected, display a message indicating that the video is real
            return render_template('real_video.html')
    else:
        return render_template('error.html', message="Invalid file format. Please upload an MP4 video file.")



if __name__ == '__main__':
    app.run(debug=True)
