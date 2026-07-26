import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from deepfake_model import Model
import numpy as np
import matplotlib.pyplot as plt  
from torch import nn # Assuming you have defined the model architecture in a separate file
im_size = 112
sm = nn.Softmax()
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
inv_normalize =  transforms.Normalize(mean=-1*np.divide(mean,std),std=np.divide([1,1,1],std))
def process_videos(path_to_videos):
    class VideoDataset(Dataset):
        def __init__(self, video_paths, sequence_length, transform=None):
            self.video_paths = video_paths
            self.sequence_length = sequence_length
            self.transform = transform

        def __len__(self):
            return len(self.video_paths)

        def __getitem__(self, idx):
            video_path = self.video_paths[idx]
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
                frame_num += 1
                if frame_num % self.sequence_length == 0:
                    yield torch.stack(frames)
                    frames = []
            if frames:
                yield torch.stack(frames)
            cap.release()

    video_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    video_dataset = VideoDataset(path_to_videos, sequence_length=20, transform=video_transforms)
    model = Model(2).cuda()
    path_to_model = r"C:\Users\shubh\Downloads\model_97_acc_100_frames_FF_data.pt"
    model.load_state_dict(torch.load(path_to_model))
    model.eval()

    predictions = []

    for video_path in path_to_videos:
        for frames_batch in video_dataset:
            frames_batch = torch.stack(list(frames_batch)).cuda()
            prediction = predict(model, frames_batch)
            predictions.append((video_path, prediction))
            del frames_batch
            torch.cuda.empty_cache()

    return predictions



def predict(model,img):
    fmap,logits = model(img.to('cuda'))
    params = list(model.parameters())
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    logits = torch.softmax(logits, dim=1)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    print('confidence of prediction:', logits[:, int(prediction.item())].item() * 100)
    idx = np.argmax(logits.detach().cpu().numpy())
    bz, nc, h, w = fmap.shape
    out = np.dot(fmap[-1].detach().cpu().numpy().reshape((nc, h*w)).T, weight_softmax[idx, :].T)
    predict = out.reshape(h,w)
    predict = predict - np.min(predict)
    predict_img = predict / np.max(predict)
    predict_img = np.uint8(255*predict_img)
    out = cv2.resize(predict_img, (im_size,im_size))
    #heatmap = 
    img = im_convert(img[:,-1,:,:,:])
    result =  img*0.8*255
    cv2.imwrite('/content/drive/MyDrive/Qriocity/Secure Vision/Subset/Deep-Fake/train/fake_0_jpg.rf.627e22b7f278a118695d126f4ee791f6.jpg',result)
    result1 = img*0.8
    r,g,b = cv2.split(result1)
    result1 = cv2.merge((r,g,b))
    plt.imshow(result1)
    plt.show()
    return [int(prediction.item()),confidence]

def im_convert(tensor):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1,2,0)
    image = image.clip(0, 1)
    cv2.imwrite('./2.png',image*255)
    return image

def main():
    path_to_videos = [r"C:\Users\shubh\Downloads\Untitled video - Made with Clipchamp (9).mp4"]
    predictions = process_videos(path_to_videos)
    for prediction in predictions:
        print(prediction)

if __name__ == "__main__":
    main()
