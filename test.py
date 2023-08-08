import sys
import os
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import transforms
from dataloader.definitions.labels_file import *
from dataloader.pt_data_loader import mytransforms
from dataset_util import Cityscapes, Kitti
from models.erfnet import ERFNet
from src.options import ERFnetOptions
labels=labels_cityscape_seg.getlabels()
PALETTE=[]
for label in labels:
    PALETTE.extend(label.color)

DATASET_DIR=os.path.join("./Dataset",'cityscapes')

# labels = self._get_labels_cityscapes()
# Get the colours from dataset.
class Visualize():
    def __init__(self):
        self.colors = [(label.id, label.color) for label in labels]
        self.colors.append((255, (0, 0, 0)))  # void class
        self.id_color = dict(self.colors)
        self.id_color_keys = [key for key in self.id_color.keys()]
        self.id_color_vals = [val for val in self.id_color.values()]

    def convert_to_colour(self, img,save_path):
        ''' Replace trainIDs in prediction with colours from dict, reshape it afterwards to input dimensions and
            convert RGB to BGR to match openCV's colour system.
        '''
        sort_idx = np.argsort(self.id_color_keys)
        idx = np.searchsorted(self.id_color_keys, img, sorter=sort_idx)
        img = np.asarray(self.id_color_vals)[sort_idx][idx]
        img = img.astype(np.uint8)
        # img = np.reshape(img, newshape=(o_size[0], o_size[1], 3))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path,img)




class Segmentation():
    def __init__(self,model_path=None,output_dir=None,dataset="cityscapes"):
        assert model_path is not None
        assert output_dir is not None
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainid_to_labels = None
        labels = None
        if dataset == "cityscapes":
            # cityscapes = Cityscapes()
            # _,self.content_dict = cityscapes.load_dataset()
            # self.content_dict = self.content_dict[""]
            # Labels
            labels = labels_cityscape_seg_train3_eval.getlabels()
            self.trainid_to_labels = labels_cityscape_seg_train3_eval.gettrainid2label()
        elif dataset== "kitti":
            # kitti=Kitti()
            # self.content_dict = kitti.train_set
            labels = labels_kitti_seg_train2_eval.getlabels()
            self.trainid_to_labels = labels_kitti_seg_train2_eval.gettrainid2label()
    
        # Train IDs
        self.train_ids = set([labels[i].trainId for i in range(len(labels))])
        self.train_ids.remove(255)
        self.train_ids = sorted(list(self.train_ids))

        self.num_classes_model = len(self.train_ids)

        self.transforms = transforms.Compose([mytransforms.CreateScaledImage(),
                        mytransforms.Resize((512, 1024), image_types=['color']),
                        mytransforms.ConvertSegmentation(),
                        mytransforms.CreateColoraug(new_element=True, scales=[0]),
                        mytransforms.RemoveOriginals(),
                        mytransforms.ToTensor(),
                        mytransforms.NormalizeZeroMean(),
                        ])
        self.load_transforms = transforms.Compose(
            [mytransforms.LoadRGB(),
             mytransforms.LoadSegmentation(),
             ])
        self.model_path = model_path
        options = ERFnetOptions()
        opt = options.parse()
        self.model = ERFNet(self.num_classes_model, opt)
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(self.model_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        self.model.to(self.device)
    
    def visualize(self,file_path,save_path):
        image_name="_".join(file_path.split(os.sep)[-1].split(".")[0].split("_")[:-1])
        save_path = os.path.join(self.output_dir,f"gt_{image_name}.png")
        img=cv2.imread(file_path, -1)
        input_color = {("color", 0, -1): img}
        # img = Image.fromarray(img)
        input_color = self.load_transforms(input_color)
        input_color = self.transforms(input_color)

        input_color = {("color_aug", 0, 0): input_color[("color_aug", 0, 0)].unsqueeze(0).to(self.device)}
        pred_logits = self.model(input_color)['segmentation_logits'].float().squeeze(0)
        pred = F.softmax(pred_logits, dim=0).detach().unsqueeze(0)
        pred = F.interpolate(pred, (1024,2048), mode='nearest').squeeze(0)
        pred_seg = torch.argmax(pred,dim=0).cpu().numpy()
        pred_seg = Image.fromarray(pred_seg.astype('uint8'))
        # print(PALETTE)
        pred_seg.putpalette(PALETTE)
        pred_seg.save(save_path)

    def process_image(self,image_path:str):
        # image_path=os.path.join(DATASET_DIR,self.content_dict['img'][image_name])

        image_name="_".join(image_path.split(os.sep)[-1].split(".")[0].split("_")[:-1])

        # Assertions
        assert os.path.isfile(image_path), "Invalid image!"
        # Required image transformations
        resize_interp = transforms.Resize((512, 1024), interpolation=transforms.InterpolationMode.BILINEAR)
        transformer = transforms.ToTensor()
        normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        # Load Image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        native_image_size = image.size

        # Transform image
        image = resize_interp(image)
        image = transformer(image)
        image = normalize(image).unsqueeze(0).to(self.device)

        # Process image
        input_rgb = {("color_aug", 0, 0): image}
        output = self.model(input_rgb)

        # Process network output
        pred_seg = F.softmax(output['segmentation_logits'].float(),dim=1)
        # pred_seg = pred_seg[:, self.start_trainId:self.end_trainId+1, ...]
        pred_seg = F.interpolate(pred_seg, (native_image_size[1], native_image_size[0]), mode='nearest')
        # pred_seg[pred_seg<self.thres]=0
        pred_seg =torch.argmax(pred_seg,dim=1).squeeze().cpu().detach().numpy().astype('uint8')
        
        for i in self.trainid_to_labels.keys():
            pred_seg[pred_seg == i]=self.trainid_to_labels[i].id

        img=Image.fromarray(pred_seg)
        img.putpalette(PALETTE)
        img.save(os.path.join(self.output_dir,"seg_{}.png".format(image_name)))


def visualize(file_path,save_dir):
    image_name="_".join(file_path.split(os.sep)[-1].split(".")[0].split("_")[:-1])
    save_path = os.path.join(save_dir,f"gt_{image_name}.png")
    img=cv2.imread(file_path, -1)
    pred_seg = Image.fromarray(img.astype('uint8'))
    # print(PALETTE)
    pred_seg.putpalette(PALETTE)
    pred_seg.save(save_path)

if __name__ == "__main__":
    # filepath = 'Dataset/cityscapes/leftImg8bit/train/stuttgart/stuttgart_000101_000019_leftImg8bit.png'
    img_name = "kitti_000172_10"
    label_path = f"Dataset/kitti/gtFine/val/kitti/{img_name}_gtFine_labelIds.png"
    image_path = f"Dataset/kitti/leftImg8bit/val/kitti/{img_name}_leftImg8bit.png"
    
    outdir = './img/kitti_10_5_5'
    seg=Segmentation('Checkpoints/erfnet/erfnet_incremental_set12/models/kitti_10_5_5/exp_1/weights_199/model.pth',outdir,"kitti")
    # seg.visualize(label_path,'./img')
    visualize(label_path,outdir )
    seg.process_image(image_path)
    # visualize("Dataset/kitti/gtFine/train/kitti/kitti_000085_10_gtFine_labelIds.png","./kitti_000085_10_color.png")