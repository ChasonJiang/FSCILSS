import os

from tqdm import tqdm
from dataset_util import Cityscapes, Kitti
import random
from dataloader.pt_data_loader.specialdatasets import cityscapes_simple_Dataset
from dataloader.definitions.labels_file import *
import dataloader.pt_data_loader.mytransforms as mytransforms
from torch.utils.data import DataLoader
import cv2
import random
import json
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import PIL.Image as pil
from PIL import Image
from models.erfnet import ERFNet
seed = 1234
random.seed(seed)


class DatasetGenerator():
    '''
        该类用于生成few shot incremental的dataset
    '''
    def __init__(self,train_set,classes_list,num_classes,num_shot,num_expriment,dataset_dir,dataset):
        self.num_classes = num_classes
        self.num_shot = num_shot
        self.num_expriment = num_expriment
        self.dataset_dir=dataset_dir
        # self.work_dir = os.path.join(DATASET_DIR,'Extra')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        if dataset == "kitti":
            self.datasetObj = Kitti(dataset_dir)
        elif dataset == "cityscapes":
            self.datasetObj = Cityscapes(dataset_dir)

        # filename = num_classes+"_way_"+num_shot+"_shot"
        sample_template={}
        for key in classes_list:
            sample_template[key] = num_shot*num_expriment
        self.index_dict = self.datasetObj.sample_from_sorted(train_set,sample_template=sample_template,save_to_disk=False)
        self.content_dict = self.datasetObj.get_trainset(train_set)
    
    def generate(self,labels,data_transforms,return_filenames=False):
        imgname_list=[]

        for key in self.index_dict.keys():
            filename_list = self.index_dict[key]
            random.shuffle(filename_list)
            imgname_list.extend(filename_list[:self.num_shot])
            self.index_dict[key]  = filename_list[self.num_shot:]

        random.shuffle(imgname_list)

        dataset_dict={}

        gt_list=[]
        img_list=[]

        for filename in imgname_list:
            gt_list.append(self.content_dict['gt'][filename])
            img_list.append(self.content_dict['img'][filename])
        
        dataset_dict['gt']=gt_list
        dataset_dict['img']=img_list
        if return_filenames:
            return imgname_list,cityscapes_simple_Dataset(data_dict=dataset_dict,labels=labels,data_transforms=data_transforms,dataset=self.dataset)
        else:
            return cityscapes_simple_Dataset(data_dict=dataset_dict,labels=labels,data_transforms=data_transforms,dataset=self.dataset)


    def generate_with_pseudo_label(self, model, filename_dict, trainId_to_labels, labels, data_transforms,thres = 0.6):
        assert model is not None
        assert isinstance(trainId_to_labels,dict)
        assert isinstance(filename_dict,dict)
        self.thres = thres
        self.trainId_to_labels = trainId_to_labels
        self.prediction_model = model
        self.pseudo_label_dir = os.path.join(self.dataset_dir,"Extra","PseudoLabels")
        if not os.path.exists(self.pseudo_label_dir):
            os.makedirs(self.pseudo_label_dir)

        
        dataset_dict={}
        img_list = []
        gt_list = []
        print("-> Pseudo-labels being predicted")
        bar =tqdm(filename_dict['pseudo_label'])
        for image_name in bar:
            self.process_image(image_name)
            city = image_name.split("_")[0]
            img_list.append(os.path.join('leftImg8bit','train',city,"{}_leftImg8bit.png".format(image_name)))
            gt_list.append(os.path.join("Extra","PseudoLabels","{}.png".format(image_name)))
        

        for image_name in filename_dict["image"]:
            city = image_name.split("_")[0]
            img_list.append(os.path.join('leftImg8bit','train',city,"{}_leftImg8bit.png".format(image_name)))
            gt_list.append(os.path.join("gtFine",'train',city,"{}_gtFine_labelIds.png".format(image_name)))

        dataset_dict['gt']=gt_list
        dataset_dict['img']=img_list
        print("-> Done")

        return cityscapes_simple_Dataset(data_dict=dataset_dict,labels=labels,data_transforms=data_transforms,dataset=self.dataset)


    def process_image(self,image_name):
        image_path=os.path.join(self.dataset_dir,self.content_dict['img'][image_name])
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
        output = self.prediction_model(input_rgb)

        # Process network output
        pred_seg = F.softmax(output['segmentation_logits'].float(),dim=1)
        # pred_seg = pred_seg[:, self.start_trainId:self.end_trainId+1, ...]
        pred_seg = F.interpolate(pred_seg, (native_image_size[1], native_image_size[0]), mode='nearest')
        # pred_seg[pred_seg<self.thres]=0
        pred_seg =torch.argmax(pred_seg,dim=1).squeeze().cpu().detach().numpy().astype('uint8')
        
        for i in self.trainId_to_labels.keys():
            pred_seg[pred_seg == i]=self.trainId_to_labels[i].id

        img=Image.fromarray(pred_seg)
        img.save(os.path.join(self.pseudo_label_dir,"{}.png".format(image_name)))



class CityscapesDatasetGenerator(DatasetGenerator):
    def __init__(self,train_set,classes_list,num_classes,num_shot,num_expriment,dataset_dir=os.path.join("./Dataset",'cityscapes')):
        super(CityscapesDatasetGenerator, self).__init__(train_set,classes_list,num_classes,num_shot,num_expriment,dataset_dir,"cityscapes")


class KittiDatasetGenerator(DatasetGenerator):
    def __init__(self,train_set,classes_list,num_classes,num_shot,num_expriment,dataset_dir=os.path.join("./Dataset",'kitti')):
        super(KittiDatasetGenerator, self).__init__(train_set,classes_list,num_classes,num_shot,num_expriment,dataset_dir,"kitti")



if __name__=="__main__":

    labels = labels_kitti_seg_train2.getlabels()
    labels_eval = labels_kitti_seg_train2_eval.getlabels()
    trainId_to_labels = labels_kitti_seg_train2.gettrainid2label()

    

    # 把labels里面的类别添加到classes_list
    classes_list=[]
    for i in range(len(labels)):
        if not labels[i].trainId == 255:
            classes_list.append(labels[i].name)

    # Train IDs
    train_ids = set([labels[i].trainId for i in range(len(labels))])
    train_ids.remove(255)

    # Num classes of teacher and student
    # 从train id中确定teacher model和student model要训练的类别数量
    num_classes_teacher = min(train_ids)
    num_classes_student = max(train_ids) + 1  # +1 due to indexing starting at zero
    # 确定本次类别增量的新类类别数量
    num_new_classes = num_classes_student - num_classes_teacher

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #                           DATASET DEFINITIONS
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Data augmentation
    
    dg=KittiDatasetGenerator(2,classes_list,num_new_classes,1,10)
    # dg=CityscapesDatasetGenerator(2,classes_list,num_new_classes,1,10)

    train_data_transforms = [mytransforms.RandomHorizontalFlip(),
                            mytransforms.CreateScaledImage(),
                            mytransforms.Resize((1024, 512), image_types=['color', 'segmentation']),
                            mytransforms.RandomRescale(1.5),
                            mytransforms.RandomCrop((640, 192)),
                            mytransforms.ConvertSegmentation(),
                            mytransforms.CreateColoraug(new_element=True, scales=[0]),
                            mytransforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
                                                    hue=0.1, gamma=0.0),
                            mytransforms.RemoveOriginals(),
                            mytransforms.ToTensor(),
                            mytransforms.NormalizeZeroMean(),
                            ]
    _,train_dataset=dg.generate(labels,train_data_transforms,True)


    train_loader= DataLoader(dataset=train_dataset,
                                       batch_size=6,
                                       shuffle=True,
                                       num_workers=0,
                                       pin_memory=True,
                                       drop_last=False)

    for batch_idx, inputs in enumerate(train_loader):
        print(batch_idx)




