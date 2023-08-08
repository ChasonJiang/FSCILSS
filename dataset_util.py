import os
import shutil
import cv2
import json
import torch
import torchvision
import numpy as np
# import pylab as plt
from PIL import Image
from tqdm import tqdm
import torchextractor as tx
from torch.nn import functional
from collections import Counter
from torchvision import transforms
from dataloader.definitions.labels_file import *
from src.city_set import CitySet

DATASET_DIR = os.path.join("./Dataset",'cityscapes')


class resnet50_feature_extractor():
    def __init__(self, transform=transforms.Compose([transforms.ToTensor()]), dataset_dir=DATASET_DIR):
        self.dataset_dir = dataset_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
        self.transform = transform
        self.save_dir = os.path.join(self.dataset_dir,'Extra','ResnetFeature')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)


    def load_model(self):
        self.model = torchvision.models.resnet50(pretrained=True).to(self.device)
        self.model = tx.Extractor(self.model, ["avgpool"])

    def extract(self,image):
        _, features = self.model(image)
        return features["avgpool"]

    def feature_extract(self, dataset_dic):
        '''
            dataset_dic like this:
            {
                <city 1>:{
                        <image_name 1>:<image_path 1>,
                        ...
                        <image_name N>:<image_path N>,
                        },
                ...
                <city N>:{
                        <image_name 1>:<image_path 1>,
                        ...
                        <image_name N>:<image_path N>,
                        },

            }
        '''
        assert isinstance(dataset_dic,dict)
        print("-> Resnet50 feature being extracted")
        dic={}
        bar =tqdm(dataset_dic.keys())
        for city in bar:
            for image_name in dataset_dic[city].keys():
                image = Image.open(os.path.join(self.dataset_dir,dataset_dic[city][image_name]))
                image = self.transform(image).unsqueeze(0).float()
                image = image.to(self.device)
                features = self.extract(image=image)
                dic[image_name]=os.path.join('Extra','ResnetFeature',"{}.pth".format(image_name))
                torch.save(features.squeeze().cpu().detach(),os.path.join(self.save_dir,"{}.pth".format(image_name)))

        with open(os.path.join(self.dataset_dir,'Extra','resnet50_feature.json'),'w') as f:
            f.write(json.dumps(dic))

        print('->Done')


class Cityscapes():
    '''
        该类会对Cityscapes数据集进行一些操作，包括但不限于：建立整个数据集的图像名称与其路径的json文件等等
    '''
    def __init__(self,dataset_dir=DATASET_DIR):
        self.dataset_dir = dataset_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        save_dir=os.path.join(self.dataset_dir,"Extra")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.all_set=CitySet.get_city_set(-2)
        if not os.path.exists(os.path.join(self.dataset_dir,"Extra","train.json")) and \
            not os.path.exists(os.path.join(self.dataset_dir,"Extra","val.json")):
            self.init_dataset()

        self.train_set_group_by_city,self.val_set_group_by_city = self.load_dataset()
        self.train_set_group_by_set = {}
        for set in self.all_set.keys():
            dic={
                "img":{},
                "gt":{}
            }
            for city in self.all_set[set]:
                for key in ["img",'gt']:
                    dic[key].update(self.train_set_group_by_city[key][city])
            self.train_set_group_by_set[set]=dic

        neighbors_path=os.path.join(self.dataset_dir,'Extra',"neighbors.json")
        if not os.path.exists(neighbors_path):
            self.neighbors=self.build_neighbor()
        else:
            self.neighbors=self.load_json(neighbors_path)



    def init_dataset(self,):
        for set in ["train","val"]:
            dataset={}
            gt_dic = {}
            img_dic ={}
            for item in self.findAllFile(os.path.join(self.dataset_dir,"gtFine",set)):
                file_name = item.split(os.sep)[-1].split(".")[0]
                file_type = file_name.split("_")[-1]
                file_name = "_".join(file_name.split("_")[:3])
                city = file_name.split("_")[0]
                if not city in img_dic.keys():
                    img_dic[city] = {}
                    gt_dic[city] = {}
                if file_type == "labelIds":
                    gt_dic[city][file_name] = os.path.join("gtFine",set,city,file_name+"_gtFine_labelIds.png")
                    img_dic[city][file_name] = os.path.join("leftImg8bit",set,city,file_name+"_leftImg8bit.png")
            dataset["gt"] = gt_dic
            dataset["img"] = img_dic
            self.write_json(dataset,os.path.join(self.dataset_dir,"Extra",set+".json"))

    def findAllFile(self,base):
        for root, ds, fs in os.walk(base):
            for f in fs:
                fullname = os.path.join(root, f)
                yield fullname


    def load_dataset(self,):
        with open(os.path.join(self.dataset_dir,"Extra","train.json"),"r") as file:
            train_set=json.loads(file.read())
        with open(os.path.join(self.dataset_dir,"Extra","val.json"),"r") as file:
            val_set=json.loads(file.read())

        return train_set,val_set

    def load_json(self,path):
        with open(path,"r") as file:
            dic=json.loads(file.read())
        return dic
    
    def write_json(self,dic,path):
        dic = json.dumps(dic)
        with open(path,"w") as file:
            file.write(dic)


    def sort_by_pixel_for_each_classes(self,save_to_disk=True):
        '''
            该函数将从整个training set中，按每张图片中各自类别的像素数量进行递减排列。
            output：
            {
                1:{
                        <classes>:[<img_name_1>,..,<img_name_N>],
                        ...
                        },
                2:{
                        <classes>:[<img_name_1>,..,<img_name_N>],
                        ...
                        },
                3:{
                        <classes>:[<img_name_1>,..,<img_name_N>],
                        ...
                        },

            }
        
        '''
        save_path=os.path.join(self.dataset_dir,"Extra","train_sorted.json")

        label =labels_cityscape_seg.getlabels()


        gt_dic = self.train_set_group_by_city["gt"]

        dic={}

        
        bar1=tqdm(self.all_set.keys())
        for set in bar1:
            cities=self.all_set[set]
            img_freq=[]
            content={}
            bar2 = tqdm(cities)
            for city in bar2:
                for filename in gt_dic[city].keys():
                    img_path=os.path.join(self.dataset_dir,gt_dic[city][filename])
                    img = cv2.imread(img_path, -1)
                    index,counts = np.unique(img,return_counts=True)
                    freq = np.zeros(34,dtype=np.uint64)

                    for id,i in enumerate(index):
                        freq[i]=counts[id]
                    img_freq.append([filename,freq])

            bar3 = tqdm(range(34))
            
            for i in bar3:
                img_freq.sort(key=lambda item:item[1][i],reverse=True)
                content[label[i].name]=[item[0] for item in img_freq]
            dic[set]=content

        if save_to_disk:
            with open(save_path,'w') as file:
                file.write(json.dumps(dic))
        return dic



    def sample_from_sorted(self,train_set,sample_template,save_to_disk=False,filename=None):
        '''
            该函式从已排序的数据集中，按sample_template抽样，并返回一个字典
            sample_template is a dictionary
            like this:
            {
                <classes>:num_samples,
                ...
            }
            eg:
            {
                "building":100,
                "person":100,
                ...
            }

            output：
            {
                <classes>:[<img_name_1>,..,<img_name_N>],
                ...
            }
        '''
        assert isinstance(train_set,int) or isinstance(train_set,str)
        if isinstance(train_set,int):
            train_set = str(train_set)
        sorted_file_path=os.path.join(self.dataset_dir,"Extra","train_sorted.json")
        if not os.path.exists(sorted_file_path):
            sorted_dataset = self.sort_by_pixel_for_each_classes()
        else:
            sorted_dataset = self.load_json(sorted_file_path)

        save_dir=os.path.join(self.dataset_dir,"Extra","Subset")
        if os.path.exists(save_dir):
            os.mkdir(save_dir)


        output = {}
        for key in sample_template.keys():
            output[key] = sorted_dataset[train_set][key][:sample_template[key]]

        if save_to_disk :
            assert isinstance(filename,str)
            self.write_json(output,os.path.join(save_dir,filename+".json"))
        
        
        return output

    def build_neighbor(self,):
        resnet_json_path=os.path.join(self.dataset_dir,'Extra','resnet50_feature.json')
        if not os.path.exists(resnet_json_path):
            rfe=resnet50_feature_extractor(dataset_dir=self.dataset_dir)
            rfe.feature_extract(self.train_set_group_by_city['img'])

        resnet_file=self.load_json(resnet_json_path)
        
        print("-> Building neighbor")
        neighbor_dic={}
        for set in self.train_set_group_by_set.keys():
            bar = tqdm(self.train_set_group_by_set[set]["img"].keys())
            dic={}
            for target_name in bar:
                target_features = torch.load(os.path.join(self.dataset_dir,resnet_file[target_name])).to(self.device)
                score_list = []
                image_name_list = list(self.train_set_group_by_set[set]['img'].keys())
                image_name_list.remove(target_name)
                for img_name in image_name_list:
                    img_features = torch.load(os.path.join(self.dataset_dir,resnet_file[target_name])).to(self.device)
                    score_list.append((img_name,functional.cosine_similarity(target_features,img_features,dim=0)))
                target_neighbor=[item[0] for item in sorted(score_list,key=lambda x:x[1],reverse=True)]
                dic[target_name]=target_neighbor
            neighbor_dic[set]=dic

        self.write_json(neighbor_dic,os.path.join(self.dataset_dir,'Extra',"neighbors.json"))
        print("-> Done")

        '''
            output like this:
            {
                1:{
                    <image_name 1>:[ neighbor list ],
                    ...
                    <image_name N>:[ neighbor list ],
                },
                2:{
                    <image_name 1>:[ neighbor list ],
                    ...
                    <image_name N>:[ neighbor list ],
                },
                3:{
                    <image_name 1>:[ neighbor list ],
                    ...
                    <image_name N>:[ neighbor list ],
                },
            }
        '''
        return neighbor_dic


    def get_neighbors(self,set):
        assert isinstance(set,int) or isinstance(set,str)
        if isinstance(set,int):
            set = str(set)
        return self.neighbors[set]

    def get_trainset(self,set):
        assert isinstance(set,int) or isinstance(set,str)
        if isinstance(set,int):
            set = str(set)
        return self.train_set_group_by_set[set]


class Kitti(Cityscapes):
    def __init__(self,dataset_dir=os.path.join("./Dataset",'kitti')):
        self.dataset_dir = dataset_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not os.path.exists(os.path.join(self.dataset_dir,"transformated")):
            self.transformat()
        if not os.path.exists(os.path.join(self.dataset_dir,"splited")):
            self.split_set()
        save_dir=os.path.join(self.dataset_dir,"Extra")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.all_set={"2":["kitti"]}
        if not os.path.exists(os.path.join(self.dataset_dir,"Extra","train.json")) and \
            not os.path.exists(os.path.join(self.dataset_dir,"Extra","val.json")):
            self.init_dataset()

        self.train_set_group_by_city,self.val_set_group_by_city = self.load_dataset()
        self.train_set_group_by_set = {}
        for set in self.all_set.keys():
            dic={
                "img":{},
                "gt":{}
            }
            for city in self.all_set[set]:
                for key in ["img",'gt']:
                    dic[key].update(self.train_set_group_by_city[key][city])
            self.train_set_group_by_set[set]=dic

        neighbors_path=os.path.join(self.dataset_dir,'Extra',"neighbors.json")
        if not os.path.exists(neighbors_path):
            self.neighbors=self.build_neighbor()
        else:
            self.neighbors=self.load_json(neighbors_path)

    def transformat(self):
        print("正在转换数据集格式...")
        resize_interp = transforms.Resize((375, 1242), interpolation=transforms.InterpolationMode.BILINEAR)
        resize_nearest = transforms.Resize((375, 1242), interpolation=transforms.InterpolationMode.NEAREST)

        source_path=os.path.join(self.dataset_dir,"source")
        os.makedirs(source_path)
        shutil.move(os.path.join(self.dataset_dir,"train"),source_path)
        shutil.move(os.path.join(self.dataset_dir,"test"),source_path)
        
        os.makedirs(os.path.join(self.dataset_dir,"leftImg8bit","train","kitti"))
        os.makedirs(os.path.join(self.dataset_dir,"leftImg8bit","test","kitti"))
        os.makedirs(os.path.join(self.dataset_dir,"gtFine","train","kitti"))

        trainset_path=os.path.join(source_path,"train", "image_2")

        for image_path in self.findAllFile(trainset_path):
            image_name = image_path.split(os.sep)[-1].split(".")[0]
            
            image = Image.open(image_path)
            image = resize_interp(image)
            image.save(os.path.join(self.dataset_dir,"leftImg8bit","train","kitti",
                 "kitti_{}_leftImg8bit.png".format(image_name)))
            # shutil.copyfile(image_path,os.path.join(self.dataset_dir,"leftImg8bit","train","kitti",
            #     "kitti_{}_leftImg8bit.png".format(image_name)))

        trainset_gt_path=os.path.join(source_path,"train", "semantic")

        for image_path in self.findAllFile(trainset_gt_path):
            image_name = image_path.split(os.sep)[-1].split(".")[0]
            image = Image.fromarray(cv2.imread(image_path, -1))
            image = resize_nearest(image)
            image.save(os.path.join(self.dataset_dir,"gtFine","train","kitti",
                 "kitti_{}_gtFine_labelIds.png".format(image_name)))
            # shutil.copyfile(image_path,os.path.join(self.dataset_dir,"gtFine","train","kitti",
            #     "kitti_{}_gtFine_labelIds.png".format(image_name)))

        test_path=os.path.join(source_path,"test", "image_2")

        for image_path in self.findAllFile(test_path):
            image_name = image_path.split(os.sep)[-1].split(".")[0]
            image = Image.open(image_path)
            image = resize_interp(image)
            image.save(os.path.join(self.dataset_dir,"leftImg8bit","test","kitti",
                 "kitti_{}_leftImg8bit.png".format(image_name)))
            # shutil.copyfile(image_path,os.path.join(self.dataset_dir,"leftImg8bit","test","kitti",
            #     "kitti_{}_leftImg8bit.png".format(image_name)))

        with open(os.path.join(self.dataset_dir,"transformated"),"w") as file:
            file.write("")

        print("转换完成...")

    def split_set(self, split={"train":100,"val":100}):
        assert isinstance(split,dict)

        num_train = split["train"]
        num_val = split["val"]
        
        assert num_train + num_val == 200
        print("开始划分数据集...")

        image_name_template = "kitti_{:0>6d}_10_leftImg8bit.png"
        gt_name_template = "kitti_{:0>6d}_10_gtFine_labelIds.png"
        trainset_image_path = os.path.join(self.dataset_dir,"leftImg8bit","train","kitti")
        trainset_gt_path = os.path.join(self.dataset_dir,"gtFine","train","kitti")
        valset_image_path = os.path.join(self.dataset_dir,"leftImg8bit","val","kitti")
        valset_gt_path = os.path.join(self.dataset_dir,"gtFine","val","kitti")
        if not os.path.exists(valset_image_path):
            os.makedirs(valset_image_path)
        if not os.path.exists(valset_gt_path):
            os.makedirs(valset_gt_path)



        for i in range(100,200,1):
            shutil.move(os.path.join(trainset_image_path, image_name_template.format(i)),
                            valset_image_path)
            shutil.move(os.path.join(trainset_gt_path, gt_name_template.format(i)),
                            valset_gt_path)

        with open(os.path.join(self.dataset_dir,"splited"),"w") as file:
            file.write("")
        print("完成...")







    
if __name__ == '__main__':
    # cityscapes=Cityscapes()
    # # cityscapes.sort_by_pixel_for_each_classes()
    # # cityscapes.build_neighbor()
    # cityscapes.get_neighbors(1)
    # cityscapes.get_neighbors("1")

    kitti = Kitti()