from __future__ import absolute_import, division, print_function

import time

import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json
import os
import shutil
import random
from dataset_sampler import CityscapesDatasetGenerator, KittiDatasetGenerator
from dataset_util import Cityscapes, Kitti
from models.erfnet import ERFNet
from models.train.losses import *
import dataloader.pt_data_loader.mytransforms as mytransforms
from dataloader.pt_data_loader.specialdatasets import CityscapesDataset
from dataloader.file_io.get_path import GetPath
from dataloader.eval.metrics import SegmentationRunningScore
from dataloader.definitions.labels_file import *
from evaluate_erfnet import Evaluator
from src.options import ERFnetOptions
from src.city_set import CitySet
from shutil import copyfile

os.environ['PYTHONHASHSEED'] = '0'
seed = 1234
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # Romera
torch.cuda.manual_seed_all(seed)  # Romera
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self, options):

        print(" -> Executing script", os.path.basename(__file__))
        self.opt = options 
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda") 

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #                           LABELS AND CITIES
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        assert self.opt.train_set in {2, 3}, "Invalid train_set!"
        keys_to_load = ['color', 'segmentation'] 

        # Labels
        if self.opt.train_set == 2:
            if self.opt.dataset == "cityscapes":
                labels = labels_cityscape_seg_train2.getlabels()
                labels_eval = labels_cityscape_seg_train2_eval.getlabels()
                self.trainId_to_labels = labels_cityscape_seg_train2.gettrainid2label()
            elif self.opt.dataset == "kitti":
                labels = labels_kitti_seg_train2.getlabels()
                labels_eval = labels_kitti_seg_train2_eval.getlabels()
                self.trainId_to_labels = labels_kitti_seg_train2.gettrainid2label()
        elif self.opt.train_set == 3:
            if self.opt.dataset == "cityscapes":
                labels = labels_cityscape_seg_train3.getlabels()
                labels_eval = labels_cityscape_seg_train3_eval.getlabels()
                self.trainId_to_labels = labels_cityscape_seg_train3.gettrainid2label()
            elif self.opt.dataset == "kitti":
                raise Exception('kitti没有set3')
        

        classes_list=[]
        for i in range(len(labels)):
            if not labels[i].trainId == 255:
                classes_list.append(labels[i].name)

        # Train IDs
        self.train_ids = set([labels[i].trainId for i in range(len(labels))])
        self.train_ids.remove(255)

        # Num classes of teacher and student
        self.num_classes_teacher = min(self.train_ids)
        self.num_classes_student = max(self.train_ids) + 1  # +1 due to indexing starting at zero
        self.num_new_classes = self.num_classes_student - self.num_classes_teacher

        self.labels = labels

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #                           DATASET DEFINITIONS
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Data augmentation
        
        if self.opt.dataset == "cityscapes":
            self.dg=CityscapesDatasetGenerator(self.opt.train_set,classes_list,self.num_new_classes,self.opt.num_shot,self.opt.num_exp)
            if self.opt.pseudo_label_mode :
                self.cityscapes = Cityscapes()
                self.neighbors = self.cityscapes.get_neighbors(self.opt.train_set)
        elif self.opt.dataset == "kitti":
            self.dg=KittiDatasetGenerator(self.opt.train_set,classes_list,self.num_new_classes,self.opt.num_shot,self.opt.num_exp)
            if self.opt.pseudo_label_mode :
                self.kitti = Kitti()
                self.neighbors = self.kitti.get_neighbors(self.opt.train_set)

        self.train_data_transforms = [mytransforms.RandomHorizontalFlip(), 
                                 mytransforms.CreateScaledImage(),
                                 mytransforms.Resize((self.opt.height, self.opt.width), image_types=keys_to_load), 
                                 mytransforms.RandomRescale(1.5), 
                                 mytransforms.RandomCrop((self.opt.crop_height, self.opt.crop_width)),
                                 mytransforms.ConvertSegmentation(),
                                 mytransforms.CreateColoraug(new_element=True, scales=self.opt.scales),
                                 mytransforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
                                                          hue=0.1, gamma=0.0),
                                 mytransforms.RemoveOriginals(),
                                 mytransforms.ToTensor(), 
                                 mytransforms.NormalizeZeroMean(), 
                                 ]


        # val_data_transforms = [mytransforms.CreateScaledImage(),
        #                        mytransforms.Resize((self.opt.height, self.opt.width), image_types=keys_to_load),
        #                        mytransforms.ConvertSegmentation(),
        #                        mytransforms.CreateColoraug(new_element=True, scales=self.opt.scales),
        #                        mytransforms.RemoveOriginals(),
        #                        mytransforms.ToTensor(),
        #                        mytransforms.NormalizeZeroMean(),
        #                        ]

        # self.val_dataset = CityscapesDataset(dataset="cityscapes",
        #                                 trainvaltest_split="train",
        #                                 video_mode='mono',
        #                                 stereo_mode='mono',
        #                                 scales=self.opt.scales,
        #                                 labels_mode='fromid',
        #                                 labels=labels_eval,
        #                                 keys_to_load=keys_to_load,
        #                                 data_transforms=val_data_transforms,
        #                                 video_frames=self.opt.video_frames,
        #                                 folders_to_load=CitySet.get_city_set(-1))


        # self.val_loader = DataLoader(dataset=self.val_dataset,
        #                              batch_size=self.opt.batch_size,
        #                              shuffle=False,
        #                              num_workers=self.opt.num_workers,
        #                              pin_memory=True,
        #                              drop_last=False)


        # self.val_iter = iter(self.val_loader)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #                           LOGGING OPTIONS
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        print("++++++++++++++++++++++ INIT TRAINING ++++++++++++++++++++++++++")
        print("Using dataset:\n  ", self.opt.dataset, "with split", self.opt.dataset_split)
        # print("There are {:d} training items and {:d} validation items\n".format(
        #     len(train_dataset), len(val_dataset)))

        path_getter = GetPath()
        log_path = path_getter.get_checkpoint_path()
        self.log_path = os.path.join(log_path, 'erfnet', self.opt.model_name)

        self.writers = {}
        for mode in ["train", "validation"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        # Copy this file to log dir
        shutil.copy2(__file__, self.log_path)

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.log_path)
        print("Training is using:\n  ", self.device)
        print("Training takes place on train set:\n  ", self.opt.train_set)
        print("num_new_classes:  ", self.num_new_classes)
        print("num_shot:  ", self.opt.num_shot)
        print("num_exp: ", self.opt.num_exp)
        if self.opt.pseudo_label_mode:
            print("pseudo_label_mode: ", self.opt.pseudo_label_mode)
            print("num_pseudo_labels: ", self.opt.num_pseudo_labels)
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #                           LOSSES
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Ordinary cross-entropy loss

        self.crossentropy = CrossEntropyLoss(ignore_background=True,
                                             train_id_0=self.num_classes_teacher,
                                             device=self.device)
        self.crossentropy.to(self.device)

        # Knowledge distillation loss
        self.distillation = KnowledgeDistillationCELossWithGradientScaling(temp=self.opt.temp,
                                                                           device=self.device,
                                                                           gs=self.opt.lambda_GS,
                                                                           )
        self.distillation.to(self.device)

        self.metric_model = SegmentationRunningScore(self.num_classes_student)

        self.best_model = {
                            "exp":1,
                            'epoch':0,
                            "score":0
                            }





    def init_models(self):
        # # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # #                           MODEL DEFINITION
        # # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # # Instantiate and load teacher
        # self.teacher = ERFNet(self.num_classes_teacher, self.opt)
        # self.load_model(model=self.teacher,
        #                 model_name=self.opt.teachers[0],
        #                 adam=False,
        #                 epoch=int(self.opt.teachers[1]),
        #                 decoder=True)
        # self.teacher.eval()
        # self.teacher.to(self.device)

        # # Instantiate student
        # self.student = ERFNet(self.num_classes_student, self.opt)
        # self.student.to(self.device)
        # self.parameters_to_train = self.student.parameters()


        # Instantiate and load teacher
        self.teacher = ERFNet(self.num_classes_teacher, self.opt)
        if self.opt.load_best_model:
            self.load_model(model=self.teacher,
                        model_name=self.opt.teachers[0],
                        load_best=True,
                        adam=False,
                        decoder=True)

        elif len(self.opt.teachers) == 4:
            self.load_model(model=self.teacher,
                        model_name=self.opt.teachers[0],
                        adam=False,
                        epoch=int(self.opt.teachers[1]),
                        exp=int(self.opt.teachers[3]),
                        decoder=True)
        else:
            self.load_model(model=self.teacher,
                            model_name=self.opt.teachers[0],
                            adam=False,
                            epoch=int(self.opt.teachers[1]),
                            decoder=True)
        self.teacher.eval()

        self.teacher.to(self.device)

        # Instantiate student

        self.student = ERFNet(self.num_classes_student, self.opt)

        self.student.encoder.load_state_dict(self.teacher.encoder.state_dict())

        for name in self.student.decoder.state_dict().keys():
            if not ('output_conv' in name):
                self.student.decoder.state_dict()[name].copy_(self.teacher.decoder.state_dict()[name])
        self.student.to(self.device)
        self.parameters_to_train = self.student.parameters()

        # WT: Output convolution layer parameters copy
        self.student.decoder.state_dict()['output_conv.weight'][:,0:self.num_classes_teacher,:,:] = \
            self.teacher.decoder.state_dict()['output_conv.weight']
        self.student.decoder.state_dict()['output_conv.bias'][0:self.num_classes_teacher] = \
            self.teacher.decoder.state_dict()['output_conv.bias']


    def init_optimizer(self):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #                           OPTIMIZER SET-UP
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.model_optimizer = optim.Adam(params=self.parameters_to_train,
                                          lr=self.opt.learning_rate,
                                          weight_decay=self.opt.weight_decay)
        lambda1 = lambda epoch: pow((1 - ((epoch - 1) / self.opt.num_epochs)), 0.9)
        self.model_lr_scheduler = optim.lr_scheduler.LambdaLR(self.model_optimizer, lr_lambda=lambda1)
        
    def init_evaluator(self):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #                           EVALUATOR DEFINITION
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        if self.opt.validate:
            self.evaluator = Evaluator(self.opt, self.student)

    def set_train(self):
        """Convert all models to training mode
        """
        self.student.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        self.student.eval()
    


    def run_experiment(self,):

        self.current_exp=0

        for exp_id in range(self.opt.num_exp):
            self.current_exp = exp_id + 1

            filenames,train_dataset = self.dg.generate(self.labels,self.train_data_transforms,return_filenames=True)

            self.normal_train(train_dataset,filenames)
            if self.opt.pseudo_label_mode:
                self.train_with_pseudo_labels(filenames)

    def normal_train(self,train_dataset,filenames):
        # Save all options to disk and print them to stdout
        self.save_opts(len(train_dataset), 0,filenames)
        # self._print_options()

        self.init_models()

        self.init_optimizer()

        self.init_evaluator()

        self.train(train_dataset)

    def train_with_pseudo_labels(self,filenames):

        neighbor_list = []
        for img_name in filenames:
            neighbor_list += self.neighbors[img_name][:self.opt.num_pseudo_labels]
        self.student.eval()
        train_dataset = self.dg.generate_with_pseudo_label(model=self.student,
                                            filename_dict={
                                                'image':filenames,
                                                'pseudo_label':neighbor_list
                                            },
                                            trainId_to_labels=self.trainId_to_labels,
                                            labels=self.labels,
                                            data_transforms=self.train_data_transforms
                                            )

        # Save all options to disk and print them to stdout
        self.save_opts(len(train_dataset), 0,filenames)
        # self._print_options()

        self.init_models()

        self.init_optimizer()

        self.init_evaluator()

        self.train(train_dataset)


    def train(self, train_dataset):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()

        for self.epoch in range(self.opt.num_epochs):

            self.run_epoch(train_dataset)

            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

            if self.opt.validate and (self.epoch + 1) % self.opt.val_frequency == 0:
                self.run_eval()
        # if self.opt.save_best_model:
        self.save_model(is_best=True)

    def run_epoch(self, train_dataset):
        """Run a single epoch of training and validation
        """
        # print("Training")

        self.set_train()
        train_loader = DataLoader(dataset=train_dataset,
                            batch_size=self.opt.batch_size,
                            shuffle=True,
                            num_workers=self.opt.num_workers,
                            pin_memory=True,
                            drop_last=False)

        for batch_idx, inputs in enumerate(train_loader):
            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)


            self.model_optimizer.zero_grad()

            losses["loss"].backward()

            self.model_optimizer.step()
            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            # early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            # late_phase = self.step % 2000 == 0


            if ('segmentation', 0, 0) in inputs.keys():
                metrics = self.compute_segmentation_losses(inputs, outputs)
                self.log_time(batch_idx, duration, losses["loss"].cpu().data, metrics["meaniou"]
                                , metrics["meanacc"])
                model_score = (metrics["meaniou"] +  metrics["meanacc"])/2
                if model_score>=self.best_model['score'] and (self.epoch + 1) % self.opt.save_frequency == 0:
                    self.best_model['exp'] = self.current_exp
                    self.best_model['epoch'] = self.epoch
                    self.best_model['score'] = model_score
            else:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data, 0, 0)
                metrics = {}
            self.log("train", losses, metrics)
            # if early_phase or late_phase:
            #     # self.val()
            self.step += 1

        self.model_lr_scheduler.step()

    def run_eval(self):
        print("Validating on full validation set")
        self.set_eval()

        self.evaluator.calculate_metrics(self.epoch)

    # def val(self):
    #     """Validate the model on a single minibatch
    #     """
    #     self.set_eval()
    #     try:
    #         inputs_val = self.val_iter.next()
    #     except StopIteration:
    #         self.val_iter = iter(self.val_loader)
    #         inputs_val = self.val_iter.next()

    #     with torch.no_grad():
    #         outputs_val, losses_val = self.process_batch(inputs_val)

    #         if ('segmentation', 0, 0) in inputs_val:
    #             metrics_val = self.compute_segmentation_losses(inputs_val, outputs_val)
    #         else:
    #             metrics_val = {}

    #         self.log("validation", losses_val, metrics_val)

    #     self.set_train()

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """

        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        outputs_student = self.student(inputs)
        outputs_teacher = self.teacher(inputs)

        losses = self.compute_losses(inputs, outputs_student, outputs_teacher)

        return outputs_student, losses

    def compute_losses(self, inputs, outputs_student, outputs_teacher):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}

        # Targets for distillation and CE losses
        targets_old_classes = F.softmax(outputs_teacher['segmentation_logits'].float() / self.opt.temp, dim=1)
        targets_new_classes = inputs[('segmentation', 0, 0)][:, 0, :, :].long()

        # Response student for CE and DST loss
        outputs_student_all_classes = outputs_student['segmentation_logits']
        outputs_student_all_classes_ce = F.log_softmax(outputs_student_all_classes.float(), dim=1)
        outputs_student_all_classes_dst = F.log_softmax(outputs_student_all_classes.float() / self.opt.temp, dim=1)

        # Loss terms
        # 对新类类别使用交叉熵损失
        ce_loss = self.crossentropy(outputs_student_all_classes_ce[:, self.num_classes_teacher:, ...], targets_new_classes)
        # 对旧类类别使用知识蒸馏损失
        kd_loss = self.distillation(outputs=outputs_student_all_classes_dst[:, :self.num_classes_teacher, ...],
                                    targets=targets_old_classes, targets_new=targets_new_classes)

        # Total loss
        losses["loss"] = ce_loss + kd_loss
        losses["ce_loss"] = ce_loss
        losses["kd_loss"] = kd_loss

        return losses

    def compute_segmentation_losses(self, inputs, outputs):
        """Compute the loss metrics based on the current prediction
        """
        label_true = np.array(inputs[('segmentation', 0, 0)].cpu())[:, 0, :, :]
        label_pred = np.array(outputs['segmentation'].detach().cpu())
        self.metric_model.update(label_true, label_pred)
        metrics = self.metric_model.get_scores()
        self.metric_model.reset()
        return metrics

    def log_time(self, batch_idx, duration, loss, miou, acc):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        print_string = "expriment {:>3} | epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f}| meaniou: {:.5f}| meanacc: {:.5f}"
        print(print_string.format(self.current_exp,self.epoch, batch_idx, samples_per_sec, loss, miou, acc))

    def log(self, mode, losses, metrics):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)
        for l, v in metrics.items():
            if l in {'iou', 'acc', 'prec'}:
                continue
            writer.add_scalar("{}".format(l), v, self.step)

    def save_opts(self, n_train, n_eval,filenames=None):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models", "exp_"+str(self.current_exp))
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()
        to_save['n_train'] = n_train
        to_save['n_eval'] = n_eval
        if not filenames is None:
            to_save['filenames'] = filenames


        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self,is_best=False):
        """Save model weights to disk
        """
        if self.opt.pseudo_label_mode:
            model_dir = "{}_{}_{}_{}".format(self.opt.dataset, self.opt.num_exp, self.opt.num_shot, self.opt.num_pseudo_labels)
        else:
            model_dir = "{}_{}_{}".format(self.opt.dataset, self.opt.num_exp, self.opt.num_shot)

        if not is_best:
            save_folder = os.path.join(self.log_path, "models", model_dir,"exp_"+str(self.current_exp), "weights_{}".format(self.epoch))
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            save_path = os.path.join(save_folder, "{}.pth".format("model"))
            to_save = self.student.state_dict()
            torch.save(to_save, save_path)

            save_path = os.path.join(save_folder, "{}.pth".format("optim"))
            torch.save(self.model_optimizer.state_dict(), save_path)
        else:
            source_folder = os.path.join(self.log_path, "models", model_dir, "exp_"+str(self.best_model['exp']), "weights_{}".format(self.best_model["epoch"]))
            target_folder = os.path.join(self.log_path, "models", model_dir, "best_model")
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            for key in ["model","optim"]:
                copyfile(os.path.join(source_folder,"{}.pth".format(key)),os.path.join(target_folder,"{}.pth".format(key)))



    def load_model(self, adam=True, model=None, model_name=None, epoch=None, exp=None, decoder=True,load_best=False):
        """Load model(s) from disk
        :param adam: whether to load the Adam state too
        :param model: instance of ERFNet in which the model should be loaded
        :param model_name: name of the model to be loaded
        :param epoch: epoch of the model to be loaded
        :param decoder: whether to load the decoder too
        """
        base_path = os.path.split(self.log_path)[0]
        if self.opt.pseudo_label_mode:
            model_dir = "{}_{}_{}_{}".format(self.opt.dataset, self.opt.num_exp, self.opt.num_shot, self.opt.num_pseudo_labels)
        else:
            model_dir = "{}_{}_{}".format(self.opt.dataset, self.opt.num_exp, self.opt.num_shot)

        if load_best:
            checkpoint_path = os.path.join(base_path, model_name, 'models', model_dir, "best_model")
        elif exp is not None:
            checkpoint_path = os.path.join(base_path, model_name, 'models', model_dir, "exp_"+str(exp),
                                        'weights_{}'.format(epoch))
        else:
            checkpoint_path = os.path.join(base_path, model_name, 'models',
                                        'weights_{}'.format(epoch))
        
        assert os.path.isdir(checkpoint_path), \
            "Cannot find folder {}".format(checkpoint_path)
        print("loading model from folder {}".format(checkpoint_path))

        path = os.path.join(checkpoint_path, "{}.pth".format('model'))
        model_dict = model.state_dict()
        if self.opt.no_cuda:
            pretrained_dict = torch.load(path, map_location='cpu')
        else:
            pretrained_dict = torch.load(path)

        if decoder:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        else:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'encoder' == k[:7])}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        if adam:
            # loading adam state
            optimizer_load_path = os.path.join(checkpoint_path, "{}.pth".format("optim"))
            if os.path.isfile(optimizer_load_path):
                print("Loading Adam weights")
                optimizer_dict = torch.load(optimizer_load_path)
                self.model_optimizer.load_state_dict(optimizer_dict)
            else:
                print("Cannot find Adam weights so Adam is randomly initialized")

    def _print_options(self):
        """Print training options to stdout so that they appear in the SLURM log
        """
        # Convert namespace to dictionary
        opts = vars(self.opt)

        # Get max key length for left justifying
        max_len = max([len(key) for key in opts.keys()])

        # Print options to stdout
        print("+++++++++++++++++++++++++++ OPTIONS +++++++++++++++++++++++++++")
        for item in sorted(opts.items()):
            print(item[0].ljust(max_len), item[1])
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


if __name__ == "__main__":
    options = ERFnetOptions()
    opt = options.parse()

    # checking height and width are multiples of 32
    assert opt.height % 32 == 0, "'height' must be a multiple of 32"
    assert opt.width % 32 == 0, "'width' must be a multiple of 32"
    assert opt.video_frames[0] == 0, "frame_ids must start with 0"

    trainer = Trainer(options=opt)
    trainer.run_experiment()
