from rt_gene.gaze_estimation_models_pytorch import GazeEstimationModelVGG
import torch
from functools import partial
from rt_gene.extract_landmarks_method_base import LandmarkMethodBase
import os
import sys
import cv2
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn


_loss_fn = {
            "mse": partial(torch.nn.MSELoss, reduction="sum")
        }
_param_num = {
    "mse": 2
}
ckpt = "rt_gene/model_nets/Alldata_1px_all_epoch=5-val_loss=0.551.model"
_model = GazeEstimationModelVGG(num_out =2)
_model.load_state_dict(torch.load(ckpt))

image_root_path = "./train_SRImage"
left_path = os.path.join(image_root_path, "left","l")
right_path = os.path.join(image_root_path, "right","r")
face_path =os.path.join(image_root_path, "face")

os.makedirs(left_path, exist_ok = True)
os.makedirs(right_path, exist_ok = True)
os.makedirs(face_path, exist_ok = True)

def clearDir():
        face_list = os.listdir(face_path)
        left_list = os.listdir(left_path)
        right_list = os.listdir(right_path)

        for face in face_list:
            _face = os.path.join(face_path,face)
            os.remove(_face)
        for i in range(len(left_list)):
            _left = os.path.join(left_path,left_list[i])
            _right = os.path.join(right_path,right_list[i])
            os.remove(_left)
            os.remove(_right)

def generateEyePatches():
    landmark_estimator = LandmarkMethodBase(device_id_facedetection="cuda:0",
                                            checkpoint_path_face= "rt_gene/model_nets/SFD/s3fd_facedetector.pth",
                                            checkpoint_path_landmark="rt_gene/model_nets/phase1_wpdc_vdc.pth.tar",
                                            model_points_file="rt_gene/model_nets/face_model_68.txt")

    image_list = os.listdir(face_path)
    for image_file_name in image_list:
        image = cv2.imread(os.path.join(face_path, image_file_name))
        if image is None:
            continue

        # faceboxes = landmark_estimator.get_face_bb(image)
        faceboxes = [[0,0,223,223]]
        if len(faceboxes) == 0:
            continue

        subjects = landmark_estimator.get_subjects_from_faceboxes(image, faceboxes)
        for subject in subjects:
            le_c, re_c, _, _,_,_ = subject.get_eye_image_from_landmarks(subject, landmark_estimator.eye_image_size)

            if le_c is not None and re_c is not None:
                img_name = image_file_name.split(".")[0]
                left_image_name = "left_"+ img_name+".png"
                left_image_path = os.path.join(left_path, left_image_name)

                right_image_name = "right_"+ img_name+ ".png"
                right_image_path = os.path.join(right_path, right_image_name)

                cv2.imwrite(left_image_path, le_c)
                cv2.imwrite(right_image_path, re_c)

def computeGazeLoss(labels):
    _criterion = _loss_fn.get("mse")()

    # load image
    left_root = os.path.join(image_root_path,"left")
    right_root = os.path.join(image_root_path,"right")

    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    left_set = datasets.ImageFolder(left_root, transform = transform)
    right_set = datasets.ImageFolder(right_root, transform = transform)

    left_dataloader = torch.utils.data.DataLoader(left_set, batch_size=len(left_set), shuffle=False)
    right_dataloader = torch.utils.data.DataLoader(right_set, batch_size=len(right_set), shuffle=False)

    left_list, _= next(iter(left_dataloader))
    right_list,_ = next(iter(right_dataloader))

    left_list = left_list.requires_grad_(True)
    right_list = right_list.requires_grad_(True)
      
    
    #load Label
    left_names = os.listdir(left_path)
    head_batch_label, gaze_batch_label =loadLabel(labels,left_names)
    angular_out = _model(left_list, right_list, head_batch_label)
    # print("angular_out :",angular_out)
    # diff =0
    # for i in range(len(angular_out)):
    #     for j in range(len(angular_out[i])):
    #         diff += pow((angular_out[i][j]-gaze_batch_label[i][j]),2)

    # print("sse  : ",diff)
    gaze_loss = _criterion(angular_out, gaze_batch_label).cuda()

    return gaze_loss



def loadLabel(labels,names):
    gaze_batch_label = []
    head_batch_label = []
    flag = False

    #load label
    for name in names:
        #image name 형식: left_s000_1
        subj = name.split('_')[1]
        file_num = name.split('_')[2].split('.')[0]
        file_num = f'{file_num:0>6}'
        image_name = subj+'_'+file_num
        
        
        #이진 탐색
        #label : idx, head1, head2, gaze1, gaze2, time
        start = 0
        end = len(labels) -1
        while start <=end:
            mid = (start + end) // 2
            label = labels[mid].split(",")
            name_label = label[0].split("_")
            curr_name = f'{name_label[0]}_{name_label[1]:0>6}'
            if curr_name == image_name:
                head_batch_label.append([float(label[1]),float(label[2])])
                gaze_batch_label.append([float(label[3]),float(label[4])])
                flag =True
                break
            elif curr_name < image_name:
                start = mid+1
            else:
                end = mid-1

        if not flag:
            print("ERROR:: label not found")
            sys.exit()
        flag =False

    head_batch_label= torch.FloatTensor(head_batch_label)
    gaze_batch_label = torch.FloatTensor(gaze_batch_label)

    return head_batch_label, gaze_batch_label


# def generateH5Dataset():
    # script_path = os.path.dirname(os.path.realpath(__file__))
    # _required_size = (224, 224)
    # subject_path ="./SRImage_to_h5/SR_Image"
    # h5_root = "./SRImage_to_h5/h5"

    # hdf_file = h5py.File(os.path.join(h5_root, 'batch_SR.hdf5')), mode='w')