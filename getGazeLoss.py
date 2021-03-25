from rt_gent.gaze_estimation_models_pytorch import GazeEstimationModelVGG
import torch


 _loss_fn = {
            "mse": partial(torch.nn.MSELoss, reduction="sum")
        }
_param_num = {
    "mse": 2
}
_models = {
            "vgg": partial(GazeEstimationModelVGG, num_out=_param_num.get(hparams.loss_fn)) 
        }



def generateEyePatches():
    image_root_path = "./train_SRImage"

    landmark_estimator = LandmarkMethodBase(device_id_facedetection="cuda:0",
                                            checkpoint_path_face=os.path.join(script_path, "/../rt_gene/rt_gene/model_nets/SFD/s3fd_facedetector.pth"),
                                            checkpoint_path_landmark=os.path.join(script_path, "/../rt_gene/rt_gene/model_nets/phase1_wpdc_vdc.pth.tar"),
                                            model_points_file=os.path.join(script_path, "/../rt_gene/rt_gene/model_nets/face_model_68.txt"))

    image_path = os.path.join(image_root_path, "SRImage")
    

    left_path = os.path.join(image_root_path, "left")
    right_path = os.path.join(image_root_path, "right")

    os.makedirs(left_path, exist_ok = True)
    os.makedirs(right_path, exist_ok = True)

    image_list = os.listdir(image_path)

    for image_file_name in image_list:
        image = cv2.imread(os.path.join(image_path, image_file_name))
        if image is None:
            continue

        faceboxes = landmark_estimator.get_face_bb(image)
        if len(faceboxes) == 0:
            continue

        subjects = landmark_estimator.get_subjects_from_faceboxes(image, faceboxes)
        for subject in subjects:
            le_c, re_c, _, _ = subject.get_eye_image_from_landmarks(subject.transformed_eye_landmarks, subject.face_color, landmark_estimator.eye_image_size)

            if le_c is not None and re_c is not None:
                img_name = image_file_name.split(".")[0]
                left_image_name = "left_"+ img_name+".png"
                left_image_path = os.path.join(left_path, left_image_name)

                right_image_name = "right_"+ img_name+ ".png"
                right_image_path = os.path.join(right_path, right_image_name)

                cv2.imwrite(left_image_path, le_c)
                cv2.imwrite(right_image_path, re_c)

def computeGazeLoss(labels):
    _model = _models.get("vgg")()
    _criterion = _loss_fn.get("mse")()


     # load image
    image_root_path = "./train_SRImage"

    left_eye_path = os.path.join(image_root_path, left_path)
    rigth_eye_path = os.path.join(image_root_path, right_path)
    left_list = os.listdir(left_eye_path)
    right_list =os.listdir(rigth_eye_path)

    #load Label
    gaze_batch_label = []
    head_batch_label = []
    head_batch_label, gaze_batch_label =loadLabel(labels)


    angular_out = _model(left_list, right_list, head_batch_label)
    gaze_loss = _criterion(angular_out, gaze_batch_label)

    return gaze_loss



def loadLabel(labels):
    gaze_batch_label = []
    head_batch_label = []
    flag = False

    #load label
    for image_name in image_names:
        subj = image_name.split('_')[0]
        file_num = image_name.split('_')[1]

        start = 0
        end = len(labels) -1
        mid = (start + end) //2

        #이진 탐색
        #label : idx, head1, head2, gaze1, gaze2, time
        #image name 형식: s000_1
        while end -start >=0:
            label = labels[mid].split(",")
            if label[0] == image_name:
                head_batch_label.append([label[1],label[2]])
                gaze_batch_label.append([label[3],label[4]])
                Flag =True
                break
            elif label[0] < image_name:
                start = mid+1
            else:
                end = mid-1
        if not flag:
            print("ERROR:: label not found")
            sys.exit()
        flag =True

    return head_batch_label, gaze_batch_label


def generateH5Dataset():
    # script_path = os.path.dirname(os.path.realpath(__file__))
            # _required_size = (224, 224)
            # subject_path ="./SRImage_to_h5/SR_Image"
            # h5_root = "./SRImage_to_h5/h5"

            # hdf_file = h5py.File(os.path.join(h5_root, 'batch_SR.hdf5')), mode='w')