import cv2
import os


dataset_path = "../rt_gene/rt_gene_model_training/rt_gene_dataset"
left_patch_path  ="inpainted/left"
right_patch_path  ="inpainted/right"
file_type = '.png'
result_path = "./dataset/benchmark/face_test/HR"
os.makedirs(result_path, exist_ok=True)


for i in range(17):
    sub_path = os.path.join(dataset_path, f's{i:0>3}_glasses')

    left_file_path = os.path.join(sub_path, left_patch_path)
    right_file_path = os.path.join(sub_path, right_patch_path)

    left_file_list = os.listdir(left_file_path)
    right_file_list = os.listdir(right_file_path)

    for num in range(len(left_file_list)):
        left_file_name = os.path.join(left_file_path, left_file_list[num])
        right_file_name = os.path.join(right_file_path, right_file_list[num])


        left = cv2.imread(left_file_name)
        right = cv2.imread(right_file_name)

        cv2.imwrite(os.path.join(result_path, f's{i:0>3}_'+left_file_list[num]), left)
        cv2.imwrite(os.path.join(result_path, f's{i:0>3}_'+right_file_list[num]), right)
        # # os.rename(file_name, os.path.join(result_path,split[0]+file_type))
        print(left_file_list[num] + ' >> ' + '  Completed!')
