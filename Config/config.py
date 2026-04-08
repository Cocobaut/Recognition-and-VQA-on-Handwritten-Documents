#Config data for train and test
Dataset_Image_Train = "dataset_project/train_data/images"
Dataset_Image_Test = "dataset_project/test_data/images"

Dataset_Json_Train_Task_1 = "dataset_project/train_data/task_1"
Dataset_Json_Train_Task_2 = "dataset_project/train_data/task_2"
Dataset_Json_Train_Task_3 = "dataset_project/train_data/task_3"
Dataset_Json_Train_Task_4 = "dataset_project/train_data/task_4"

Dataset_Json_Test_Task_1 = "dataset_project/test_data/task_1"
Dataset_Json_Test_Task_2 = "dataset_project/test_data/task_2"
Dataset_Json_Test_Task_3 = "dataset_project/test_data/task_3"
Dataset_Json_Test_Task_4 = "dataset_project/test_data/task_4"

weight_task_1 = "E:/AI Competition/TextOCR/SubmissionFinalCode/Task1/Train/Weight"
weight_task_3 = "E:/AI Competition/TextOCR/SubmissionFinalCode/Task3/Train/Weight"
weight_task_4 = "E:/AI Competition/TextOCR/SubmissionFinalCode/Task4/Train/Weight"

Task_1_Train_And_Test_Config = {
    "input_images_train": Dataset_Image_Train,
    "input_images_test": Dataset_Image_Test,
    "json_train": Dataset_Json_Train_Task_1,
    "json_test": Dataset_Json_Test_Task_1,
    "weight": weight_task_1
}

Task_3_Train_And_Test_Config = {
    "input_images_train": Dataset_Image_Train,
    "input_images_test": Dataset_Image_Test,
    "json_train": Dataset_Json_Train_Task_3,
    "json_test": Dataset_Json_Test_Task_3,
    "weight": weight_task_3
}

Task_4_Train_And_Test_Config = {
    "input_images_train": Dataset_Image_Train,
    "input_images_test": Dataset_Image_Test,
    "json_train": Dataset_Json_Train_Task_4,
    "json_test": Dataset_Json_Test_Task_4,
    "weight": weight_task_4
}

def return_Task1_Train_Test_Config():
    return Task_1_Train_And_Test_Config

def return_Task3_Train_Test_Config():
    return Task_3_Train_And_Test_Config

def return_Task4_Train_Test_Config():
    return Task_4_Train_And_Test_Config

#Config data for inference
Dataset_Image_Inference = "dataset_project/predict_data"
Output_Json_Task_1 = "SubmissionFinalCode/Task1/Inference/Task_1_predict_json"
Output_Json_Task_2 = "SubmissionFinalCode/Task2/Inference/Task_2_predict_json"
Output_Json_Task_3 = "SubmissionFinalCode/Task3/Inference/Task_3_predict_json"
Output_Json_Task_4 = "SubmissionFinalCode/Task4/Inference/Task_4_predict_json"
Output_Json_Task_5 = "SubmissionFinalCode/Task5/Inference/Task_5_predict_json"

Task_1_Predict_Config = {
    "input_images": Dataset_Image_Inference,
    "output_json": Output_Json_Task_1,
    "weight": weight_task_1
}

Task_2_Predict_Config = {
    "input_images_train": Dataset_Image_Train,
    "input_images_test": Dataset_Image_Test,
    "json_train": Dataset_Json_Train_Task_2,
    "json_test": Dataset_Json_Test_Task_2,
    "input_json_inference": Output_Json_Task_1,
    "input_images_inference": Dataset_Image_Inference,
    "output_json_inference": Output_Json_Task_2,
}

Task_3_Predict_Config = {
    "input_json": Output_Json_Task_2,
    "input_images": Dataset_Image_Inference,
    "output_json": Output_Json_Task_3,
    "weight": weight_task_3
}

Task_4_Predict_Config = {
    "input_json": Output_Json_Task_3,
    "input_images": Dataset_Image_Inference,
    "output_json": Output_Json_Task_4,
    "weight": weight_task_4
}

Task_5_Predict_Config = {
    "input_json": Output_Json_Task_4,
    "input_images": Dataset_Image_Inference,
    "output_json": Output_Json_Task_5
}

def return_Task1_Predict_Config():
    return Task_1_Predict_Config

def return_Task2_Predict_Config():
    return Task_2_Predict_Config

def return_Task3_Predict_Config():
    return Task_3_Predict_Config

def return_Task4_Predict_Config():
    return Task_4_Predict_Config

def return_Task5_Predict_Config():
    return Task_5_Predict_Config