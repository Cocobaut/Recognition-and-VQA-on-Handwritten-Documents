@echo off
call venv\Scripts\activate
python -m SubmissionFinalCode.Task1.Inference.set_up_model_task1
python -m SubmissionFinalCode.Task3.Inference.set_up_model_task3
python -m SubmissionFinalCode.Task4.Inference.set_up_model_task4
python -m SubmissionFinalCode.Task5.Inference.set_up_model_task5
python -m SubmissionFinalCode.Task1.Inference.Task_1_predict
python -m SubmissionFinalCode.Task2.Inference.Task_2_predict
python -m SubmissionFinalCode.Task3.Inference.Task_3_predict
python -m SubmissionFinalCode.Task4.Inference.Task_4_predict
python -m SubmissionFinalCode.Task5.Inference.Task_5_predict
pause