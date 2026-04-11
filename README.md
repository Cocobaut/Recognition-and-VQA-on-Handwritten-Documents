# Document AI System - TextOCR Competition

This project is a comprehensive Document AI system, covering everything from handwritten text recognition to layout analysis and a Retrieval-Augmented Generation (RAG) system using state-of-the-art deep learning models.

## 👥 Development Team
This project was developed by:
* **Hồ Hồng Phúc Nguyên**
* **Nguyễn Hồ Quang Khải**
* **Nguyễn Tiến Minh**
* **Tán Khánh Phong**

---

## 🚀 Installation & Execution Guide

To ensure the system functions correctly and leverages hardware acceleration (GPU), please follow these steps:

### 1. Environment Setup
First, ensure all necessary libraries are installed from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 2. Model Setup (Crucial Step)
The system is divided into several stages (Tasks). Before running inference, you **must** execute the setup scripts to download and configure the model weights locally.

Please run the following scripts in order:
* `set_up_model_task1.py`: Configures the text region detection model.
* `set_up_model_task3.py`: Downloads the TrOCR weights for handwritten recognition.
* `set_up_model_task4.py`: Downloads the LayoutLMv3 weights for document layout analysis.
* `set_up_model_task5.py`: Configures the LLM (Llama 3) for the RAG system.

**Note:** These scripts ensure that all models are loaded onto the local machine before any prediction takes place.

### 3. Execution
Once the environment and models are ready, start the end-to-end pipeline by running:

```powershell
./run_setup.bat
```

---

## 4. 🛠 System Workflow

The system is designed as a sequential pipeline:

1.  **Setup Phase:** Downloads model weights from Hugging Face or local directories into the device memory (CUDA/GPU prioritized).
2.  **Inference Phase:**
    * **Task 1 & 2:** Detects and crops text-containing regions.
    * **Task 3 (OCR):** Converts cropped images into digital text.
    * **Task 4 (Layout Analysis):** Classifies text blocks (Header, Body, Footer, etc.) to understand document structure.
    * **Task 5 (Document AI/RAG):** Utilizes Llama 3 to answer questions based on the extracted data.

---

### 5. Dataset
Dataset is GNHK and HW-SQuAD