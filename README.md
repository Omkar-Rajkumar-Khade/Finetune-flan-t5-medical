# MediBot

This repository contains code for fine-tuning a T5 model on a medical question-answering dataset using the Hugging Face Transformers library. The model is trained to generate answers to medical questions.

## Dataset
The medical question-answering dataset used in this project is available at [Omkar7/Medical_data](https://huggingface.co/datasets/omkar7/Medical_data) on the Hugging Face dataset. The dataset is split into training and testing sets.

## Getting Started

Follow these steps to get started with the Medical Question Answering project:

### 1. Clone the Repository

```bash
git clone https://github.com/Omkar-Rajkumar-Khade/Finetune-flan-t5-medical.git
cd Finetune-flan-t5-medical
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

The medical question-answering dataset used in this project is available at [Omkar7/Medical_data](https://huggingface.co/datasets/omkar7/Medical_data). Download the dataset and place it in the appropriate directory.

### 4. Training the Model 

To train the model using ipynb notebook
This will launch a prompt where you can input medical questions, and the model will generate answers.

### 5. Application

To use the streamlit application from the base directory
```bash
streamlit run src/app.py
```

![screenshot](https://github.com/SRDdev/Finetune-flan-t5-medical/blob/8e33064dd62d68beb5ff23e3000a5beed4b2c12b/assets/Screenshot.png)

## Training
The training process involves loading the dataset, tokenization, model initialization, preprocessing, setting up evaluation metrics (ROUGE score), and training using a Seq2Seq trainer.

#### Hyperparameters:
- Learning Rate: 3e-4
- Batch Size: 5
- Epochs: 5
- Logging Steps: 500
- Evaluation Strategy: Epoch
- Logging Strategy: Steps
- Weight Decay: 0.01
- Predict with Generate: True

#### Training Process:
- Load dataset, tokenize, initialize model, preprocess, set up ROUGE score metrics, and train using Seq2Seq trainer.
- Logs and model checkpoints are saved in the "./finetuned-t5-medical" directory.

## Evaluation
After training, the model is evaluated on a test dataset. ROUGE score is used as an evaluation metric.
- After training, the model undergoes evaluation on a test dataset. ROUGE score serves as the primary evaluation metric. The evaluation results include metrics such as loss, ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum.
- Evaluation results include:
  1. eval_loss: 1.8686
  2. eval_rouge1: 0.1114
  3. eval_rouge2: 0.0304
  4. eval_rougeL: 0.0872
  5. eval_rougeLsum: 0.1020
  6. eval_runtime: 23.6847 seconds
  7. eval_samples_per_second: 4.475
  8. eval_steps_per_second: 0.929
  9. epoch: 5.0

- Base T5 Model vs Fine-tuned T5 Model:
  | Question                      | Base Model Answers               | Fine-tuned Model Answers                                              |
  |-------------------------------|----------------------------------|-----------------------------------------------------------------------|
  | What is (are) Glaucoma?       | ear infection                    | Glaucoma is the most common type of eye disease in the United States. It is the second most common type of disease in the eye, behind macular degeneration and cataracts.                    |
  | What is High Blood Pressure?   | high blood pressure              | High blood pressure is a risk factor for heart disease, stroke, diabetes, and other health problems. The risk of developing high blood pressure rises with age.               |
  | What causes Kidney Disease?    | ureters                          | Kidney disease is a disease in which the kidneys do not produce enough blood to carry out the functions they should.                    |
  | What is (are) Low Vision?      | low vision                       | Low vision is a vision loss caused by a disease called macular degeneration or macular degeneration. Macular degeneration is a condition in which the macula of the eye becomes infected with macular degeneration.                |
  | What is (are) Diabetes?        | diabetes                         | Diabetes is one of the most common forms of diabetes. It is the most common type of disease among older adults. It is the second leading cause of death in the United States, behind heart disease and stroke.   |

## Conclusion
This repository presents a comprehensive solution for Medical Question Answering using the T5 model and the Hugging Face Transformers library. The primary objective is to generate accurate and contextually relevant answers to medical queries.
The results of the evaluation showcase the model's ability to generate meaningful answers to medical questions. The ROUGE scores provide a quantitative measure of the model's performance, with a focus on precision and recall of generated text. The successful completion of this project demonstrates the potential of fine-tuned transformer models in addressing specific domain-related queries, paving the way for further advancements in medical question answering.

## Contributing

If you'd like to contribute to this project, please follow the [contributing guidelines](CONTRIBUTING.md).
