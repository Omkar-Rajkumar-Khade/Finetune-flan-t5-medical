# Medical Question Answering with T5

This repository contains code for fine-tuning a T5 model on a medical question-answering dataset using the Hugging Face Transformers library. The model is trained to generate answers to medical questions.

## Dataset
The medical question-answering dataset used in this project is available at *Omkar7/Medical_data* on the huggingface dataset. The dataset is split into training and testing sets.

## Training:
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

## Evaluation:
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
      
## Conclusion:
This repository presents a comprehensive solution for Medical Question Answering using the T5 model and the Hugging Face Transformers library. The primary objective is to generate accurate and contextually relevant answers to medical queries.
The results of the evaluation showcase the model's ability to generate meaningful answers to medical questions. The ROUGE scores provide a quantitative measure of the model's performance, with a focus on precision and recall of generated text. The successful completion of this project demonstrates the potential of fine-tuned transformer models in addressing specific domain-related queries, paving the way for further advancements in medical question answering.
