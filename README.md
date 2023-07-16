# Anatomy of Jokes
`Explaining a joke is like dissecting a frog. You understand it better but the frog dies in the process.🐸`

This project aimed at creating a joke detection model by fine-tuning an existing language model using `LoRA`, and applying `SHAP` for detection explanation to identify the punch words in jokes. 

The entire project was completed in Google Colab to take advantage of the free GPU resources provided by the platform. For optimal display of the detection explanation results, it is recommended to open [the notebook in Colab](https://colab.research.google.com/github/terry-chongkalok/anatomy_of_joke/blob/main/Anatomy_of_Jokes.ipynb).

- Data Source: Dataset from Github Repository: [ColBERT-Using-BERT-Sentence-Embedding-for-Humor-Detection](https://github.com/Moradnejad/ColBERT-Using-BERT-Sentence-Embedding-for-Humor-Detection)
- Base model: [distilroberta-base ](https://huggingface.co/distilroberta-base)
- Model Evauation after fine-tuning:
    - Accuracy: `0.98`
    - F1 Score: `0.98`
