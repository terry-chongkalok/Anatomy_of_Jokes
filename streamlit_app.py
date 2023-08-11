from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
import torch
import os

import shap
import streamlit as st
from streamlit_shap import st_shap

model_name = "distilroberta-base"
num_class = 2

@st.cache_resource
def get_tokenizer_and_model(model_name=model_name, num_class=num_class):
	if os.path.isdir('tokenizer'):
		tokenizer = AutoTokenizer.from_pretrained('tokenizer', padding_side='right')
	else:
		tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='right')
		tokenizer.save_pretrained('tokenizer')

	if os.path.isdir('base_model'):
		model = AutoModelForSequenceClassification.from_pretrained('base_model')
	else:
		model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_class, return_dict=True)
		model.save_pretrained('base_model')
	return tokenizer, model

tokenizer, model = get_tokenizer_and_model()

pretrained_peft_config = LoraConfig.from_pretrained("joke_detector")
model = get_peft_model(model, pretrained_peft_config)
adapter_weights = torch.load('joke_detector/adapter_model.bin', map_location='cpu')
set_peft_model_state_dict(model, adapter_weights)

def pred_prob(txt_list: list, model=model, tokenizer=tokenizer):
  seq_tensor = torch.tensor([tokenizer(txt, padding='max_length', truncation=True, max_length=128)['input_ids'] for txt in txt_list])
  model.eval()
  pred = model(seq_tensor).logits
  return torch.softmax(pred, -1).detach().cpu().numpy()


st.title("👨‍⚕️🔪 Anatomy of Jokes 🤪🤡🥶")
st.markdown("### Joke Detection + Detection Explanability")

explain_txt = """
After entering a sentence in the input box and pressing `Enter`, the language model will determine if it is a joke. \n
Only sentences with Joke Detection Score greater than 0.5 will be classified as jokes.\n
Additionally, `SHAP` will be used to calculate the significance of each word (token) for the detection score, and the result will be presented in a chart.\n
When selecting `Joke🤣` above the chart, words highlighted in red with a positive score indicate the humorous parts of the sentence, while words in blue with a negative score are non-humorous.
"""
st.sidebar.title("How to read the anatomy results:")
st.sidebar.write(explain_txt)

st.markdown("`Explaining a joke is like dissecting a frog. You understand it better but the frog dies in the process.🐸`")

input = st.text_input(label="", placeholder="Put a sentence here to see if it is funny or not!", label_visibility='collapsed')

if input:
    test_prob = pred_prob([input])
    if test_prob.argmax():
        st.success("It is a joke.🤣")
    else:
        st.warning("It is not a joke.😴")

    col1, col2 = st.columns([1, 4])
    col1.metric("Joke Detection Score", f"{test_prob[0][1]:.2f}")

    with col2:
        st.write("🔪 Anatomy Result:")
        with st.spinner("Anatomizing the sentence..."):
            explainer = shap.Explainer(pred_prob, tokenizer, output_names=['No Joke😴', ' Joke🤣'])
            shap_values = explainer([input])
            st.write("Click on `No Joke😴` or `Joke🤣` to see how each word contributes to the detection score!")
            st_shap(shap.plots.text(shap_values), height=300)