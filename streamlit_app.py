import os
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st

import config as cfg
from inference import QAModelInference
from PIL import Image


def get_proba(ans_dict: dict, model_tag: str):
    """
    Returns probability distribution over start and end words, together with confidence.

    Parameters
    ----------
    ans - dictionary containing inference result.
    model_tag - string denoting the model, can be either "possible" or "plausible"

    Returns
    -------
    start_praba - nd.array containing probability distribution over start word
    end_praba - nd.array containing probability distribution over end word
    p - float, combined probability of highest probable start/end words.
    """

    start_proba_ = ans_dict[f'start_word_proba_{model_tag}_model'][0]
    end_proba_ = ans_dict[f'end_word_proba_{model_tag}_model'][0]
    p = (np.max(start_proba_) + np.max(end_proba_)) / 2
    return start_proba_, end_proba_, p


def fetch_cache_models():
    """
    If models don't exits on dist, download and store them.
    This is due to Streamlit Sharing current limiations (Oct 2020)
    """

    folder = cfg.model_folder
    if not os.path.exists(folder):
        os.makedirs(folder)

    for model_name, url in cfg.mappings.items():
        fn = f"{model_name}.pt"
        if not os.path.exists(os.path.join(folder, fn)):
            urllib.request.urlretrieve(url, os.path.join(folder, fn))


@st.cache(allow_output_mutation=True)
def load_model():
    inf = QAModelInference(models_path=cfg.model_folder, plausible_model_fn="model_plausible.pt",
                           possible_model_fn="model_possible.pt")
    return inf


with st.spinner("Caching models..."):
    fetch_cache_models()

with st.spinner("Loading models..."):
    model = load_model()

image = Image.open('Logo-DTC.png')
st.image(image, caption='DTC YO')

st.title("Question Answering System")

st.write("Built by Akash , Prateek , Sidharth , Vinay ")


st.info(
    ":bulb: How does it work? Enter a context, then ask a question about it."
    " The system will attempt to find and extract the answer, given it exists in the context.")

example_context = """Ratan Naval Tata (Hindi: रतन टाटा, Ratan Ṭāṭā, born 28 December 1937) is an Indian industrialist, philanthropist, and a former chairman of Tata Sons. He was also chairman of Tata Group, from 1990 to 2012, and again, as interim chairman, from October 2016 through February 2017, and continues to head its charitable trusts.[3][4] He is the recipient of two of the highest civilian awards of India, the Padma Vibhushan (2008) and Padma Bhushan (2000).[5]."""

example_question = "When Ratan Tata born?"



st.subheader("Context")
context = st.text_area("Provide context", value=example_context, height = 150, max_chars=3000)
#st.markdown(context)

st.subheader("Question")
question = st.text_input("Ask a question", value=example_question)


#st.markdown(question)
st.subheader("Model")
model_selection = st.selectbox("Choose a model", options=['Automatic', 'Trained on correct questions',
                                                                  'Trained on tricky questions'])


if st.button("Get an answer"):

    ans = model.extract_answer(context, question)

    if model_selection == 'Automatic':
        s_p, e_p, pr_p = get_proba(ans, model_tag="possible")
        s_pl, e_pl, pr_pl = get_proba(ans, model_tag="plausible")
        #print(ans)
        if ans['plausible_answer'] != '':
            start_p, end_p, confidence, answer = s_pl, e_pl, pr_pl, f"Models didn't agee on the answer. Choosing most probable: " \
                                                                    f"\"{ans['plausible_answer']}\"."
        else:
            start_p, end_p, confidence, answer = s_p, e_p, pr_p, ans['answer']
    elif model_selection == 'Trained on correct questions':
        start_p, end_p, confidence = get_proba(ans, model_tag="possible")
        answer = ans['answer']
    else:
        start_p, end_p, confidence = get_proba(ans, model_tag="plausible")
        answer = ans['plausible_answer']

    st.subheader("Answer")
    if not answer:
        st.markdown("Can't determine the answer.")
    else:
        st.markdown(answer)

    st.markdown("**Confidence**: {:.3f}".format(confidence))

    # print(ans['start_word_proba_possible_model'][0])

    st.markdown("---")
    st.markdown("**Probability distributions of start/end indices**")
    df = pd.DataFrame(columns=['start', 'end'])
    df['start'] = start_p
    df['end'] = end_p
    st.bar_chart(df)
    # st.bar_chart(ans['start_word_proba_possible_model'][0])
    # st.bar_chart(ans['end_word_proba_possible_model'][0])
