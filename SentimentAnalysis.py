import streamlit as st #creates webapp
from transformers import pipeline

#classifier
classifier = pipeline("sentiment-analysis")

st.title("sentiment analysis") #creates webapp
text = st.text_input("Enter a sentence:") #gets text input
if text:
    result = classifier(text)[0]
    st.write(f"**Sentiment:** {result['label']}")
    st.write(f"**Confidence:** {result['score']}")

print(classifier("I love this product!"))
print(classifier("this is awful"))

print(classifier("not bad"))
print(classifier("fine food"))
print(classifier("fine dining"))
print(classifier("fine wine"))