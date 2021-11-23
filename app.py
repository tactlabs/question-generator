import streamlit as st 
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy 
import os 


os.system('python -m spacy download en_core_web_lg ')
mdl = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
tknizer = AutoTokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
nlp = spacy.load("en_core_web_sm")

def get_question(sentence, answer):

    text = "context: {} answer: {}".format(sentence,answer)
    max_len = 256
    encoding = tknizer.encode_plus(text,max_length=max_len, pad_to_max_length=False,truncation=True, return_tensors="pt")

    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outs = mdl.generate(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    early_stopping=True,
                                    num_beams=5,
                                    num_return_sequences=1,
                                    no_repeat_ngram_size=2,
                                    max_length=300)


    dec = [tknizer.decode(ids,skip_special_tokens=True) for ids in outs]


    Question = dec[0].replace("question:","")
    Question= Question.strip()
    return Question

def get_sent(context):
    doc = nlp(context)
    return list(doc.sents)

def get_vector(doc):
    stop_words = "english"
    n_gram_range = (1,1)
    df = CountVectorizer(ngram_range = n_gram_range, stop_words = stop_words).fit([doc])
    return df.get_feature_names()

def get_key_words(context):

    keywords = []
    top_n = 5
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    for txt in get_sent(context):
        keywd = get_vector(str(txt))
        doc_embedding = model.encode([str(txt)])
        keywd_embedding = model.encode(keywd)
        
        distances = cosine_similarity(doc_embedding, keywd_embedding)
        print(distances)
        keywords += [(keywd[index], str(txt)) for index in distances.argsort()[0][-top_n:]]

    return keywords

def main():
    st.title('Question generator')
    text = st.text_area(label = 'context')
    if text:
        loader = st.progress(0)
        keywords = get_key_words(text)
        loader.progress(50)
        questions = {get_question(cont, ans) for ans, cont in keywords}
        loader.progress(100)
        if questions:
            st.write('### Questions :')
            st.write()
            for num, op in enumerate(questions):
                st.write(f'#### Q{num + 1} ) {op}')


if __name__ == '__main__':
    main()