import pickle as pkl
import string
import warnings

# from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import spacy
from flask import Flask, redirect, render_template, request, session, url_for
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from tensorflow.keras.preprocessing.sequence import pad_sequences

warnings.filterwarnings("ignore")
parser = English()
punctuations = string.punctuation

stopwords = set(STOP_WORDS)

app = Flask(__name__)
app.secret_key = 'key'


with open(file="input/label_names.pkl", mode="rb") as file:
    label_names = pkl.load(file=file)

with open(file="model/tokens.pkl", mode="rb") as file:
    token_model = pkl.load(file=file)

with open(file="model/RF_model.pkl", mode="rb") as file:
    rf_model = pkl.load(file=file)

nlp = spacy.load("en_core_web_sm")


def preprocess_text(docx):
    sentence = parser(docx)
    sentence = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in sentence]
    sentence = [word for word in sentence if word not in stopwords and word not in punctuations]
    sentence = [word for word in sentence if len(word) > 3 and word.isalpha()]
    return sentence


def cosine_similarity(x, y):
    # Ensure length of x and y are the same
    if len(x) != len(y):
        return None
    # Compute the dot product between x and y
    dot_product = np.dot(x, y)

    # Compute the L2 norms (magnitudes) of x and y
    magnitude_x = np.sqrt(np.sum(x ** 2))
    magnitude_y = np.sqrt(np.sum(y ** 2))

    # Compute the cosine similarity
    cosine_similarity = dot_product / (magnitude_x * magnitude_y)
    return cosine_similarity


@app.route("/submit", methods=['GET', 'POST'])
def get_hours():
    if request.method == 'POST':
        input_text = request.form['text']
        #input_text = str(input_text)
        #input_data = nlp(input_text)
        cleaned_text_data = preprocess_text(nlp(input_text))
        encd_text = token_model.texts_to_sequences([cleaned_text_data])
        pad_text = pad_sequences(sequences=encd_text, maxlen=100, padding="post")
        prediction = rf_model.predict(pad_text)
        print(prediction)
        label = label_names[prediction[0]]
        test_data = pd.read_csv("input/ResumeData.csv")
        need_data = test_data.loc[test_data["Category"] == label]
        need_data.reset_index(drop=True, inplace=True)
        input_data = pd.DataFrame([input_text], columns=["input"])

        s_values = []

        for i in range(need_data.shape[0]):
            corpus = [input_data["input"][0], need_data["Resume"][i]]
            X = CountVectorizer().fit_transform(corpus).toarray()
            sims = cosine_similarity(X[0], X[1])
            s_values.append(sims)
        need_data["probability_score"] = s_values
        sorted_data = need_data.sort_values(["probability_score"], ascending=False)
        sorted_data = sorted_data.reset_index(drop=True).head(10)
        ID_list = list(sorted_data["Resume_ID"])
        print(ID_list)
        hist = pd.read_csv('input/info.csv')

        final_result = []

        for i in range(len(ID_list)):
            id_ = ID_list[i]
            for j in range(len(hist)):
                hist_id = hist["Resume_ID"][j]
                if id_ == hist_id:
                    result = hist.loc[hist["Resume_ID"] == hist_id]
                    # df = final_result.iloc[["Resume_ID"]].reset_index(drop=True)

                    final_result.append(result)
        fr = pd.concat(objs=final_result).reset_index(drop=True)
        fr[["probability_score", "Category"]] = sorted_data[["probability_score", "Category"]]
        final_result = fr.drop(labels=["GraduationDate", "DegreeType", "WorkHistoryCount"], axis=1)
        print(final_result)

        cols = list(hist.columns)
        df = np.asarray(final_result)
        data1 = df
        return render_template("result.html", result=final_result, cols=cols, df=data1)


@app.route('/')
@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        email = request.form["email"]
        pwd = request.form["password"]
        r1 = pd.read_excel("user.xlsx")
        for index, row in r1.iterrows():
            if row["email"] == str(email) and row["password"] == str(pwd):
                return redirect(url_for('home'))
        else:
            msg = 'Invalid Login Try Again'
            return render_template('login.html', msg=msg)
    return render_template('login.html')


@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['Email']
        Password = request.form['Password']
        col_list = ["name", "email", "password"]
        r1 = pd.read_excel('user.xlsx', usecols=col_list)
        new_row = {'name': name, 'email': email, 'password': Password}
        r1 = r1.append(new_row, ignore_index=True)
        r1.to_excel('user.xlsx', index=False)
        print("Records created successfully")
        # msg = 'Entered Mail ID Already Existed'
        msg = 'Registration Successful !! U Can login Here !!!'
        return render_template('login.html', msg=msg)
    return render_template('login.html')


@app.route("/home", methods=['GET', 'POST'])
def home():
    return render_template("home.html")


@app.route('/password', methods=['POST', 'GET'])
def password():
    if request.method == 'POST':
        current_pass = request.form['current']
        new_pass = request.form['new']
        verify_pass = request.form['verify']
        r1 = pd.read_excel('user.xlsx')
        for index, row in r1.iterrows():
            if row["password"] == str(current_pass):
                if new_pass == verify_pass:
                    r1.replace(to_replace=current_pass, value=verify_pass, inplace=True)
                    r1.to_excel("user.xlsx", index=False)
                    msg1 = 'Password changed successfully'
                    return render_template('password_change.html', msg1=msg1)
                else:
                    msg2 = 'Re-entered password is not matched'
                    return render_template('password_change.html', msg2=msg2)
        else:
            msg3 = 'Incorrect password'
            return render_template('password_change.html', msg3=msg3)
    return render_template('password_change.html')


@app.route('/graphs', methods=['POST', 'GET'])
def graphs():
    return render_template('Graphs.html')


@app.route('/knn')
def knn():
    return render_template('knn.html')


@app.route('/logout')
def logout():
    session.clear()
    msg = 'You are now logged out', 'success'
    return redirect(url_for('login', msg=msg))


if __name__ == '__main__':
    app.run(port=5002, debug=True)
