import pandas as pd
import nltk
from nltk.corpus import stopwords
from collections import Counter
from tqdm import tqdm



def main():

    df = pd.read_csv("/home/ivan/projects/find_acronym_understand_meaning/data/full_data.csv")
    stop_words = set(stopwords.words("english"))

    #remove 1 char words
    print("remove 1 char words")
    df["TEXT_stage1"] = [" ".join([word for word in sentence.split(" ") if len(str(word)) >1]) for sentence in tqdm(df["TEXT"])]

    #remove stopwords
    print("remove stopwords")
    df["TEXT_stage2"] = [" ".join([word for word in nltk.word_tokenize(sentence) if (not word in stop_words)]) for sentence in tqdm(df["TEXT_stage1"])]

    cnt = Counter()
    print("remove most common words")
    for text in tqdm(df["TEXT_stage2"]):
        for word in text.split():
            cnt[word] += 1
        

    FREQWORDS = set([w for (w, wc) in cnt.most_common(100)])
    def remove_freqwords(text):
        """custom function to remove the frequent words"""
        return " ".join([word for word in str(text).split() if word not in FREQWORDS])

    # removing frequent words
    df["TEXT_stage3"] = [remove_freqwords(text) for text in tqdm(df["TEXT_stage2"])]

    print("saving cleaner version of the data")
    df.to_parquet("clean_data_v1.parquet")

if __name__ == "__main__":
    main()