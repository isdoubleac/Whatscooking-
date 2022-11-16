from gensim.models import Word2Vec
import pandas as pd
import config
import sys
# sys.path is a list of absolute path strings
sys.path.append('/home/user/Whatscooking-/src')
from ingredient_parser import ingredient_parser

# get corpus with the documents sorted in alphabetical order
def get_and_sort_corpus(data):
    corpus_sorted = []
    for doc in data.parsed.values:
    	#print(doc)
    	doc = sorted(doc.split(' '))
    	corpus_sorted.append(doc)
    return corpus_sorted
  
# calculate average length of each document 
def get_window(corpus):
    lengths = [len(doc) for doc in corpus]
    avg_len = float(sum(lengths)) / len(lengths)
    return round(avg_len)

if __name__ == "__main__":
    # load in data
    data = pd.read_csv('input/clean_data.csv')
    # parse the ingredients for each recipe
    data['parsed'] = data.ingredients.apply(ingredient_parser)
    # get corpus
    corpus = get_and_sort_corpus(data)
    print(f"Length of corpus: {len(corpus)}")
    # train and save CBOW Word2Vec model
    model_cbow = Word2Vec(
      corpus, sg=0, workers=8, window=get_window(corpus), min_count=1, vector_size=100
    )
    model_cbow.save('models/model_cbow.bin')
    print("Word2Vec model successfully trained")
