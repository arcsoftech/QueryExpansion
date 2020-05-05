"""
Name: Arihant Chhajed
"""
import numpy as np 
import re 
import string
import pandas as pd
import nltk
from collections import Counter
import math
from nltk.stem import PorterStemmer



documents = [ 'Sipping Chianti in rural Tuscany. Eating pizza on a Rome backstreet. Or exploring ancient history at Pompeii.', 
     'One in eight children and young people between the ages of 5 and 19 in England has a mental disorder, according to a new report.', 
     'The romantic boulevards and cobbled streets of Paris. The sparkling waters of the Cote dAzur. The slow-paced villages of Provence. Each is enough to make travelers swoon.', 
      'A troubling gap in life expectancy among the rich and poor has emerged in the United Kingdom, and researchers say it also has been seen in the United States.', 
     'The National Weather Service defines a hurricane as a "tropical cyclone with maximum sustained winds of 74 mph (64 knots) or higher.', 
     'Celery also provides a healthy dose of fiber, as well as vitamins C and K and potassium, and it is a very low-calorie snack.',
 'Tinseltown. La La Land. Los Angeles is easily reduced to cliché. But it doesnt take much to discover theres so much more than the glitz and glamor for which its renowned.',
 'The Sunshine State suffered the brunt of Hurricane Michaels punishing winds, which decimated beach towns and left little more than debris in their wake.',
 'Lake effect snow is expected to pound areas near the eastern Great Lakes on Tuesday into Wednesday -- especially just south and east of Buffalo, New York, and far northwestern Pennsylvania.',
 'After two and a half weeks of historic destruction, the Camp Fire in Northern California is 100 contained, but the search for remains threatens to push the death toll over 88, where it stood late Monday.']

 # Stopwords File
stopwords = open('stopwords').readlines()
ps = PorterStemmer()
# Remove Stopwords from the documents
def remove_stop_words(dataset):
    stop_word_filtered_docs =[]
    for doc in dataset:
        words = doc.split(" ")
        words = [ps.stem(word) for word in words if word not in stopwords]
        a = re.sub(' +', ' ', " ".join(words))
        stop_word_filtered_docs.append(a.split())
    return stop_word_filtered_docs
# Preprocess the documents
def remove_special_character(filtered):
    """
        Remove the special characters from the text line.
    """
    filtered = filtered.lower()
    table = str.maketrans('', '', string.punctuation)
    filtered = filtered.translate(table)
    filtered=filtered.replace('\n',' ')   
    filtered = re.sub('\s+', ' ', filtered)     
    filtered = re.sub('\d+', ' ', filtered) 
    return filtered 

if __name__ == "__main__":
    dataset = []
    for doc in documents:
        text = remove_special_character(doc)
        dataset.append(text)
    for i in range(len(stopwords)):
        stopwords[i] = stopwords[i].strip()

    dataset  = remove_stop_words(dataset)
    
    corpus = []
    
    fij = {}
    tj = {}
    for i,doc in enumerate(dataset):
        corpus+=doc
        tj[i] = Counter(doc)
    t = corpus.__len__()
    N= len(dataset)
    count=0
    vocabulary = {}
    for x in corpus:
        if x not in vocabulary:
            vocabulary[x] =count
            count+=1
            
    for term in vocabulary:
        for doc in dataset:
            if term in fij:
                fij[term].append(doc.count(term))
            else:
                fij[term]=[doc.count(term)]

   
    
  
    frequency = Counter(vocabulary)
    itf = {}
    doclen = []
    for i,doc in enumerate(tj):
        itf[i]=math.log(t/len(tj.get(doc)),10)
        doclen.append(len(tj.get(doc)))


    print(fij.get("healthi"))
    print(fij.get("disord"))
    print(doclen)
    print(itf)
    
    vectors = {}
    
    for word in frequency:
        lnorm =0
        for k,docitf in itf.items():
            tf=0
            if word in fij:
                # temp = (0.5 * (1.0 * fij[i].get(word)) * docitf)
                tf = fij[word][k]

            wij = (0.5 + (0.5*tf/max(fij[word])))*docitf
            lnorm+=(wij**2)
         
            if word in vectors:
                vectors[word].append(wij)
            else:
                vectors[word]=[wij]
        lnorm = math.sqrt(lnorm)
        vectors[word] = np.divide(np.array(vectors[word]),lnorm)
    weights_dataframe = pd.DataFrame.from_dict(vectors).transpose()
    
    # Computation of corelation of terms with the query terms
    size = len(vocabulary)
    correlation = [[0]*size,[0]*size]
    a= weights_dataframe.loc["healthi"]
    b = weights_dataframe.loc["disord"]
    print("term vectors for health")
    print(vectors.get("healthi"))
    print("term vectors for disorder")
    print(vectors.get("disord"))
    for term,row in weights_dataframe.iterrows():
        
        for i in weights_dataframe.columns:
            q0= a[i]
            q1=b[i]
            w = row[i]
            correlation[0][vocabulary.get(term)] +=  q0 * w
            correlation[1][vocabulary.get(term)] +=  q1 * w

    # Vocabulary
    print(vocabulary.keys())

    # Expanded query terms computation
    w1=math.sqrt(2)
    w2=math.sqrt(2)
    sim = {}
    for key in vocabulary:
        sim[key] = (w1 * correlation[0][vocabulary.get(key)]) + (w2 * correlation[1][vocabulary.get(key)])

    expanded_Query = []
    sim = Counter(sim)
    expanded_Query = [k for k,v in sim.most_common(10)]


    print("Corrleation Vectors for health:-")
    print(correlation[0])
    print("Correlation Vectors for disorder:-")
    print(correlation[1])

    print(expanded_Query)
    import random
    term1 = random.choice(expanded_Query)
    term2 = random.choice(expanded_Query)
    while(term1  in  [ "healthi","disord"]):
        term1 = random.choice(expanded_Query)
    while(term2  in  [term1, "healthi","disord"]):
        term2 = random.choice(expanded_Query)
    print("expanded query is:- ")
    print("health disorder {} {}".format(term1,term2))
    






