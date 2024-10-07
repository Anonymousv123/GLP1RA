PTH = '.../X_GLP1'

import os
import pandas as pd
import time
from datetime import datetime

"""# **3. Load data**"""

df = pd.read_csv(os.path.join(PTH,'twitter_scrap.csv'))
df = df[['id','text','likes','replies', 'retweets', 'quotes', 'searchQuery','timestamp']]

"""Time range of tweets for each drug"""

# Convert the 'timestamp' column to datetime objects
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Group the DataFrame by 'searchQuery' and find the maximum and minimum timestamps
result = df.groupby('searchQuery')['timestamp'].agg(['max', 'min'])

# Calculate the time duration between max and min
result['time_duration'] = result['max'] - result['min']

# Reset the index
result = result.reset_index()


"""## Search keywords in posts"""

keywords = [
    'dulaglutide',
    'trulicity',
    'byetta',
    'exenatide',
    'bydureon',
    'liraglutide',
    'victoza',
    'lixisenatide',
    'adlyxin',
    'semaglutide',
    'ozempic',
    'rybelsus',
]

df['searchQuery'].value_counts()

"""# **Detect side effects**

## ** Use scispacy for NER to identify ASD symptoms in posts**

https://github.com/allenai/SciSpaCy#installation
"""

import pandas as pd
import os
import ast
import re
import scispacy
import spacy

nlp = spacy.load("en_ner_bc5cdr_md")

UML_entities = []

k =0

print(len(df['text']))

for text in df['text']:
    print(k)
    k = k + 1
    if isinstance(text, str):
        doc = nlp(text)
        ents = list(set(doc.ents))
        UML_entities.append(ents)
    else:
        # Handle non-string values, e.g., NaN or None
        UML_entities.append([])  # Append an empty list or handle as needed


  # Examine the entities extracted by the mention detector.
  # Note that they don't have types like in SpaCy, and they
  # are more general (e.g including verbs) - these are any
  # spans which might be an entity in UMLS, a large
  # biomedical database.

df['UML'] = UML_entities
df.to_csv(os.path.join(PTH,'spacy.csv'), index = False)

"""## **Create a list of biological entities using spacy data**"""

combinded_df = pd.read_csv(os.path.join(PTH,'spacy.csv'))
combinded_df

import pandas as pd
import re

# Define a function to parse the string into a list
def parse_custom_string(input_string):
    if isinstance(input_string, str):  # Check if the input is a string
        elements = re.split(r',(?![^\[]*\])', input_string)
        cleaned_elements = [element.strip('[] ').strip() for element in elements]
        cleaned_elements = [element for element in cleaned_elements if element]
        return cleaned_elements
    else:
        return []  # Return an empty list if the input is not a string

# Apply the parse_custom_string function to convert the 'UML' column strings into lists
combinded_df['UML'] = combinded_df['UML'].apply(parse_custom_string)

bio_entities = []
k = 0
for sublist in combinded_df['UML']:
  for item in sublist:
    searchQuery = combinded_df.iloc[k]['searchQuery']
    bio_entities.append([k,item,searchQuery])
  k = k + 1

bio_entities = pd.DataFrame(bio_entities)
bio_entities.columns = ['id','entity','searchQuery']

# Initialize an empty list to store the flattened elements
flattened_list = []

# Iterate through the nested list and extend the flattened_list
for sublist in bio_entities['entity']:
    flattened_list.extend(sublist)

flattened_list = bio_entities.explode('entity', ignore_index=True)

# Reset the index to get the original row ID (id) for each value
flattened_list.reset_index(drop=True, inplace=True)

# Rename the columns if needed
flattened_list.columns = ['id', 'entity','searchQuery']

flattened_list.to_csv(os.path.join(PTH,'symptoms_list_twitter.csv'), index = False)

flattened_list = [word.strip() for word in flattened_list]

bio_entities = [x.lower() for x in flattened_list ]
bio_entities = list(set(bio_entities))

"""## **Filter non-side effects**"""

import pandas as pd
import os

flattened_list = pd.read_csv(os.path.join(PTH,'symptoms_list_twitter.csv'))
flattened_list = flattened_list[['id','entity','searchQuery']]
flattened_list['entity'] =  flattened_list['entity'].astype(str)
flattened_list['entity'] = flattened_list['entity'].str.lower()

# Filter entries with mor than 2 words in 'entity' column
flattened_list = flattened_list[flattened_list['entity'].str.split().str.len() <= 2]
flattened_list['entity'] = flattened_list['entity'].str.lower()
flattened_list

"""# Identify side effects using SIDER and stemming"""

import pandas as pd
import nltk
from nltk.stem import PorterStemmer
import os

# Download NLTK data (if not already downloaded)
nltk.download('punkt')

# read side effect list from SIDER
side_effects = pd.read_csv(os.path.join(PTH,'sider.csv'))
side_effects = side_effects['se'].tolist()
side_effects = side_effects + (['cramp'])

# Initialize the Porter Stemmer
stemmer = PorterStemmer()

# stem each side effect in SIDER
stemmed_side_effects = [stemmer.stem(x) for x in side_effects]

# Function to filter out non-side effects
def filter_side_effects(entity):
    words = nltk.word_tokenize(entity.lower())  # Tokenize and convert to lowercase
    filtered_words = [word for word in words if stemmer.stem(word) in stemmed_side_effects]
    return ' '.join(filtered_words)

# Apply the filter function to 'entity' column
flattened_list['entity'] = flattened_list['entity'].apply(filter_side_effects)

# Remove rows with empty 'entity' entries
flattened_list = flattened_list[flattened_list['entity'].str.strip() != '']

flattened_list['stem'] = [stemmer.stem(word) for word in flattened_list['entity']]

flattened_list.loc[flattened_list['entity']== 'anxiety depression','entity'] = 'depression'
flattened_list.loc[flattened_list['entity']=='constipation nausea','entity'] = 'nausea'
flattened_list.loc[flattened_list['entity']=='nausea vomiting','entity'] = 'vomiting'
flattened_list.loc[flattened_list['entity']=='pancreatic inflammation','entity'] = 'inflammation'
flattened_list.loc[flattened_list['entity']=='pneumonia pneumonia','entity'] = 'pneumonia'
flattened_list.loc[flattened_list['entity']=='fatigue fatigue','entity'] = 'fatigue'

# manual removal of non-side effects
to_remove = [
              'alcoholic',
              'alcohol intoxication',
              'alcohol',
              'autism',
              'clumsiness',
              'death',
              'disabl',
              'diabetes',
              'diabetes burn',
              'diabetic coma',
              'diabetic depression',
              'diabetic ketoacidosis',
              'diabetic ketosis',
              'diabetic neuropathy',
              'diabetic ulcers',
              'diabet',
              'dyslexia',
              'eczema',
              'fast',
              'abortion myocarditis',
              'hunger pains',
              'injury',
              'neuropathy pain',
              #'numbing pain',
              'obesity',
              #'painful cramping',
              'painful death',
              'parkinsons',
              #'violent vomit',
              'vulvovaginal',
              'acne',
              'obes',
              #'alcohol nausea',
              #'aching pain',
              #'cold sweats',
              'parkinson',
              'fibrosi'
]

mask = flattened_list['stem'].apply(lambda x: any(word in x for word in to_remove))
flattened_list = flattened_list[~mask]

flattened_list.to_csv(os.path.join(PTH,'side_effect_stem_dictionary_Twitter.csv'),index = False)

"""## side effects by GLP-1 drug name and mention frequency"""

flattened_list = pd.read_csv(os.path.join(PTH,'side_effect_stem_dictionary_Twitter.csv'))
flattened_list

import pandas as pd

# Group by 'searchQuery' and 'entity', then count the occurrences
grouped = flattened_list.groupby(['searchQuery', 'entity']).size().reset_index(name='count')

# Sort the DataFrame by count in descending order
sorted_df = grouped.sort_values(by='count', ascending=False)

len(set(sorted_df['entity'])) # number of side effects

import matplotlib.pyplot as plt
import seaborn as sns

# Pivot the DataFrame to prepare it for the heatmap
pivot_df = sorted_df.pivot(index='searchQuery', columns='entity', values='count')

# Create the heatmap using seaborn
plt.figure(figsize=(20, 6))  # Adjust the figure size as needed
sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='g')

# Set labels for the axes
plt.xlabel('Side Effect')
plt.ylabel('GLP-1 Drug')

# Show the plot
plt.show()

"""## edgelist"""

# Perform a self-join and filter for A.entity > B.entity
edge_list = flattened_list.merge(flattened_list, on='id', suffixes=('_A', '_B'))
edge_list = edge_list[edge_list['stem_A'] > edge_list['stem_B']]

# Group by B.entity and A.entity, then count occurrences
edge_list = edge_list.groupby(['stem_A', 'stem_B']).size().reset_index(name='count')

edge_list.to_csv(os.path.join(PTH,'Twitter_side_efect_edgelist.csv'),index=False)

"""# **Figures**

## wordcloud of clusters
"""

import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download("stopwords")
nltk.download('punkt')
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer

from collections import Counter

# Download the WordNet resource
nltk.download('wordnet')

se = pd.read_csv(os.path.join(PTH,'Twitter_chatgpt_symptoms.csv'))
se = se[se['y']==1] # keep only side effects
se

def plotWordCloud(text, words_to_remove=[]):
    # Combine custom stopwords with the built-in STOPWORDS
    combined_stopwords = STOPWORDS.union(words_to_remove)

    # Join the list of strings into a single string
    text = ' '.join(text)


    # Create a WordCloud object
    wordcloud = WordCloud(width=1000,
                          height=500,
                          colormap='Pastel1',
                          collocations=False,
                          stopwords=combined_stopwords,
                          background_color='black').generate(text)

    word_frequency = Counter(text.split())

    # Display the WordCloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    return word_frequency


# combine similar words using lemmatization

def preprocess_and_combine(word_list):
    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize each word and combine them into a single list
    processed_words = [lemmatizer.lemmatize(word) for word in word_list]

    return processed_words

# Generate word cloud for PTSD
symptoms = se['text'].tolist()
symptoms = [ x.lower() for x in symptoms]
symptoms = preprocess_and_combine(symptoms)

words_to_remove = ['failure','loss','syndrome','damage','symptoms','disorders',
                    'disorder','&','problems','acute','painful','attacks', 'distress',
                    'reactions','sudden','dysfunction', 'ache','chronic','’s','disease',
                    'adverse','blood','reaction','bad']

word_frequency = plotWordCloud(symptoms, words_to_remove)

# Filter out stopwords from word_frequency

present_words = STOPWORDS.union( words_to_remove)
filtered_word_frequency = {word: count for word, count in word_frequency.items() if word not in present_words}

# Sort the filtered word frequency data
sorted_filtered_word_frequency = dict(sorted(filtered_word_frequency.items(), key=lambda x: x[1], reverse=True))

# Get the top 30 words after removing stopwords
top_30_words = list(sorted_filtered_word_frequency.items())[:30]

# Create a bar plot for the top 30 words
top_words, top_word_counts = zip(*top_30_words)
plt.figure(figsize=(12, 6))
plt.bar(top_words, top_word_counts)
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.title('Top 30 Most Frequent Words (After Removing Stopwords)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# pain related heatmap
pain_words = [x for x in symptoms if 'pain' in x]

pain_words = [ x.lower() for x in pain_words]
pain_words = preprocess_and_combine(pain_words)


word_frequency = plotWordCloud(pain_words, ['painful'])

# Filter out stopwords from word_frequency

present_words = STOPWORDS.union( words_to_remove)
filtered_word_frequency = {word: count for word, count in word_frequency.items() if word not in present_words}

# Sort the filtered word frequency data
sorted_filtered_word_frequency = dict(sorted(filtered_word_frequency.items(), key=lambda x: x[1], reverse=True))

# Get the top 30 words after removing stopwords
top_30_words = list(sorted_filtered_word_frequency.items())[:30]

# Create a bar plot for the top 30 words
top_words, top_word_counts = zip(*top_30_words)
plt.figure(figsize=(12, 6))
plt.bar(top_words, top_word_counts)
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.title('Top 30 Most Frequent Words (After Removing Stopwords)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# pain related heatmap
loss_words = [x for x in symptoms if 'loss' in x]

loss_words = [ x.lower() for x in loss_words]
loss_words = preprocess_and_combine(loss_words)


word_frequency = plotWordCloud(loss_words, [])

# Filter out stopwords from word_frequency

present_words = STOPWORDS.union(words_to_remove)
filtered_word_frequency = {word: count for word, count in word_frequency.items() if word not in present_words}

# Sort the filtered word frequency data
sorted_filtered_word_frequency = dict(sorted(filtered_word_frequency.items(), key=lambda x: x[1], reverse=True))

# Get the top 30 words after removing stopwords
top_30_words = list(sorted_filtered_word_frequency.items())[:30]

# Create a bar plot for the top 30 words
top_words, top_word_counts = zip(*top_30_words)
plt.figure(figsize=(12, 6))
plt.bar(top_words, top_word_counts)
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.title('Top 30 Most Frequent Words (After Removing Stopwords)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
