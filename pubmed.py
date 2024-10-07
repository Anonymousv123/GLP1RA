# -*- coding: utf-8 -*-
import pandas as pd
import os
import time
from Bio import Entrez

PTH = '/.../pubmed_GLP1/'

# Set your PubMed API key here
Entrez.email = '...'
Entrez.api_key ='...'

keywords = [
    'Dulaglutide',
    'Trulicity',
    'Byetta',
    'Exenatide',
    'Bydureon',
    'Liraglutide',
    'Victoza',
    'Lixisenatide',
    'Adlyxin',
    'Semaglutide',
    'Ozempic',
    'Rybelsus',
]

"""## Collect PubMed data"""

def search_pubmed(query, max_results):
    Entrez.email = "..."  # Set your email address
    search_results = Entrez.read(
        Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    )
    return search_results

def fetch_pubmed_records(pmid_list):
    pmid_str = ",".join(pmid_list)
    handle = Entrez.efetch(db="pubmed", id=pmid_str, rettype="xml")
    records = Entrez.read(handle)
    return records

def parse_pubmed_records(records):
    paper_data = []
    for record in records["PubmedArticle"]:
        pmid = record["MedlineCitation"]["PMID"]
        title = record["MedlineCitation"]["Article"]["ArticleTitle"]
        if "Abstract" in record["MedlineCitation"]["Article"]:
            abstract = record["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
        else:
            abstract = "N/A"
        pub_date = record["PubmedData"]["History"][0]["Year"]

        paper_data.append({
            "PMID": pmid,
            "Title": title,
            "Abstract": abstract,
            "PublicationDate": pub_date
        })

    return paper_data

for i in range(len(keywords)):
  query = keywords[i]
  max_results = 1325  # Set the maximum number of results you want to retrieve
  search_results = search_pubmed(query, max_results)
  pmid_list = search_results["IdList"]

  total_results = int(search_results["Count"])
  print(f"Total number of papers found: {total_results} for ", query)

  records_per_query = 200  # Number of records to fetch per query (adjust as needed)
  all_paper_data = []

  for start in range(0, len(pmid_list), records_per_query):
      pmid_subset = pmid_list[start:start + records_per_query]
      records = fetch_pubmed_records(pmid_subset)
      paper_data = parse_pubmed_records(records)
      all_paper_data.extend(paper_data)

  # Create a DataFrame from the collected data
  df = pd.DataFrame(all_paper_data)
  df.to_csv(os.path.join(PTH,'pubmed',query+".csv"), index = False)

# combine all data frames
combined_df = pd.read_csv(os.path.join(PTH,'pubmed',keywords[0]+".csv"))
combined_df['drug'] = keywords[0]

for i in range(1, len(keywords)):
  file = keywords[i]
  df = pd.read_csv(os.path.join(PTH,'pubmed',file+".csv"))
  df['drug'] = file
  combined_df = pd.concat([combined_df, df], axis=0, ignore_index=True)

combined_df.to_csv(os.path.join(PTH,'pubmed',"pubmed_combined.csv"),index=False)

"""## Use scispacy for NER to identify ASD symptoms in posts

https://github.com/allenai/SciSpaCy#installation
"""

import scispacy
import spacy
import pandas as pd

nlp = spacy.load("en_ner_bc5cdr_md")

combined_df = pd.read_csv(os.path.join(PTH,"pubmed_combined.csv"))
combined_df

# time range of papers
print(combined_df['PublicationDate'].max(),
      combined_df['PublicationDate'].min(),
      combined_df['PublicationDate'].max()- combined_df['PublicationDate'].min(), 'years \n'
      )


combined_df['PublicationDate'].value_counts()

combined_df['Combined_Content'] = combined_df['Title'].fillna('') + ' ' + combined_df['Abstract'].fillna('')

UML_entities = []

k = 0

print(len(combined_df['Combined_Content']))

for text in combined_df['Combined_Content']:
    print(k)
    if isinstance(text, str):
        doc = nlp(text)
        ents = list(set(doc.ents))
        UML_entities.append(ents)
    else:
        # Handle non-string values, e.g., NaN or None
        UML_entities.append([])  # Append an empty list or handle as needed
    k = k + 1

  # Examine the entities extracted by the mention detector.
  # Note that they don't have types like in SpaCy, and they
  # are more general (e.g including verbs) - these are any
  # spans which might be an entity in UMLS, a large
  # biomedical database.

combined_df['UML'] = UML_entities
combined_df.to_csv(os.path.join(PTH,'spacy.csv'), index = False)

"""## **Create a list of biological entities using spacy data**"""

combinded_df = pd.read_csv(os.path.join(PTH,'spacy.csv'))
combinded_df

# a function to parse the string into a list

import re

def parse_custom_string(input_string):
    elements = re.split(r',(?![^\[]*\])', input_string)
    cleaned_elements = [element.strip('[] ').strip() for element in elements]
    cleaned_elements = [element for element in cleaned_elements if element]
    return cleaned_elements

# Apply the parse_custom_string function to convert the 'UML' column strings into lists
combinded_df['UML'] = combinded_df['UML'].apply(parse_custom_string)

bio_entities = []
k = 0
for sublist in combinded_df['UML']:
  for item in sublist:
    bio_entities.append([k,item])
  k = k + 1

bio_entities = pd.DataFrame(bio_entities)
bio_entities.columns = ['id','entity']
bio_entities

bio_entities['entity'] = [ x.split(',') for x in bio_entities['entity'] ]

# Initialize an empty list to store the flattened elements
flattened_list = []

# Iterate through the nested list and extend the flattened_list
for sublist in bio_entities:
    flattened_list.extend(sublist)

flattened_list = bio_entities.explode('entity', ignore_index=True)

# Reset the index to get the original row ID (id) for each value
flattened_list.reset_index(drop=True, inplace=True)

# Rename the columns if needed
flattened_list.columns = ['id', 'entity']

flattened_list

flattened_list.to_csv(os.path.join(PTH,'symptoms_list_pubmed.csv'), index = False)

"""## **Filter non-side effects**"""

import pandas as pd
import os

flattened_list = pd.read_csv(os.path.join(PTH,'symptoms_list_pubmed.csv'))
flattened_list = flattened_list[['id','entity']]
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

# Download NLTK data (if not already downloaded)
nltk.download('punkt')

# read side effect list from SIDER
side_effects = pd.read_csv(os.path.join(PTH,'sider.csv'))
side_effects = side_effects['se'].tolist()
side_effects = side_effects + (['cramp'])

# Initialize the Porter Stemmer
stemmer = PorterStemmer()
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

flattened_list

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
              'hunger pains',
              'injury',
              'neuropathy pain',
              'numbing pain',
              'obesity',
              'painful cramping',
              'painful death',
              'parkinsons',
              'violent vomit',
              'vulvovaginal',
              'acne',
              'alcohol nausea',
              'aching pain',
              'cold sweats'
]

mask = flattened_list['stem'].apply(lambda x: any(word in x for word in to_remove))
flattened_list = flattened_list[~mask]

flattened_list.to_csv(os.path.join(PTH,'side_effect_stem_dictionary.csv'),index = False)
flattened_list

# Perform a self-join and filter for A.entity > B.entity
edge_list = flattened_list.merge(flattened_list, on='id', suffixes=('_A', '_B'))
edge_list = edge_list[edge_list['stem_A'] > edge_list['stem_B']]

# Group by B.entity and A.entity, then count occurrences
edge_list = edge_list.groupby(['stem_A', 'stem_B']).size().reset_index(name='count')

edge_list[edge_list['count']>0]

edge_list.to_csv(os.path.join(PTH,'PubMed_side_efect_edgelist.csv'),index=False)
