# -*- coding: utf-8 -*-

PTH = '/.../Reddit_GLP1'

import csv
import os
import pandas as pd
import time
import datetime
from datetime import datetime
import prawcore
from prawcore.exceptions import TooManyRequests
import praw
from prawcore.exceptions import TooManyRequests

"""# **2. Data Collection & Pre-Processing**"""

# credentials
reddit = praw.Reddit(
    client_id = '...',  # personal use script
    client_secret = '...', #secret
    username = '...', # developers
    password = '...', # your Reddit password
    user_agent = '...', # what ever...
    check_for_async=False
)

"""## 2.1. Define subreddits to scrape"""

# reddit posts related to ozempic and alike
subressits = [
    # Direct GLP1 discuaaions
                'Ozempic', # 50.1k
                'OzempicForWeightLoss', # 13.7k
                'GLP1_Ozempic_Wegovy', #1.3k
    # Non-direct GLP1 discuaaions
                'loseit', #3.9M
                'TheMorningToastSnark', #11.6k
                'MaintenancePhase', # 25.6k
                'Semaglutide', #45k
                'diabetes_t2', # 29.4k
                'semaglutidecompounds', #7.2k
                'trulicity', #1.1k
                'diabetes', # 109k
                'type2diabetes', # 8.4k
                'liraglutide', # 11.7k
                'medicine', # 453k
            ]

# Set up Reddit instance
subreddit_name = subressits[0] # change from 0,...,13
print(subreddit_name)

"""## 2.2 Collecting Posts & Comments from Reddit"""

subreddit = reddit.subreddit(subreddit_name)

# Create a CSV file and write header row
csv_file = open(os.path.join(PTH,subreddit_name+'_posts_comments_new.csv'), 'w', encoding='utf-8', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
                      'Post ID',
                      'Post Author',
                      'Post Content',
                      'post_date',
                      'Comment ID',
                      'Comment Author',
                      'Comment Content',
                      'Parent ID',
                      'Parent Author'
                      ])

total_posts = 0
total_comments = 0

# help function

def get_comments(comment, post, csv_writer, cutoff_date):
    """
    A recursive function to get all comments and their replies
    """
    # Check if the comment has MoreComments
    if isinstance(comment, praw.models.MoreComments):
        # Get all the comments under MoreComments
        for more_comments in comment.comments():
            get_comments(more_comments, post, csv_writer)
    else:
        # Write the comment information to the CSV file

        post_d = datetime.datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S')
        post_d_d = datetime.datetime.strptime(post_d, '%Y-%m-%d %H:%M:%S')
        cutoff_date = datetime.datetime.strptime(cutoff_date, '%Y-%m-%d %H:%M:%S')

        if post_d_d > cutoff_date:
          csv_writer.writerow([
                                post.id,
                                post.author.name if post.author else '',
                                '', # post contant
                                post_d, # post date
                                comment.id,
                                comment.author.name if comment.author else '',
                                comment.body,
                                comment.parent_id,
                                comment.parent().author.name if comment.parent() and comment.parent().author else ''
                              ])

        # Check if there are replies to the comment and write them to the CSV file
        if comment.replies:
            for reply in comment.replies:
                get_comments(reply, post, csv_writer,'2023-09-09 00:00:00')

# delete posts dataframe if exists
try:
  del posts
  print("posts deleted.")
except NameError:
  print("posts does not exist.")

import datetime
try:
    # initializing posts and other variables...
    try:
      posts
      print("posts exists!")
    except NameError:
      print("posts does not exist.")
      posts = list(subreddit.new(limit=1000))


    start_index = 0  # Manually update this value if you need to restart from a specific index

    max_posts = len(posts)

    for i in range(start_index, len(posts)):
        try:
            post = posts[i]

            # Convert the UNIX timestamp to a readable datetime format
            post_date = datetime.datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S')

            # Print the current index
            print(f"Processing post at index: {i} out of {max_posts} posts")

            # Write the post information to the CSV file (Include post_date in the list)
            csv_writer.writerow(
                [
                    post.id,
                    post.author.name if post.author else '',
                    post.title,
                    post_date,
                    '',
                    '',
                    '',
                    ''
                ]
            )

            csv_file.flush() # Flush the buffer to write the row immediately

            # Increment the total post count
            total_posts += 1

            # Write the post content to the CSV file
            csv_writer.writerow(
                [
                    post.id,
                    post.author.name if post.author else '',
                    post.selftext,
                    post_date,
                    '',
                    '',
                    '',
                    ''
                ]
            )

            csv_file.flush() # Flush the buffer to write the row immediately

            # Loop through each comment in the post
            for comment in post.comments:
                get_comments(comment, post, csv_writer)

                # Increment the total comment count
                total_comments += 1

        except TooManyRequests:
            print("Hit a rate limit, waiting...")
            while reddit.auth.limits['remaining'] == 0:
                # Calculate the waiting time in seconds
                waiting_time = reddit.auth.limits['reset_timestamp'] - time.time()
                print(f"waiting for: {waiting_time:.2f} seconds")
                if waiting_time < 0:
                    print("break while loop")
                    break
                else:
                    print(f"You need to wait for approximately {waiting_time:.2f} seconds.")
                    time.sleep(waiting_time)

except Exception as e:
    print(f"An error occurred: {e}")

finally:
  # Close the CSV file if it's open and not already closed
    if csv_file and not csv_file.closed:
        csv_file.close()

# Write the total post count and total comment count to the CSV file
print('Total Posts', total_posts, '', 'Total Comments', total_comments)

subreddit_name

"""# **3. Read data and combine**"""

import os
import pandas as pd
import time
from datetime import datetime

dataframes = []
for forum in subressits:
  df = pd.read_csv(os.path.join(PTH,'FirstRound',forum+"_posts_comments.csv"))
  df['subreddit'] = forum
  dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)
combined_df

dataframes = []
for forum in subressits:
  df = pd.read_csv(os.path.join(PTH,forum+"_posts_comments_new.csv"))
  df['subreddit'] = forum
  dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)
combined_df

"""### 3.1. Statistics"""

combined_df['subreddit'].value_counts()

# 'post_date' is in string format, convert it to datetime
combined_df['post_date'] = pd.to_datetime(combined_df['post_date'])

# Group the DataFrame by 'subreddit'
grouped = combined_df.groupby('subreddit')

# Calculate the maximum and minimum dates for each subreddit
max_dates = grouped['post_date'].max()
min_dates = grouped['post_date'].min()

# Calculate the duration in weeks
duration_in_weeks = (max_dates - min_dates).dt.days / 7  # Convert days to weeks

# Combine the results into a new DataFrame
result_df = pd.DataFrame({
    'Subreddit': max_dates.index,
    'Max Date': max_dates.values,
    'Min Date': min_dates.values,
    'Duration (Weeks)': duration_in_weeks.values
})

# Now, 'result_df' contains the maximum date, minimum date, and duration in weeks for each subreddit
result_df

"""### 3.2. Search keywords in posts"""

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

# keep all posts from these subreddits:
keep = [
          'Ozempic',
          'OzempicForWeightLoss',
          'GLP1_Ozempic_Wegovy',
        ]


keep_df = combined_df[combined_df['subreddit'].isin(keep)]
clean_df = combined_df[~combined_df['subreddit'].isin(keep)]

# Create a boolean mask for each keyword
keyword_masks = []
for keyword in keywords:
    post_content_mask = clean_df['Post Content'].str.contains(keyword, case=False, na=False)
    comment_content_mask = clean_df['Comment Content'].str.contains(keyword, case=False, na=False)
    keyword_mask = post_content_mask | comment_content_mask
    keyword_masks.append(keyword_mask)

# Combine the keyword masks using logical OR (|) to get the final mask
final_mask = pd.concat(keyword_masks, axis=1).any(axis=1)

# Use the final mask to filter rows that mention the keywords
filtered_df = clean_df[final_mask]

# `filtered_df` contains rows that mention any of the specified keywords in a case-insensitive manner

# id of posts that mention a keyword
post_ids = (set(filtered_df['Post ID']))
len(post_ids)

# get all posts of a thread that at least one of its posts mention a keyword (or more)
df = clean_df[clean_df['Post ID'].isin(post_ids)]

# number of posts that at least one post in a thread contains a keyword
df['subreddit'].value_counts()

# combinde filtered_df and keep_df
combinded_df = pd.concat([df, keep_df], axis =0)
combinded_df

# number of posts that contain a keyword
combinded_df['subreddit'].value_counts()

combinded_df.to_csv(os.path.join(PTH,"reddit1_combined_new.csv"), index = False)
print(os.path.join(PTH,"reddit_combined.csv"))

combinded_df

"""# **4. Detect side effects**

## 4.1. Use scispacy for NER to identify side effects in posts

https://github.com/allenai/SciSpaCy#installation
"""

import pandas as pd
import os
import ast
import re
import scispacy
import spacy

nlp = spacy.load("en_ner_bc5cdr_md")

combinded_df = pd.read_csv(os.path.join(PTH,"reddit1_combined_new.csv"))
combinded_df['Combined_Content'] = combinded_df['Post Content'].fillna('') + ' ' + combinded_df['Comment Content'].fillna('')

UML_entities = []

k = 0

print(len(combinded_df['Combined_Content']))

for text in combinded_df['Combined_Content']:
    if k%%1000:
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

combinded_df['UML'] = UML_entities
combinded_df.to_csv(os.path.join(PTH,'spacy_new.csv'), index = False)

"""## **Create a list of biological entities using spacy data**"""

combinded_df_1 = pd.read_csv(os.path.join(PTH,'spacy_new.csv'))

combinded_df_2 = pd.read_csv(os.path.join(PTH,'FirstRound','spacy.csv'))

combinded_df = pd.concat([combinded_df_2, combinded_df_1], ignore_index=True)
combinded_df

# a function to parse the string into a list
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
    post_date = combinded_df.iloc[k]['post_date']
    Combined_Content = combinded_df.iloc[k]['Combined_Content']
    bio_entities.append([k,item,Combined_Content,post_date])
  k = k + 1

bio_entities = pd.DataFrame(bio_entities)
bio_entities.columns = ['id','entity', 'Combined_Content','post_date']
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
flattened_list.columns = ['id', 'entity', 'Combined_Content','post_date']

flattened_list

flattened_list.to_csv(os.path.join(PTH,'symptoms_list_reddit_All.csv'), index = False)

"""## **Filter non-side effects**"""

import pandas as pd
import os

flattened_list = pd.read_csv(os.path.join(PTH,'symptoms_list_reddit_All.csv'))
flattened_list = flattened_list[['id','entity','Combined_Content','post_date']]
flattened_list['entity'] =  flattened_list['entity'].astype(str)
flattened_list['entity'] = flattened_list['entity'].str.lower()

# Filter entries with mor than 2 words in 'entity' column
flattened_list = flattened_list[flattened_list['entity'].str.split().str.len() <= 2]
flattened_list['entity'] = flattened_list['entity'].str.lower()
flattened_list

"""## **Identify side effects using SIDER and stemming**"""

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
              #'cold sweats'
]

mask = flattened_list['stem'].apply(lambda x: any(word in x for word in to_remove))
flattened_list = flattened_list[~mask]


flattened_list.loc[flattened_list['entity']=='aching pain','entity'] = 'pain'
flattened_list.loc[flattened_list['entity']=='aching pain','stem'] = stemmer.stem('pain')

flattened_list.loc[flattened_list['entity']=='alcohol nausea','entity'] = 'nausea'
flattened_list.loc[flattened_list['entity']=='alcohol nausea','stem'] = stemmer.stem('nausea')

flattened_list.loc[flattened_list['entity']=='violent vomit','entity'] = 'vomit'
flattened_list.loc[flattened_list['entity']=='violent vomit','stem'] = stemmer.stem('vomit')

flattened_list.loc[flattened_list['entity']=='painful cramping','entity'] = 'cramping'
flattened_list.loc[flattened_list['entity']=='painful cramping','stem'] = stemmer.stem('cramping')

flattened_list.loc[flattened_list['entity']=='numbing pain','entity'] = 'pain'
flattened_list.loc[flattened_list['entity']=='numbing pain','stem'] = stemmer.stem('pain')

flattened_list.loc[flattened_list['entity']=='neuropathy pain','entity'] = 'pain'
flattened_list.loc[flattened_list['entity']=='neuropathy pain','stem'] = stemmer.stem('pain')

flattened_list.loc[flattened_list['entity']=='hunger pains','entity'] = 'pains'
flattened_list.loc[flattened_list['entity']=='hunger pains','stem'] = stemmer.stem('pains')

flattened_list.loc[flattened_list['entity']=='diabetic depression','entity'] = 'depression'
flattened_list.loc[flattened_list['entity']=='diabetic depression','stem'] = stemmer.stem('depression')

flattened_list.loc[flattened_list['entity']=='diabetes burn','entity'] = 'burn'
flattened_list.loc[flattened_list['entity']=='diabetes burn','stem'] = stemmer.stem('burn')

flattened_list.to_csv(os.path.join(PTH,'side_effect_stem_dictionary_All.csv'),index = False)

"""## **Plot heatmap of side effext and drug**"""

flattened_list = pd.read_csv(os.path.join(PTH,'side_effect_stem_dictionary_All.csv'))
flattened_list

# search if a drug name is mentioned in the text

keyword_list = [
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


# Function to check if a keyword is in the text
def keyword_in_text(text, keywords):
    for keyword in keywords:
        if keyword in text:
            return keyword
    return False

# Create a new column 'Keyword_Present' to indicate if a keyword is in 'Combined_Content'
flattened_list['drug'] = flattened_list['Combined_Content'].apply(lambda x: keyword_in_text(x, keyword_list))

# keep only posts that spesifically mention drug names to create a database of drugs and side effects
drug_side_effect_df = flattened_list[flattened_list['drug'] != False][['drug','Combined_Content','entity','post_date']]

print(drug_side_effect_df['drug'].value_counts())

drug_side_effect_df

# Convert 'post_date' to a datetime data type
drug_side_effect_df['post_date'] = pd.to_datetime(drug_side_effect_df['post_date'])

# Set the time interval for splitting (e.g., 14 days)
interval = pd.DateOffset(days=14)

# Create a list of DataFrames, each representing a 14-day interval
intervals = [drug_side_effect_df[(drug_side_effect_df['post_date'] >= start_date) & (drug_side_effect_df['post_date'] < start_date + interval)] for start_date in pd.date_range(start=drug_side_effect_df['post_date'].min(), end=drug_side_effect_df['post_date'].max(), freq=interval)]

# Initialize an empty result DataFrame
result_df = pd.DataFrame(columns=['interval_index', 'drug', 'start_date', 'end_date', 'entity_count'])

# Iterate through the intervals and count entities for each drug
for interval_index, interval_df in enumerate(intervals):
    interval_start = interval_df['post_date'].min()
    interval_end = interval_df['post_date'].max()

    drug_entity_counts = interval_df.groupby(['drug', 'entity']).size().reset_index(name='entity_count')
    drug_entity_counts['start_date'] = interval_start
    drug_entity_counts['end_date'] = interval_end
    drug_entity_counts['interval_index'] = interval_index

    result_df = pd.concat([result_df, drug_entity_counts], ignore_index=True)

# Display the result DataFrame
result_df.columns = ['time_interval','drug','start_date','end_date','side_effect_frequency','side_effect']
result_df


# combine similar side_effects
result_df.loc[result_df['side_effect'] == 'vomit','side_effect'] = 'vomiting'
result_df.loc[result_df['side_effect'] == 'migraines','side_effect'] = 'migraine'

import matplotlib.pyplot as plt

# Assuming you have the original DataFrame named result_df
# First, create a DataFrame with the desired grouping
grouped_df = result_df.groupby(['time_interval', 'side_effect'])['side_effect_frequency'].sum().reset_index()

# Rename the 'side_effect_frequency' column to 'sum_side_effect_frequency'
grouped_df = grouped_df.rename(columns={'side_effect_frequency': 'sum_side_effect_frequency'})

grouped_df = grouped_df[grouped_df['sum_side_effect_frequency'] > 10]

# Create a figure and axis
fig, ax = plt.subplots()

# Define a list of line styles and colors for each side effect
line_styles = ['-', '--', '-.', ':']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# Create dictionaries to store the mapping of side effects to line styles and colors
side_effect_line_styles = {}
side_effect_colors = {}

# Iterate over unique side_effects and assign line styles and colors
for i, (side_effect, group) in enumerate(grouped_df.groupby('side_effect')):
    side_effect_line_styles[side_effect] = line_styles[i % len(line_styles)]
    side_effect_colors[side_effect] = colors[i % len(colors)]

# Iterate over unique side_effects and plot a curve for each with different line styles and colors
for side_effect, group in grouped_df.groupby('side_effect'):
    time_intervals = group['time_interval']
    frequencies = group['sum_side_effect_frequency']

    # Plot the curve for the current side_effect with a logarithmic y-axis and assigned line style and color
    ax.semilogy(time_intervals, frequencies, label=side_effect, linestyle=side_effect_line_styles[side_effect], color=side_effect_colors[side_effect])

# Set labels and title
ax.set_xlabel('Time Interval')
ax.set_ylabel('Log of Sum of Side Effect Frequency')

# Set x-axis limit to go up to 32
ax.set_xlim(26, 36)

# Place the legend to the right of the figure and adjust the bbox_to_anchor
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

# Save the plot to a PDF file
plt.savefig('side_effect_over_time.pdf', bbox_inches='tight')

# Show the plot
plt.show()

# how many days are in the last time interval

result_df['time_interval'].value_counts()

# Group by 'searchQuery' and 'entity', then count the occurrences
grouped = drug_side_effect_df.groupby(['drug', 'entity']).size().reset_index(name='count')

# Sort the DataFrame by count in descending order
sorted_df = grouped.sort_values(by='count', ascending=False)

# Display the result
sorted_df

import matplotlib.pyplot as plt
import seaborn as sns

# Pivot the DataFrame to prepare it for the heatmap
pivot_df = sorted_df.pivot(index='drug', columns='entity', values='count')

# Define the number of values to show in each subplot
values_per_subplot = 20

# Calculate the number of subplots needed
num_subplots = (len(pivot_df.columns) + values_per_subplot - 1) // values_per_subplot

# Create separate figures with one heatmap per line
for i in range(num_subplots):
    start_idx = i * values_per_subplot
    end_idx = (i + 1) * values_per_subplot
    subset_df = pivot_df.iloc[:, start_idx:end_idx]

    plt.figure(figsize=(10, 3))  # Adjust the figure size as needed
    sns.heatmap(subset_df, annot=True, cmap='YlGnBu', fmt='g')

    # Set labels for the axes
    plt.xlabel('Side Effect')
    plt.ylabel('GLP-1 Drug')

    plt.savefig(f'heatmap_{i}.pdf', format='pdf', bbox_inches='tight')

    # Show the plot
    plt.show()

    plt.close()  # Close the current figure

"""## **Edgelist**"""

# Perform a self-join and filter for A.entity > B.entity
edge_list = flattened_list.merge(flattened_list, on='id', suffixes=('_A', '_B'))
edge_list = edge_list[edge_list['stem_A'] > edge_list['stem_B']]

# Group by B.entity and A.entity, then count occurrences
edge_list = edge_list.groupby(['stem_A', 'stem_B']).size().reset_index(name='count')

edge_list[edge_list['count']>0]

edge_list.to_csv(os.path.join(PTH,'Reddit_side_efect_edgelist.csv'),index=False)

"""# **4. Figures**

### 4.1 wordcloud of clusters
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

se = pd.read_csv(os.path.join(PTH,'symptomps_cluster_df.csv'))
se['entity']=se['entity'].astype(str)

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
symptoms = se['entity'].tolist()
symptoms = [ x.lower() for x in symptoms]
symptoms = preprocess_and_combine(symptoms)

words_to_remove = ['weight','diabetes','loss','diabetic','disorder','d','cancer','adhd','metformin','disease','calorie']

word_frequency = plotWordCloud(symptoms, words_to_remove)

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

top_words

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
