PTH = '...'

import os
import pandas as pd

"""# Load data"""

# load Twitter data
df_twitter = pd.read_csv(os.path.join(PTH,'Twitter_GLP1','twitter_scrap.csv'))
df_twitter = df_twitter[['id','text','searchQuery']]
df_twitter

df_reddit = pd.read_csv(os.path.join(PTH,'Reddit_GLP1','reddit1_combined_new.csv'))
df_reddit_2  = pd.read_csv(os.path.join(PTH,'Reddit_GLP1','FirstRound','reddit1_combined.csv'))

df_reddit['Combined_Content'] = df_reddit['Post Content'].fillna('') + ' ' + df_reddit['Comment Content'].fillna('')
df_reddit = df_reddit[['Post ID','Combined_Content','subreddit']]


df_reddit_2['Combined_Content'] = df_reddit_2['Post Content'].fillna('') + ' ' + df_reddit_2['Comment Content'].fillna('')
df_reddit_2 = df_reddit_2[['Post ID','Combined_Content','subreddit']]

df_reddit = pd.concat([df_reddit, df_reddit_2])

df_reddit

"""# Sentiment analysis Twitter"""

import pandas as pd
from textblob import TextBlob

# Define a function to perform sentiment analysis and return sentiment scores
def analyze_sentiment(tweet):
    if isinstance(tweet, str):
        analysis = TextBlob(tweet)
        return analysis.sentiment.polarity, analysis.sentiment.subjectivity
    else:
        return 0.0, 0.0  # Replace missing values with neutral sentiment

# Apply the sentiment analysis function to the 'text' column
df_twitter[['polarity', 'subjectivity']] = df_twitter['text'].apply(
    lambda tweet: pd.Series(analyze_sentiment(tweet))
)

# Function to categorize sentiment based on polarity score
def categorize_sentiment(polarity):
    if polarity > 0:
        return 'pos'
    elif polarity < 0:
        return 'neg'
    else:
        return 'neutral'

# Apply sentiment categorization
df_twitter['sentiment'] = df_twitter['polarity'].apply(categorize_sentiment)

# Save the DataFrame with sentiment results to a new file or overwrite the existing DataFrame
df_twitter.to_csv(os.path.join(PTH,'Twitter_GLP1','tweets_with_sentiment.csv'), index=False)

df_twitter

"""# Sentiment analysis Reddit"""

import pandas as pd
from textblob import TextBlob

# Define a function to perform sentiment analysis and return sentiment scores
def analyze_sentiment(tweet):
    if isinstance(tweet, str):
        analysis = TextBlob(tweet)
        return analysis.sentiment.polarity, analysis.sentiment.subjectivity
    else:
        return 0.0, 0.0  # Replace missing values with neutral sentiment

# Apply the sentiment analysis function to the 'text' column
df_reddit[['polarity', 'subjectivity']] = df_reddit['Combined_Content'].apply(
    lambda tweet: pd.Series(analyze_sentiment(tweet))
)

# Function to categorize sentiment based on polarity score
def categorize_sentiment(polarity):
    if polarity > 0:
        return 'pos'
    elif polarity < 0:
        return 'neg'
    else:
        return 'neutral'

# Apply sentiment categorization
df_reddit['sentiment'] = df_reddit['polarity'].apply(categorize_sentiment)

# Save the DataFrame with sentiment results to a new file or overwrite the existing DataFrame
df_reddit.to_csv(os.path.join(PTH,'Reddit_GLP1','tweets_with_sentiment.csv'), index=False)

"""# Violon plots Twitter"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Assuming df_filtered is your DataFrame
replacements = {
    'Trulicity': 'Dulaglutide',
    'Byetta': 'Exenatide',
    'Bydureon': 'Exenatide',
    'Victoza': 'Liraglutide',
    'Adlyxin': 'Lixisenatide',
    'Ozempic': 'Semaglutide',
    'Rybelsus': 'Semaglutide'
}

# Replace values in the 'searchQuery' column
df_twitter['searchQuery'] = df_twitter['searchQuery'].replace(replacements)

df_twitter['searchQuery'].value_counts()

# Filter out neutral sentiment
df_filtered = df_twitter[df_twitter['sentiment'].isin(['pos', 'neg'])]

# Create a violin plot with sentiment on the x-axis and polarity on the y-axis, and separate by 'searchQuery'
plt.figure(figsize=(8, 6))
sns.violinplot(x='sentiment', y='polarity', data=df_filtered, inner='box', hue='searchQuery', dodge=True)
plt.title('Sentiment Analysis by Group (Polarity)')
plt.xlabel('Sentiment')
plt.ylabel('Sentiment Polarity')

# Create a custom legend with two rows
custom_legend = plt.legend(title='Group', loc='upper right', bbox_to_anchor=(1, 1), ncol=2)

# Save the plot as a PDF
plt.savefig("tukey_hsd_plot_Twitter.pdf", bbox_inches='tight')

plt.show()

"""## Anova tests to Twitter

a p-value of less than 0.05, means that there are significant differences among the groups.

It doesn't mean that all groups significantly differ, but it does indicate that there is a difference among at least one pair of groups.

To determine which specific groups differ from one another, post-hoc tests (e.g., Tukey's HSD, Bonferroni) can be performed.
"""

# Filter out neutral sentiment and select only 'pos'
df_filtered = df_twitter[(df_twitter['sentiment'] == 'pos')]

# Perform t-tests for significant differences among the 'pos' groups
grouped_data = [group_data['polarity'] for _, group_data in df_filtered.groupby('searchQuery')]
t_stat, p_value = stats.f_oneway(*grouped_data)
print("One-way ANOVA F-statistic for Pos group:", t_stat)
print("P-value:", p_value)

# Filter out neutral sentiment and select only 'pos'
df_filtered = df_twitter[(df_twitter['sentiment'] == 'neg')]

# Perform t-tests for significant differences among the 'pos' groups
grouped_data = [group_data['polarity'] for _, group_data in df_filtered.groupby('searchQuery')]
t_stat, p_value = stats.f_oneway(*grouped_data)
print("One-way ANOVA F-statistic for Neg group:", t_stat)
print("P-value:", p_value)

"""### Tukey's HSD (Honestly Significant Difference) test: pairwise group comparisons to identify which groups are statistically significant after a one-way ANOVA


"""

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multicomp import MultiComparison
import seaborn as sns
import matplotlib.pyplot as plt

df = df_twitter[(df_twitter['sentiment'] == 'pos')]

# Step 1: Perform Tukey's HSD test
mc = MultiComparison(df['polarity'], df['searchQuery'])
result = mc.tukeyhsd()

# Step 2: Visualize using a heatmap
# Create a pivot table for the Tukey's HSD results
tukey_df = pd.DataFrame(data=result._results_table.data[1:], columns=result._results_table.data[0])

# Convert p-values to numeric and extract the mean differences
tukey_df['p-adj'] = pd.to_numeric(tukey_df['p-adj'])
tukey_df['meandiff'] = pd.to_numeric(tukey_df['meandiff'])

# Create a heatmap
plt.figure(figsize=(10, 6))
heatmap_data = tukey_df.pivot(index='group1', columns='group2', values='p-adj')

# Create a heatmap with p-values
heatmap = sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, cbar_kws={'label': 'p-value'})



# Save the plot as a PDF
plt.savefig("tukey_hearmap_positive_x.pdf", bbox_inches='tight')

plt.title("Tukey's HSD Post-Hoc Test")

# Add a legend
cbar = heatmap.collections[0].colorbar
cbar.set_label('p-value')

plt.show()

df = df_twitter[(df_twitter['sentiment'] == 'neg')]

# Step 1: Perform Tukey's HSD test
mc = MultiComparison(df['polarity'], df['searchQuery'])
result = mc.tukeyhsd()

# Step 2: Visualize using a heatmap
# Create a pivot table for the Tukey's HSD results
tukey_df = pd.DataFrame(data=result._results_table.data[1:], columns=result._results_table.data[0])

# Convert p-values to numeric and extract the mean differences
tukey_df['p-adj'] = pd.to_numeric(tukey_df['p-adj'])
tukey_df['meandiff'] = pd.to_numeric(tukey_df['meandiff'])

# Create a heatmap
plt.figure(figsize=(10, 6))
heatmap_data = tukey_df.pivot(index='group1', columns='group2', values='p-adj')

# Create a heatmap with p-values
heatmap = sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", cbar=True, cbar_kws={'label': 'p-value'})



# Save the plot as a PDF
plt.savefig("tukey_hearmap_x_negative.pdf", bbox_inches='tight')

plt.title("Tukey's HSD Post-Hoc Test")

# Add a legend
cbar = heatmap.collections[0].colorbar
cbar.set_label('p-value')

plt.show()
