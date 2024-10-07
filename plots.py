# -*- coding: utf-8 -*-

import pandas as pd
import os
import matplotlib.pyplot as plt

PTH = '/.../data'

edges = pd.read_csv(os.path.join(PTH,'g_edgelist.csv')) # no pubmed
edges.columns = ['node_from', 'node_to','weight']
edges

nodes = pd.concat(
              [
                  edges[['node_from','weight']].rename(columns={'node_from': 'node'}),
                  edges[['node_to','weight']].rename(columns={'node_to': 'node'})
              ],
              axis = 0
).reset_index(drop=True)


nodes = nodes.groupby('node')['weight'].sum().reset_index()

nodes

nodes['weight'].sum()

"""# Define side effect frequency

The frequency of a drug side effect in the population can be:

* very rare (<1 in 10,000),
* rare (1 in 10,000 to 1 in 1000),
* infrequent (1 in 1000 to 1 in 100),
* frequent (1 in 100 to 1 in 10), or
* very frequent (>1 in 10)
"""

!pip install wordcloud matplotlib

nodes['freq'] = nodes['weight']/nodes['weight'].max()

nodes.loc[ (nodes['freq'] < 0.0001),'group'] = 'very rare'
nodes.loc[ (nodes['freq'] >= 0.0001) & (nodes['freq'] < 0.001) ,'group'] = 'rare'
nodes.loc[ (nodes['freq'] >= 0.001) & (nodes['freq'] < 0.01) ,'group'] = 'infrequent'
nodes.loc[ (nodes['freq'] >= 0.01) & (nodes['freq'] < 0.1) ,'group'] = 'frequent'
nodes.loc[ (nodes['freq'] >= 0.1),'group'] = 'very frequent'

nodes['group'].value_counts()

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have a DataFrame 'nodes' with a 'group' column
word_counts = nodes['group'].value_counts().reindex(["very rare", "rare", "infrequent", "frequent", "very frequent"])

sns.barplot(x=word_counts.index, y=word_counts.values, color='lightblue')

# Add labels and a title
plt.xlabel('Side effect frequency class value')
plt.ylabel('Counts')
plt.title('Side Effect Counts')

# Add numbers on top of each bar
for i, count in enumerate(word_counts.values):
    label = str(count) if count > 0 else "0"
    plt.text(i, count, label, ha='center', va='center', fontsize=11)

# Save the figure as a PDF
plt.savefig("barplot_side_effect_frequency_class.pdf", format="pdf")

# Show the plot
plt.show()

nodes['group'].value_counts()

import matplotlib.pyplot as plt
import numpy as np

# Sort the DataFrame by "Number of Drugs" in descending order
df = nodes.sort_values(by='freq', ascending=False)

# Define the order in which groups should be plotted
group_order = ['very frequent', 'frequent', 'infrequent', 'rare', 'very rare']  # Customize this order

# Define a colormap for "group" values
colormap = plt.get_cmap('tab10')  # You can choose a different colormap


# Create the line plot with color mapping
plt.figure(figsize=(10, 5))  # Adjust the figure size as needed
for group in group_order:
    group_df = df[df['group'] == group]
    if not group_df.empty:  # Check if the group has members
        color = colormap(group_order.index(group) % 10)  # Ensure it loops through colors if there are more than 10 groups
        plt.plot(group_df['node'], group_df['freq'], marker='o', linestyle='-', label=f'{group}', color=color)

# Rotate the x-axis labels for better readability
plt.xticks(rotation=90, ha='right')

# Add labels and title
plt.xlabel('Side Effect')
plt.ylabel('Fraction of Side Effect')
plt.title('Long-Tailed Distribution of Side Effects')

# Add a legend to distinguish groups for the groups with data
plt.legend()

# Adjust the layout to prevent truncation of x-axis labels
plt.tight_layout()

# Save the figure as a PDF
plt.savefig("side_effect_distribution.pdf", format="pdf")

# Show the plot
plt.show()
