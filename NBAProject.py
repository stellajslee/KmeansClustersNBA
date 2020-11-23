# This code uses a machine learning algorithm called KMeans to group
# NBA Hall of Famers and NBA All-Stars of 2018-19 and 2019-20 in clusters.
# It will show which players are similar to certain Hall of Famers!

# Import dependencies needed
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Get csv file with the data
data = pd.read_csv('nba_players.csv')
# print(data.head(5)) # Check
# print(data.shape) # Check

# Make the cluster of players using KMeans
kmeans_model = KMeans(n_clusters=5, random_state=0)

# Clean data
cleaned_data = data.drop(['Age', 'Seasons', 'Tm', 'Lg', 'Pos', 'G'], axis='columns')
cleaned_data.set_index('Player', drop=True, inplace=True)
# print(cleaned_data.head(5)) # Check

# Train model with cleaned data
kmeans_model.fit(cleaned_data)
labels = kmeans_model.labels_ # Get the cluster label for each player
# print(labels) # Check

# Plot players by cluster
pca2 = PCA(2) # 2 dimensions
plot_columns = pca2.fit_transform(cleaned_data) # Get x and y coordinates
# print(plot_columns) # Check

# Scatter plot
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
# plt.show() # Check

# Testing KMeans model
# Find Kawhi Leonard in the data
Kawhi_L = cleaned_data.loc[cleaned_data.index == 'Kawhi Leonard', :]
# Covert data into a list for model prediction
Kawhi_list = Kawhi_L.values.tolist()
# Cluster classification of Kawhi Leonard
Kawhi_cluster = kmeans_model.predict(Kawhi_list)
# print(Kawhi_cluster) # Check; Kawhi belongs to cluster 3

# Cluster all the players
cleaned_data["Cluster"] = kmeans_model.fit_predict(cleaned_data[cleaned_data.columns[1:]])
# print(cleaned_data.tail(5)) # Check

# Get players in each cluster
# Players in cluster 0
zero = cleaned_data[cleaned_data['Cluster'] == 0]
zero_list = zero.index.to_list()
# print(zero_list)

# Players in cluster 1
one = cleaned_data[cleaned_data['Cluster'] == 1]
one_list = one.index.to_list()
# print(one_list)

# Players in cluster 2
two = cleaned_data[cleaned_data['Cluster'] == 2]
two_list = two.index.to_list()
# print(two_list)

# Players in cluster 3
three = cleaned_data[cleaned_data['Cluster'] == 3]
three_list = three.index.to_list()
# print(three_list)

# Players in cluster 4
four = cleaned_data[cleaned_data['Cluster'] == 4]
four_list = four.index.to_list()
# print(four_list)

# Creating WebApp
# Make a title and subtitle
st.write("""
# NBA Basketball HOF and All Stars
Using a machine learning algorithm called KMeans to group NBA Hall of Famers and 2018-19 and 2019-20 NBA All-Stars into 5 different clusters!
""")

# Open and display image
image = Image.open('hofimage.jpg')
st.image(image, caption='Tim Duncan, Kobe Bryant, Kevin Garnett', use_column_width=True)

# A bit of explanation
st.write('I found a dataset called "NBA Players Peak Age - All of Hall of Famers" on kaggle.com with statistics of 152 Hall of Famers and their best performance seasons determined by summing up the normalized key statistics shown below:')

st.write('1. FG%, FT%, eFG%, AST, PTS (Offensive Ability)')
st.write('2. BLK (Defensive Ability)')
st.write('3. TRB, STL (Ability to Get The Ball)')
st.write('4. TOV, PF (Inability to Keep Ball Possession)')
st.write('Performance = nFG% + nFT% + neFG% + nAST + nPTS + nBLK + nTRB + nSTL – nTOV – nPF')
st.write('Then, I selected all the Hall of Famers whose best performance season was after 1980-81.')

st.write('For the All-Stars, the most recent active season statistics were used!')

# Display my dataset
st.subheader('Dataset: ')
st.dataframe(data)

# Display statistics on the data
st.write(data.describe())

# Show the data as a chart
image2 = Image.open('scatterplot.png')
st.image(image2, caption='Scatter plot of clusters', use_column_width=True)

# Get the cluster from user
user_input_cluster = st.sidebar.slider('Cluster', 0, 4, 2)

# Set a subheader for user input
st.header('Which cluster do you want to see?')
st.write(user_input_cluster)

# Display the players in selected cluster
st.subheader('Players in selected cluster: ')
selected_cluster = cleaned_data[cleaned_data['Cluster'] == user_input_cluster]
cluster_list = selected_cluster.index.to_list()
st.write(cluster_list)

# Options to see the cluster for some of the most popular players!
st.subheader('Find some of the most popular players!')
player_name = st.selectbox("Select player", ("Michael Jordan", "Lebron James", "Kawhi Leonard", "Tim Duncan",
                                             "Jimmy Butler", "Kobe Bryant", "Anthony Davis", "Chris Paul",
                                             "Dennis Rodman", "Joel Embiid"))
player_data = cleaned_data.loc[cleaned_data.index == player_name, 'Cluster']
player_cluster = player_data
st.write(player_cluster)

st.write(" * ")
st.write(" * ")
st.write(" * ")
st.write(" * ")
st.write(" * ")
st.write('Image used: ESPN Illustration. “Tim Duncan, Kobe Bryant and Kevin Garnett Were Named as Part of the 2020 Naismith Memorial Basketball Hall of Fame Class on Saturday.” Espn.com, Kevin Pelton, 1 Apr. 2020, www.espn.com/nba/story/_/id/28983161/where-kobe-bryant-tim-duncan-kevin-garnett-rank-greatest-hall-fame-classes.')
