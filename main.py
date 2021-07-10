import sys
import numpy as np
import pandas as pd
from parser import new_rules,get_dict
import ast

#import time
#time_begin = time.time()  # Time

# Getting entry files
content_file = sys.argv[1]
ratings_file = sys.argv[2]
targets_file = sys.argv[3]

# TFIDF (TODO MODULARIZAR)
def TFIDF(data, data_ids):
  n = len(data)

  all_words_id = [i for l, i in zip(data, data_ids) for aux in range(len(l))]
  all_words_list = [word for l in data for word in l]

  # Starting new DF
  words_df = pd.DataFrame(columns=['ItemID','Word']) 
  words_df['ItemID'] = all_words_id
  words_df['Word'] = all_words_list

  # TF
  words_df['TF'] = words_df.groupby(['ItemID']).transform('count')
  words_df['TF'] = words_df.groupby(['ItemID','Word']).transform('count')

  # Drop duplicates
  words_df = words_df.drop_duplicates().copy()

  # IDF
  words_df['count appearences'] = words_df.groupby(['Word'])['ItemID'].transform('count')
  words_df = words_df[words_df['count appearences'] > 1] # tirar palavras que aparecem uma vez

  words_df['IDF']  = np.log(n/words_df['count appearences'])

  # TF*IDF
  words_df['TFIDF'] = words_df['TF']*words_df['IDF']

  # Drop unnecessary
  words_df = words_df[['ItemID','Word','TFIDF']]
  return words_df

# Loads dos e manipoulação dos atributos de filme
with open(content_file, 'r') as f:
    f = f.readlines()

columns = f.pop(0)
columns = columns.strip()
columns = [columns.split(',')]

item = [l[:8] for l in f]
item_dict = [ast.literal_eval(l[9:].strip()) for l in f] 
#item_dict = [get_dict(l[9:].strip()) for l in f]

content_df = pd.DataFrame(item_dict)
content_df['ItemId'] = item

content_df['ItemId'] = content_df['ItemId'].transform(lambda x: int(x[1:]))
content_df = content_df[~content_df['Title'].isin([np.nan])].copy()

# Tokenizar Plots
pontuation = "'.,())-\":!#$/%&*+?°;[]^=_£"
content_df['tokenized Plot'] = content_df['Plot'].transform(lambda x: new_rules(x.lower().translate(str.maketrans('', '', pontuation)).split()) if x != 'N/A' else x)

content_df = content_df[['ItemId','Title','Year','Runtime','tokenized Plot','Language','Country','Awards','imdbRating','imdbVotes']]

## Realizar TFIDF com plot tokenizados
plots = content_df[content_df['tokenized Plot'] != 'N/A'][['tokenized Plot','ItemId']]
plots_ids, plots = plots['ItemId'].values, plots['tokenized Plot'].values
del content_df

words_df = TFIDF(plots, plots_ids)

#time_this = time.time()  # Time

## Gerar matrix com TFIDF de cada palavra em cada plot
language = pd.unique(words_df['Word'])
language.sort()
language_index = pd.DataFrame(language, columns=['Word'])
language_index['WordID'] = language_index.index
word_df_index = pd.merge(language_index, words_df, on="Word")
del words_df
#word_df_index = word_df_index[['WordID', 'ItemID', 'TFIDF']].copy()

lookup_item_id = {id:i for id, i in zip(plots_ids, range(len(plots_ids)))}
TFIDF_matrix = np.zeros((len(plots_ids),len(language)))
#print(TFIDF_matrix.shape)

## Calcular item similarity matrix
for index, row in word_df_index.iterrows():
  item_id = row['ItemID']
  word_id = row['WordID']
  TFIDF = row['TFIDF']
  TFIDF_matrix[lookup_item_id[item_id]][word_id] = TFIDF
del word_df_index

## Calcular item similarity matrix
norm = np.reshape(np.linalg.norm(TFIDF_matrix, axis=1), (1, TFIDF_matrix.shape[0]))
norm = norm * norm.T
TFIDF_matrix = TFIDF_matrix @ TFIDF_matrix.T
TFIDF_matrix = TFIDF_matrix / norm
del norm
item_similarity_matrix = TFIDF_matrix

#print("similarity:",time.time() - time_this)

## Carregar e manipular ratings
ratings_df = pd.read_csv(ratings_file)
ratings_df['UserId'] = ratings_df['UserId:ItemId'].transform(lambda x: int(x.split(":")[0][1:]))
ratings_df['ItemId'] = ratings_df['UserId:ItemId'].transform(lambda x: int(x.split(":")[1][1:]))
ratings_df = ratings_df[['UserId','ItemId','Prediction']]
mean = ratings_df['Prediction'].mean()
ratings_df = ratings_df[ratings_df['ItemId'].isin(plots_ids)]

## Gerar  matrix numpy com os ratings
unique_users = pd.unique(ratings_df['UserId'])
unique_users.sort() # removivel

lookup_users = {id:i for id, i in zip(unique_users, range(len(unique_users)))}

ratings_matrix = np.zeros((len(unique_users), len(plots_ids)))
for u,i,r in zip(ratings_df['UserId'].values, ratings_df['ItemId'].values, ratings_df['Prediction'].values):
  ratings_matrix[lookup_users[u]][lookup_item_id[i]] = r


## Carregar e manipular targets
targets_df = pd.read_csv(targets_file)
targets_df['UserId'] = targets_df['UserId:ItemId'].transform(lambda x: int(x.split(":")[0][1:]))
targets_df['ItemId'] = targets_df['UserId:ItemId'].transform(lambda x: int(x.split(":")[1][1:]))
targets_df = targets_df[['UserId','ItemId']]

## Realizar recomendações (IMPROVE, make all recommendations at once with matriz operation)
np.seterr(divide='ignore', invalid='ignore')
f = open("results.csv", "w")
f.write("UserId:ItemId,Prediction\n")

count = 0
for u,i in zip(targets_df['UserId'].values,targets_df['ItemId'].values):
  rate = -1.0
  if u in lookup_users and i in lookup_item_id:
    m_u = lookup_users[u]
    m_i = lookup_item_id[i]

    ratings_array = ratings_matrix[m_u,:]
    similarity_array = item_similarity_matrix[:,m_i]

    pos_array = np.where(ratings_array != 0.0)
    norm_similarity = [similarity_array[pos] for pos in pos_array]

    rate = similarity_array@ratings_array/np.sum(norm_similarity)
    if np.isnan(rate):
      rate = -1.0 
  
  if rate == -1.0:
    rate = mean
    count += 1

  string ="u{:07d}".format(u) + ":" + "i{:07d}".format(i) + "," + str(rate) + "\n"
  f.write(string)

f.close()

#print("Invalid:", count/len(targets_df))
#print("Total:",time.time() - time_begin)
