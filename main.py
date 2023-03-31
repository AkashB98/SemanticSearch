# %%
#OpenAI Word Embeddings, Semantic Search
#Word embeddings are a way of representing words and phrases as vectors. They can be used for a variety of tasks, including semantic search, anomaly detection, 
# and classification. In the video on OpenAI Whisper, I mentioned how words whose vectors are numerically similar are also similar in semantic meaning. 
# In this tutorial, we will learn how to implement semantic search using OpenAI embeddings. Understanding the Embeddings concept will be crucial to the next 
# several videos in this series since we will use it to build several practical applications.

# %%
!pip install openai -q

# %%
import openai
import pandas as pd
import numpy as np
from getpass import getpass

openai.api_key = getpass()

# %%
#Read Data File Containing Words

#Now that we have configured OpenAI, let's start with a simple CSV file with familiar words. From here we'll build up to a more complex semantic search 
# using sentences from the Fed speech. Save the linked "words.csv" as a CSV and upload it to Google Colab. Once the file is uploaded, let's read it 
# into a pandas dataframe using the code below:

df = pd.read_csv('words.csv')
print(df)

# %%
#Calculate Word Embeddings

#To use word embeddings for semantic search, you first compute the embeddings for a corpus of text using a word embedding algorithm. What does 
# this mean? We are going to create a numerical representation of each of these words. To perform this computation, we'll use OpenAI's 'get_embedding' function.

#Since we have our words in a pandas dataframe, we can use "apply" to apply the get_embedding function to each row in the dataframe. 
# We then store the calculated word embeddings in a new text file called "word_embeddings.csv" so that we don't have to call OpenAI again to perform these calculations.

from openai.embeddings_utils import get_embedding

df['embedding'] = df['text'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
df.to_csv('word_embeddings.csv')

# %%
get_embedding("the fox crossed the road", engine='text-embedding-ada-002')

# %%
#Semantic Search
#Now that we have our word embeddings stored, let's load them into a new dataframe and use it for semantic search. Since the 'embedding' in the CSV is 
# stored as a string, we'll use apply() and to interpret this string as Python code and convert it to a numpy array so that we can perform calculations on it.

df = pd.read_csv('word_embeddings.csv')
df['embedding'] = df['embedding'].apply(eval).apply(np.array)
df

# %%
#Let's now prompt ourselves for a search term that isn't in the dataframe. We'll use word embeddings to perform a semantic 
# search for the words that are most similar to the word we entered. I'll first try the word "hot dog". Then we'll come back and try the word "yellow".

search_term = input('Enter a search term: ')


# %%
# semantic search

#Now that we have a search term, let's calculate an embedding or vector for that search term using the OpenAI get_embedding function.

search_term_vector = get_embedding(search_term, engine="text-embedding-ada-002")
search_term_vector

# %%
#Once we have a vector representing that word, we can see how similar it is to other words in our dataframe by calculating the cosine similarity of our search 
# term's word vector to each word embedding in our dataframe.
from openai.embeddings_utils import cosine_similarity

df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))

df

# %%
#Sorting By Similarity
#Now that we have calculated the similarities to each term in our dataframe, we simply sort the similarity values to find the terms 
# that are most similar to the term we searched for. Notice how the foods are most similar to "hot dog". Not only that, it puts fast 
# food closer to hot dog. Also some colors are ranked closer to hot dog than others. Let's go back and try the word "yellow" and walk through the results.

df.sort_values("similarities", ascending=False).head(20)

# %%
#Adding Words Together
#What's even more interesting is that we can add word vectors together. What happens when we add the numbers for milk and espresso, 
# then search for the word vector most similar to milk + espresso? Let's make a copy of the original dataframe and call it food_df. 
# We'll operate on this copy. Let's try adding word together. Let's add milk + espresso and store the results in milk_espresso_vector.

food_df = df.copy()

milk_vector = food_df['embedding'][10]
espresso_vector = food_df['embedding'][19]

milk_espresso_vector = milk_vector + espresso_vector
milk_espresso_vector

# %%
#Now let's find the words most similar to milk + espresso. If you have never done this before, it's pretty surprising that you 
# can add words together like this and find similar words using numbers.


food_df["similarities"] = food_df['embedding'].apply(lambda x: cosine_similarity(x, milk_espresso_vector))
food_df.sort_values("similarities", ascending=False)

# %%
#Microsoft Earnings Call Transcript
#Let's tie this back to finance. I have attached some text from a recent Microsoft earnings call here. Click on "raw" and save the file 
# as a CSV. Upload it to Google Colab as microsoft-earnings.csv. Let's use what we just learned to perform a semantic search on sentences in 
# the Microsoft earnings call. We'll start by reading the paragraphs into a pandas dataframe.


earnings_df = pd.read_csv('microsoft-earnings.csv')
earnings_df

# %%
#Once we have the dataframe, we'll once again compute the embeddings for each line in our CSV file.

earnings_df['embedding'] = earnings_df['text'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
earnings_df.to_csv('earnings-embeddings.csv')

# %%
#If you download the earnings_embeddings.csv file locally and open it up, you'll see that our embeddings are for 
# entire paragraphs - not just words. This means that we'll be able to search on similar sentences even if there isn't an exact
# match for the string we search for. We are searching on meaning.


#artificial intelligence demand cloud products
earnings_search = input("Search earnings for a sentence:")

# %%
earnings_search_vector = get_embedding(earnings_search, engine="text-embedding-ada-002")
earnings_search_vector

# %%
earnings_df["similarities"] = earnings_df['embedding'].apply(lambda x: cosine_similarity(x, earnings_search_vector))

earnings_df

# %%
earnings_df.sort_values("similarities", ascending=False)

# %%
#Sentences of the Fed Speech
#Let's use the Fed Speech example once more. Let's calculate the word embeddings for a particular sentence in the November 2nd speech that we 
# discussed in the OpenAI Whisper tutorial. Then we'll take a new sentence from a future speech that isn't in our dataset, and find the most similar 
# sentence in our dataset. Here is the sentence we will use to search for similarity:

#"the inflation is too damn high"
#As we did previously, take the linked CSV file and upload it to Google Colab as fed-speech.csv. We'll once again read it into a pandas dataframe.

fed_df = pd.read_csv('fed-speech.csv')
fed_df

# %%
#We'll once again calculate the embeddings and save them in a new CSV file.
fed_df['embedding'] = fed_df['text'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
fed_df.to_csv('fed-embeddings.csv')

# %%
#We'll then enter the new sentence that we want to find similarity for:
#"We will continue to increase interest rates and tighten monetary policy"
fed_sentence = input('Enter something Jerome Powell said: ')


# %%
#Enter something Jerome Powell said: the inflation is too damn high
#Again we'll get the vector for this sentence, find the cosine similarity, and sort by most similar.

fed_sentence_vector = get_embedding(fed_sentence, engine="text-embedding-ada-002")
fed_sentence_vector

# %%
fed_df = pd.read_csv('fed-embeddings.csv')
fed_df['embedding'] = fed_df['embedding'].apply(eval).apply(np.array)
fed_df


# %%
fed_df["similarities"] = fed_df['embedding'].apply(lambda x: cosine_similarity(x, fed_sentence_vector))

fed_df

# %%
fed_df.sort_values("similarities", ascending=False)

# %%
#Calculating Cosine Similarity
#We used the Cosine Similarity function, but how does it actually work? Cosine similarity is just calculating the similarity between two vectors. 
#There is a mathematical equation for calculating the angle between two vector.

v1 = np.array([1,2,3])
v2 = np.array([4,5,6])

# (1 * 4) + (2 * 5) + (3 * 6)
dot_product = np.dot(v1, v2)
dot_product

# %%
# square root of (1^2 + 2^2 + 3^2) = square root of (1+4+9) = square root of 14
np.linalg.norm(v1)

# %%
# square root of (4^2 + 5^2 + 6^2) = square root of (16+25+36) = square root of 14
np.linalg.norm(v2)

# %%
magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
magnitude

# %%
dot_product / magnitude


# %%
from scipy import spatial

result = 1 - spatial.distance.cosine(v1, v2)

result

# %%


# %%



