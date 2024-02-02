#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!pip install pandas openpyxl
import pandas as pd
import re
df = pd.read_excel("inputData/Social_Media_Instagram.xlsx")
df


# In[3]:


# Main task: Create a clean text that does not contain any special characters, links, hashtags or similar.
# Create a new column that specifies the number of characters for the cleaned text (without spaces). 

# Additional tasks
# Create new names (in English) for the columns.
# Create a new column containing the hashtags.
# Specifies the number of hashtags and the number of characters used (without spaces and "#") in 2 new columns.
# Clean up data. Delete unnecessary columns or rows

# Special bonus task (Bonus points!): Get along with the emojis (e.g.: the string "=Ø\tÞ")
# How many emojis are in the post and which emojis are in the post (e.g. sad, etc.)? 
# 3 bonus points for the module if the correct determination of these values is found for the entire data set.


# In[4]:


df


# In[5]:


# rename columns
df = df.rename(columns={"Titel des Beitrags":"text",
                        "Reichweite":"impressions",
                        "_x001E_ Gefällt mir_x001C_ -Angaben und Reaktionen":"likes",
                        "Kommentare":"comments",
                        })

# slice the data set
df = df.iloc[:,0:4]

# Examine the data set
df.describe()
# no missing values


# In[6]:


df.at[0,"text"]


# In[7]:


# extract hashtags
hashtags = df['text'].str.extractall(r'(#\w+)')[0] 
# str.extractall extracts all hashtags from the text column. (multiindex Dataframe) --> see hashtags
# "#\w+" searches for words beginning with a "#"
df['hashtags'] = hashtags.groupby(level=0).apply(' '.join)
# hashtags.groupby(level=0) --> The hashtags found in each row (in column "text") are assigned to a group.
# .apply(' '.join) --> merges all elements of the hashtag group (using groupby) into a single string.
df.head()


# In[7]:





# In[8]:


# count hashtags
df['hashtag_count'] = df['hashtags'].str.split().apply(len)
df.head()
# We seem to have empty values under hashtags


# In[ ]:


# Fill the values with an empty string
df['hashtags'] = df['hashtags'].fillna('')
df['hashtag_count'] = df['hashtags'].str.split().apply(len)
df.head()


# In[ ]:


# count the hashtag characters
df['hashtag_char_count'] = df['hashtags'].apply(lambda x: len(x.replace(" ", "").replace("#", "")))
df.head()


# In[ ]:


# Clean up text and extract emojis

# clear "\n"
df['text_clear'] = df['text'].str.replace(r'\n', ' ', regex=True)


# clear links
def remove_links(text):
    link_pattern = re.compile(r'www\.[^\s]+')
    text_without_links = re.sub(link_pattern, '', text)
    return text_without_links

df['text_clear'] = [remove_links(s) for s in df["text_clear"]]


# find and count emojis
def fix_spacing(text):
    # Replace "." with ". "
    text = re.sub(r'\.(?!\s)', '. ', text)
    
    # Replace "(" and ")" with " ( " and " ) "
    text = re.sub(r'(?<!\s)\((?!\s)', ' ( ', text)
    text = re.sub(r'(?<!\s)\)(?!\s)', ' ) ', text)

    
    return text
    
df['text_clear'] = df['text_clear'].apply(fix_spacing)

def process_text(text):
    # Divide the text into sub-strings using spaces
    substrings = text.split()

    # Create a list to save the sub-strings
    result_substrings = []

    # Run through the sub-strings and check for the "Ø" and so on characters
    for substring in substrings:
        if 'Ø' in substring:
            result_substrings.append(substring)
        if "_x" in substring:
            result_substrings.append(substring)        
        if "Þ" in substring:
            result_substrings.append(substring)
            
    # Merge the sub-strings found into one string, separated by spaces
    result = ' '.join(result_substrings)

    # Remove the sub-strings found from the original text
    cleaned_text = ' '.join([substring for substring in substrings if substring not in result_substrings])

    return result, cleaned_text

df[['emojis', 'text_clear']] = df['text_clear'].apply(process_text).apply(pd.Series)

df['emojis_count'] = df['emojis'].apply(lambda x: len(x.split()))
df

# Clear Hashtags
df['text_clear'] = df['text_clear'].str.replace(r'#\w+', '', regex=True)


# In[ ]:


df.at[12,"text_clear"]


# In[ ]:


# count characters
df['char_count'] = df['text_clear'].apply(lambda x: len(x.replace(" ", "")))
df.head()


# In[ ]:


df.to_excel("test.xlsx", index=False, engine='openpyxl')


# In[ ]:





# In[ ]:





# In[ ]:





# In[8]:





# In[8]:





# In[8]:





# In[8]:





# In[8]:





# In[8]:





# In[8]:





# In[8]:





# In[8]:





# In[8]:





# In[8]:





# In[8]:





# In[8]:





# In[8]:





# In[8]:




