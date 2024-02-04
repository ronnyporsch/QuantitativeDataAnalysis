#!/usr/bin/env python
# coding: utf-8
import numpy as np
# In[47]:


# !pip install pandas openpyxl
import pandas as pd
import re


from util.util import printDf


def renameAllColumns(df):
    df.rename(columns={'Titel des Beitrags': 'Original Title', 'Reichweite': 'Impressions',
                       '_x001E_ Gefällt mir_x001C_ -Angaben und Reaktionen': 'Likes', 'Kommentare': 'Comments',
                       'Geteilte Inhalte': 'Shared Content', 'Ergebnisse': 'Results',
                       'Kosten pro Ergebnis': 'Cost per result', 'Link-Klicks': 'Clicks on Link'}, inplace=True)


def getCleanVersion(string):
    # return ''.join(e for e in string if e.isalnum())
    return re.sub('[^A-Za-z0-9]+', '', string)


def addColumnCleanTitle(df):
    cleanTitles = []
    for title in df['Original Title']:
        cleanTitles.append(getCleanVersion(title))
    df.insert(1, "Clean Title", cleanTitles, True)


def addColumnCharCountCleaned(df):
    numChars = []
    for title in df['Clean Title']:
        numChars.append(len(title))
    df.insert(2, 'NumberOfCharsInCleanTitle', numChars, True)


def countEmojis(string):
    return string.count("=Ø") + string.count("<Ø") + string.count(">Ø")


def addColumnEmojiCount(df):
    emojiCount = []
    for title in df['Original Title']:
        emojiCount.append(countEmojis(title))
    df.insert(2, 'NumberOfEmojisOriginalTitle', emojiCount, True)


def findHashtags(string):
    regex = "#[A-Za-z0-9]+"
    return re.findall(regex, string)


def addColumnsHashtagsAndHashtagCount(df):
    hashtags = []
    hashtagCount = []
    for title in df['Original Title']:
        hashtagsOfTitle = findHashtags(title)
        hashtags.append(hashtagsOfTitle)
        hashtagCount.append(len(hashtagsOfTitle))
    df.insert(2, 'Hashtags', hashtags, True)
    df.insert(3, 'HashtagCount', hashtagCount, True)


def deleteUnnecessaryColumns(df):
    del df['Geteilte Inhalte']


df = pd.read_excel("inputData/Social_Media_Instagram.xlsx")
renameAllColumns(df)
addColumnCleanTitle(df)
addColumnCharCountCleaned(df)
addColumnEmojiCount(df)
addColumnsHashtagsAndHashtagCount(df)
printDf(df)


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

# df.to_excel("test.xlsx", index=False, engine='openpyxl')
