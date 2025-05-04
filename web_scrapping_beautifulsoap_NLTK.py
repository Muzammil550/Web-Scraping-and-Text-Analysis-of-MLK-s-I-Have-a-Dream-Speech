# -*- coding: utf-8 -*-
"""
Web Scraping and Text Analysis of MLK's "I Have a Dream" Speech using beautifulsoap + req + NLTK

"""


#%% Import libraries
import requests
from bs4 import BeautifulSoup
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from wordcloud import WordCloud
import seaborn as sns
import pandas as pd
import nltk
from nltk.corpus import stopwords
#%% Download necessary NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)


# Set the page appearance for better visualization
plt.style.use('ggplot')


# Define the URL of the speech
url = 'http://www.analytictech.com/mb021/mlk.htm'
#%% Send HTTP request
response = requests.get(url, timeout=10)
response.raise_for_status()  # Raise an exception for HTTP errors


#%% Parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')


# Extract all paragraphs and clean the text
p_elements = soup.find_all('p')


# Process paragraphs and join them with proper spacing
paragraphs = []
for p in p_elements:
    text = p.text.strip()
    if text:  # Skip empty paragraphs
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        paragraphs.append(text)


# Join all paragraphs to create the complete speech
speech = ' '.join(paragraphs)


# Display the first 300 characters of the speech
print("First 300 characters of the speech:")
print(speech[:300] + "...\n")


# Count the total words in the speech
words = speech.split()
total_words = len(words)
print(f"Total word count: {total_words}")


#%% Basic text preprocessing for analysis
# Convert to lowercase
speech_lower = speech.lower()


# Remove punctuation
speech_no_punct = re.sub(r'[^\w\s]', '', speech_lower)


# Tokenize the text
words = speech_no_punct.split()


# Remove stop words (common words like 'the', 'a', 'an', etc.)
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word not in stop_words] # If words are not in stop words - give those


# Count word frequencies
word_freq = Counter(filtered_words)
most_common_words = word_freq.most_common(10)


# Prepare data for visualization
common_words_df = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])



# Create sentence-level analysis
sentences = nltk.sent_tokenize(speech)
sentence_lengths = [len(nltk.word_tokenize(sentence)) for sentence in sentences]


#%%# Visualization 1: Bar chart of most frequent words
ax=sns.barplot(x='Word', y='Frequency', data=common_words_df)
plt.title("Top 10 Most Frequent Words in MLK's 'I Have a Dream' Speech",fontsize=16)
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)
ax.set_xlabel("Word", fontsize=16) 
ax.set_ylabel("Frequency", fontsize=16) 
plt.tight_layout()
#%% Visualization 2: Word Cloud
wordcloud = WordCloud(width=800, height=400, background_color='white', 
                     max_words=50, contour_width=3, contour_color='steelblue')
wordcloud.generate(' '.join(filtered_words))

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of MLK's 'I Have a Dream' Speech")
plt.tight_layout()
#%% Visualization 3: Distribution of sentence lengths
plt.hist(sentence_lengths, bins=15, alpha=0.7, color='skyblue')
plt.axvline(np.mean(sentence_lengths), color='red', linestyle='dashed', linewidth=2)
plt.title("Distribution of Sentence Lengths in MLK's Speech",fontsize=16)
plt.xlabel("Number of Words in Sentence",fontsize=16)
plt.ylabel("Frequency",fontsize=16)
plt.annotate(f'Mean: {np.mean(sentence_lengths):.1f} words', 
            xy=(np.mean(sentence_lengths), plt.ylim()[1]*0.9),
            xytext=(np.mean(sentence_lengths)+5, plt.ylim()[1]*0.9),
            arrowprops=dict(arrowstyle='->'))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
#%% Visualization 4: Word frequency over the speech (simplified)
# Split the speech into segments for analysis
segments = 5
words_per_segment = len(filtered_words) // segments
segment_data = []


for i in range(segments):
    start = i * words_per_segment
    end = start + words_per_segment if i < segments - 1 else len(filtered_words)
    segment_words = filtered_words[start:end]
    segment_counter = Counter(segment_words)
    segment_data.append({
        'Segment': i+1,
        'dream': segment_counter.get('dream', 0),
        'freedom': segment_counter.get('freedom', 0),
        'justice': segment_counter.get('justice', 0)
    })


segment_df = pd.DataFrame(segment_data)


plt.plot(segment_df['Segment'], segment_df['dream'], marker='o', label='dream')
plt.plot(segment_df['Segment'], segment_df['freedom'], marker='s', label='freedom')
plt.plot(segment_df['Segment'], segment_df['justice'], marker='^', label='justice')
plt.title("Frequency of Key Words Throughout the Speech",fontsize=16)
plt.xlabel("Speech Segment",fontsize=16)
plt.ylabel("Word Frequency",fontsize=16)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(segment_df['Segment'],fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
#%% Add error handling for the request
try:
    # Send HTTP request
    response = requests.get(url, timeout=10)
    response.raise_for_status()  # Raise an exception for HTTP errors
    
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract all paragraphs and clean the text
    p_elements = soup.find_all('p')
    
    # Process paragraphs and join them with proper spacing
    paragraphs = []
    for p in p_elements:
        text = p.text.strip()
        if text:  # Skip empty paragraphs
            text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
            paragraphs.append(text)
    
    # Join all paragraphs to create the complete speech
    speech = ' '.join(paragraphs)
    
    # Display the first 300 characters of the speech
    print("First 300 characters of the speech:")
    print(speech[:300] + "...\n")
    
    # Count the total words in the speech
    words = speech.split()
    total_words = len(words)
    print(f"Total word count: {total_words}")
    
    # Basic text preprocessing for analysis
    # Convert to lowercase
    speech_lower = speech.lower()
    
    # Remove punctuation
    speech_no_punct = re.sub(r'[^\w\s]', '', speech_lower)
    
    # Tokenize the text
    words = speech_no_punct.split()
    
    # Remove stop words (common words like 'the', 'a', 'an', etc.)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    
    # Count word frequencies
    word_freq = Counter(filtered_words)
    most_common_words = word_freq.most_common(10)
    
    # Prepare data for visualization
    common_words_df = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])
    
    # Create sentence-level analysis
    sentences = nltk.sent_tokenize(speech)
    sentence_lengths = [len(nltk.word_tokenize(sentence)) for sentence in sentences]
    
    # Visualization 1: Bar chart of most frequent words
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Word', y='Frequency', data=common_words_df)
    plt.title("Top 10 Most Frequent Words in MLK's 'I Have a Dream' Speech")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('mlk_word_frequency.png', dpi=300)
    plt.show()
    
    # Visualization 2: Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                         max_words=100, contour_width=3, contour_color='steelblue')
    wordcloud.generate(' '.join(filtered_words))
    
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud of MLK's 'I Have a Dream' Speech")
    plt.tight_layout()
    plt.savefig('mlk_wordcloud.png', dpi=300)
    plt.show()
    
    # Visualization 3: Distribution of sentence lengths
    plt.figure(figsize=(12, 6))
    plt.hist(sentence_lengths, bins=15, alpha=0.7, color='skyblue')
    plt.axvline(np.mean(sentence_lengths), color='red', linestyle='dashed', linewidth=1)
    plt.title("Distribution of Sentence Lengths in MLK's Speech")
    plt.xlabel("Number of Words in Sentence")
    plt.ylabel("Frequency")
    plt.annotate(f'Mean: {np.mean(sentence_lengths):.1f} words', 
                xy=(np.mean(sentence_lengths), plt.ylim()[1]*0.9),
                xytext=(np.mean(sentence_lengths)+5, plt.ylim()[1]*0.9),
                arrowprops=dict(arrowstyle='->'))
    plt.tight_layout()
    plt.savefig('mlk_sentence_length.png', dpi=300)
    plt.show()
    
    # Visualization 4: Word frequency over the speech (simplified)
    # Split the speech into segments for analysis
    segments = 5
    words_per_segment = len(filtered_words) // segments
    segment_data = []
    
    for i in range(segments):
        start = i * words_per_segment
        end = start + words_per_segment if i < segments - 1 else len(filtered_words)
        segment_words = filtered_words[start:end]
        segment_counter = Counter(segment_words)
        segment_data.append({
            'Segment': i+1,
            'dream': segment_counter.get('dream', 0),
            'freedom': segment_counter.get('freedom', 0),
            'justice': segment_counter.get('justice', 0)
        })
    
    segment_df = pd.DataFrame(segment_data)
    
    plt.figure(figsize=(12, 6))
    plt.plot(segment_df['Segment'], segment_df['dream'], marker='o', label='dream')
    plt.plot(segment_df['Segment'], segment_df['freedom'], marker='s', label='freedom')
    plt.plot(segment_df['Segment'], segment_df['justice'], marker='^', label='justice')
    plt.title("Frequency of Key Words Throughout the Speech")
    plt.xlabel("Speech Segment")
    plt.ylabel("Word Frequency")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(segment_df['Segment'])
    plt.tight_layout()
    plt.savefig('mlk_word_progression.png', dpi=300)
    plt.show()
    
except requests.exceptions.RequestException as e:
    print(f"Error fetching the webpage: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
