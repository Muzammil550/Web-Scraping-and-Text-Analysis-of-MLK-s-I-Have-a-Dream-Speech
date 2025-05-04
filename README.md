# 🧠 Web Scraping and Text Analysis of MLK's "I Have a Dream" Speech

This project demonstrates how to scrape and analyze Martin Luther King Jr.'s iconic "I Have a Dream" speech using Python. It utilizes:

- **Requests** and **BeautifulSoup** for web scraping
- **NLTK** for natural language processing tasks such as tokenization, stopword removal, and lemmatization
- **Regular Expressions (re)** for text cleaning

## 📄 Project Overview

The script performs the following steps:

1. **Web Scraping**: Fetches the speech text from a specified URL.
2. **Text Extraction**: Parses the HTML content to extract relevant text.
3. **Text Cleaning**: Removes unwanted characters and punctuation using regular expressions.
4. **Tokenization**: Splits the text into individual words.
5. **Stopword Removal**: Eliminates common English stopwords that do not contribute to the analysis.
6. **Lemmatization**: Reduces words to their base or dictionary form.
7. **Frequency Analysis**: Counts the frequency of each lemmatized word.

## 🛠️ Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Muzammil550/Web-Scraping-and-Text-Analysis-of-MLK-s-I-Have-a-Dream-Speech.git
   cd Web-Scraping-and-Text-Analysis-of-MLK-s-I-Have-a-Dream-Speech
Create a Virtual Environment (Optional but Recommended)

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
Note: Ensure that requirements.txt includes the necessary packages such as requests, beautifulsoup4, and nltk.

Download NLTK Data

The script requires certain NLTK datasets. You can download them using the following commands:

python
Copy
Edit
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
🚀 Usage
Run the script using Python:

bash
Copy
Edit
python web_scrapping_beautifulsoap_NLTK.py
The script will output:

The original text of the speech.

The cleaned and lemmatized text.

A list of the most frequent words used in the speech.

📊 Sample Output
Example of the top 10 most frequent words:

markdown
Copy
Edit
Word       Frequency
--------------------
freedom    20
justice    15
dream      12
nation     10
...
Note: Actual output may vary based on the source and preprocessing steps.

📁 Project Structure
bash
Copy
Edit
Web-Scraping-and-Text-Analysis-of-MLK-s-I-Have-a-Dream-Speech/
├── web_scrapping_beautifulsoap_NLTK.py  # Main script
├── requirements.txt                     # List of dependencies
└── README.md                            # Project documentation
📚 References
NLTK Documentation

BeautifulSoup Documentation

Requests Documentation

📝 License
This project is licensed under the MIT License. See the LICENSE file for details.
