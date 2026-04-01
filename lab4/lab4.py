from gensim.downloader import load 
from transformers import pipeline 
import nltk 
import string 
from nltk.tokenize import word_tokenize 
nltk.download('punkt_tab') 
print("loading pre trained word vectors") 
word_vectors=load("glove-wiki-gigaword-100") 
 
def replace_keyword_in_prompt(prompt,keyword,word_vectors,topn=1): 
  words=word_tokenize(prompt) 
  enriched_words=[] 
  for word in words: 
    cleaned_word=word.lower().strip(string.punctuation) 
    if cleaned_word==keyword.lower(): 
      try: 
        similar_words=word_vectors.most_similar(cleaned_word,topn=topn) 
        if similar_words: 
          replacement_word=similar_words[0][0] 
          print(f"Replacing {word}-> {replacement_word}") 
          enriched_words.append(replacement_word) 
          continue 
      except KeyError: 
        print(f"{keyword} not found in vocabulary using original word") 
        enriched_words.append(word) 
        
  enriched_prompt=" ".join(enriched_words) 
  print(f"\n Enriched Prompt:{enriched_prompt}") 
  return enriched_prompt 


print("\n Loading GPT-4 model") 
generator=pipeline("text-generation",model="gpt2") 
 
def generate_response(prompt,max_length=100): 
  try: 
    response=generator(prompt,max_length=max_length,num_return_sequences=1) 
    return response[0]['generated_text'] 
  except Exception as e: 
    print(f"error generating response {e}") 
    return None 
  
original_prompt="write an essay on natural disaster" 
print(f"Original prompt: {original_prompt}") 
k_term="disaster" 
enriched_prompt=replace_keyword_in_prompt(original_prompt,k_term,word_vectors) 
print("\n generating response for original prompt") 
original_response=generate_response(original_prompt) 
print(original_response) 
 
print("\n generating response for enriched prompt") 
enriched_response= generate_response(enriched_prompt) 
print(enriched_response) 
 
print("\n comparison of responses") 
print("original prompt response length",len(original_response)) 
print("enriched prompt response length",len(enriched_response)) 
print("original prompt response detail",original_response.count(".")) 
print("enriched prompt response detail",enriched_response.count("."))