import random 
from gensim.downloader import load 
 

print("loading the glove model") 
model = load("glove-wiki-gigaword-200") 
 
def get_similar_words(seed_word, top_n=50): 
    """Retrieve similar words for the given seed word.""" 
    try: 
        similar_words = [word for word, _ in model.most_similar(seed_word, 
topn=top_n)] 
        return similar_words 
    except KeyError: 
        return [] 
 
def create_paragraph(seed_word): 
    """Generate a short paragraph using the seed word and its similar words.""" 
    similar_words = get_similar_words(seed_word) 
 
    if not similar_words: 
        return f"Could not find similar words for '{seed_word}'. Try another word!" 
 
    paragraph = f"""Once upon a time, a {seed_word} embarked on a journey.
Along the way, it encountered a {random.choice(similar_words)}, which led it to a hidden {random.choice(similar_words)}.
Despite the challenges, it found {random.choice(similar_words)} and embraced the adventure with {random.choice(similar_words)}.
In the end, the journey was a tale of {random.choice(similar_words)} and discovery.""" 
 
    return paragraph 
 

seed_word = input("Enter a seed word: ").strip().lower() 
print("\nGenerated Story:\n") 
print(create_paragraph(seed_word))