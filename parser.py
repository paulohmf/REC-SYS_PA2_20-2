# Parser das palavras
stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are","arent", "was","wasnt", "were","wasnt", "be", "been", "being", "have","havent", "has","hasnt", "had", "hadnt","having", "dont","do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
def new_rules(l):
  words = list()
  for word in l:
    new_word = rules(word)
    if type(new_word) != type(None) :
      words.append(new_word)
  return words

def rules(word):
  if word in stopwords:
    return None
  l = len(word)
  if l == 1:
    return None

  if l > 3:
    if word[-4:] == 'sses':
      return word[:-2]
    elif word[-3:] == 'ies':
      return word[:-2]
    elif word[-2:] == 'ss':
      return word
    elif word[-1] == 's':
      return word[:-1]
    elif word[-2:] == 'ly':
      return word[:-2]
    elif word[-3:] == 'ing':
      return word[:-3]
    elif word[-2:] == 'ed':
      return word[:-2]
  return word
