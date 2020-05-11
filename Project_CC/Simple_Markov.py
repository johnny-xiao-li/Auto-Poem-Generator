import random
import sys
from nltk.translate.bleu_score import corpus_bleu

# Open the sample poems
poems = open("poems.txt", "r").read()

# Remove the double return between each poem
poems = ''.join([i for i in poems if not i.isdigit()]).replace("\n\n", " ").split(' ')
# print (poems)

index = 1
chain = {}
count = 20  # Desired word count of output

# This loop creates a dictionary called "chain". Each key is a word, and the value of each key is an array of the
# words that immediately followed it.
for word in poems[index:]:
    key = poems[index - 1]
    if key in chain:
        chain[key].append(word)
    else:
        chain[key] = [word]
    index += 1

length = 0
for word in poems:
    length += len(chain[word])

word1 = random.choice(list(chain.keys()))  # random first word
message = word1.capitalize()

# Picks the next word over and over until word count achieved
while len(message.split(' ')) < count:
    word2 = random.choice(chain[word1])
    word1 = word2
    message += ' ' + word2

# creates new file with output and prints it to the terminal
with open("output.txt", "w") as file:
    file.write(message)
output = open("output.txt", "r")

