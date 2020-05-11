import random
import sys
import os
import re
import markovify
import pronouncing
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import LSTM


# %%

# Count the word frequency
def show_word_frequency(poems):
    words = poems.replace('\n', ' ').split(' ')
    word_freq = dict()
    for word in words:
        if word_freq.get(word) is None:
            word_freq[word] = 1
        else:
            word_freq[word] += 1
    word_freq = dict(sorted(word_freq.items(), key=lambda f: f[1], reverse=True))

    # Plot the word frequency
    plot_size = 50
    plt.figure(figsize=(plot_size, plot_size))
    plt.bar(list(word_freq.keys())[:plot_size], list(word_freq.values())[:plot_size], color='red')
    plt.show()


# show_word_frequency(poems)


# %%

# Create the LSTM Model
def create_LSTM(layers_num):
    # Initialized the model sequence
    model = Sequential()
    model.add(LSTM(4, input_shape=(2, 2), return_sequences=True))
    for i in range(layers_num):
        model.add(LSTM(8, return_sequences=True))
    model.add(LSTM(2, return_sequences=True))
    model.summary()

    # Compile the model
    model.compile(optimizer="rmsprop", loss='mse')
    return model


# %%

# Create the Markov Model
def create_Markov(poems):
    text_model = markovify.NewlineText(poems)
    return text_model


# %%

def syllables(line):
    count = 0
    words = line.split(" ")
    vowels = 'aeiouAEIOU'

    # Check every word in the line
    for word in words:
        if word[0] in vowels:
            count += 1
        for i in range(1, len(word)):
            if word[i] in vowels and word[i - 1] not in vowels:
                count += 1

        # For special case
        if word[len(word) - 1] == 'e':
            count -= 1
        elif word[len(word) - 1] == 'le':
            count += 1

        # If no vowel in a word
        if count == 0:
            count += 1

    return (count / max_syl_num)


# %%

# Define the rhyme end
def rhyme(line, rhyme_list):
    word = re.sub(r"\W+", '', line.split(' ')[-1]).lower()
    rhymes = pronouncing.rhymes(word)
    rhyme_list_ends = []

    for r in rhymes:
        rhyme_list_ends.append(r[-2:])
    try:
        rhymes_scheme = max(set(rhyme_list_ends), key=rhyme_list_ends.count)
    except Exception:
        rhymes_scheme = word[-2:]
    try:
        float_rhyme = rhyme_list.index(rhymes_scheme)
        float_rhyme = float_rhyme / float(len(rhyme_list))
        return float_rhyme
    except Exception:
        float_rhyme = None
        return float_rhyme


# %%

# Get the rhyme index
def rhyme_index(lyrics):
    rhyme_master_list = []
    print("Building rhymes list:")

    for lyric in lyrics:

        rhyme_list_ends = []

        words = lyric.split(' ')
        word = re.sub(r"\W+", '', words[-1].lower())
        rhymes = pronouncing.rhymes(word)

        for r in rhymes:
            rhyme_list_ends.append(r[-2:])
        try:
            rhyme_scheme = max(set(rhyme_list_ends), key=rhyme_list_ends.count)
        except Exception:
            rhyme_scheme = word[-2:]
        rhyme_master_list.append(rhyme_scheme)

    rhyme_master_list = list(set(rhyme_master_list))
    reverse_list = [x[::-1] for x in rhyme_master_list]
    reverse_list = sorted(reverse_list)
    rhyme_list = [x[::-1] for x in reverse_list]

    print("List of Sorted 2-Letter Rhyme Ends:")
    print(rhyme_list)

    return rhyme_list


# %%

# Split the poem into words
def split_lyrics(poem):
    poem = poem.split('\n')
    return poem


# %%

# Vectorised the context
def build_Xy(lyrics, rhyme_list):
    dataset = []
    line_list = []
    for line in lyrics:
        line_list = [line, syllables(line), rhyme(line, rhyme_list)]
        dataset.append(line_list)
    x_data = []
    y_data = []
    for i in range(len(dataset) - 3):
        line1 = dataset[i][1:]
        line2 = dataset[i + 1][1:]
        line3 = dataset[i + 2][1:]
        line4 = dataset[i + 3][1:]
        x = [line1[0], line1[1], line2[0], line2[1]]
        x = np.array(x)
        x = x.reshape(2, 2)
        x_data.append(x)
        y = [line3[0], line3[1], line4[0], line4[1]]
        y = np.array(y)
        y = y.reshape(2, 2)
        y_data.append(y)
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    return x_data, y_data


# %%

# Train the model
def train_model(poems, depth):
    model = create_LSTM(depth)
    lyrics = split_lyrics(poems)
    rhyme_list = rhyme_index(lyrics)
    X_data, y_data = build_Xy(lyrics, rhyme_list)
    model.fit(x=np.array(X_data),
              y=np.array(y_data),
              batch_size=2,
              epochs=5,
              verbose=1)
    model.save_weights("trained.lstm")
    return model


# %%

# Generate lyrics
def generate_lyrics(markov_model, poems):
    lyrics = []
    last_words = []
    lyric_length = len(split_lyrics(poems))
    count = 0

    while len(lyrics) < lyric_length / 9 and count < lyric_length * 2:
        lyric = markov_model.make_sentence(max_overlap_ratio=.49, tries=100)
        if type(lyric) != type(None) and syllables(lyric) < 1:
            def get_last_word(bar):
                last_word = bar.split(" ")[-1]
                if last_word[-1] in "!.?,":
                    last_word = last_word[:-1]
                return last_word

            last_word = get_last_word(lyric)
            if lyric not in lyrics and last_words.count(last_word) < 3:
                lyrics.append(lyric)
                last_words.append(last_word)
                count += 1
    return lyrics


# %%

# Compose the poem
def compose_poem(rhyme_list, poems, model):
    poem_vectors = []
    human_lyrics = split_lyrics(poems)
    initial_index = random.choice(range(len(human_lyrics) - 1))
    initial_lines = human_lyrics[initial_index:initial_index + 2]
    starting_input = []
    for line in initial_lines:
        starting_input.append([syllables(line), rhyme(line, rhyme_list)])
    starting_vectors = model.predict(np.array([starting_input]).flatten().reshape(1, 2, 2))
    poem_vectors.append(starting_vectors)
    for i in range(3):
        poem_vectors.append(model.predict(np.array([poem_vectors[-1]]).flatten().reshape(1, 2, 2)))
    return poem_vectors


# %%

# Return the vectorised context in to pom
def vectors_into_poem(vectors, generated_lyrics, rhyme_list):
    print("\n\n")
    print("Writing verse:")
    print("\n\n")

    def last_word_compare(poem, line2):
        penalty = 0
        for line1 in poem:
            word1 = line1.split(" ")[-1]
            word2 = line2.split(" ")[-1]
            while word1[-1] in "?!,. ":
                word1 = word1[:-1]
            while word2[-1] in "?!,. ":
                word2 = word2[:-1]
            if word1 == word2:
                penalty += 0.2
        return penalty

    def calculate_score(vector_half, syllables, rhyme, penalty):
        desired_syllables = vector_half[0]
        desired_rhyme = vector_half[1]
        desired_syllables = desired_syllables * max_syl_num
        desired_rhyme = desired_rhyme * len(rhyme_list)
        score = 1.0 - abs(float(desired_syllables) - float(syllables)) + abs(
            float(desired_rhyme) - float(rhyme)) - penalty
        return score

    dataset = []
    for line in generated_lyrics:
        line_list = [line, syllables(line), rhyme(line, rhyme_list)]
        dataset.append(line_list)
    poem = []
    vector_halves = []
    for vector in vectors:
        vector_halves.append(list(vector[0][0]))
        vector_halves.append(list(vector[0][1]))
    for vector in vector_halves:
        score_list = []
        for item in dataset:
            line = item[0]
            if len(poem) != 0:
                penalty = last_word_compare(poem, line)
            else:
                penalty = 0
            total_score = calculate_score(vector, item[1], item[2], penalty)
            score_entry = [line, total_score]
            score_list.append(score_entry)
        fixed_score_list = [0]
        for score in score_list:
            fixed_score_list.append(float(score[1]))
        max_score = max(fixed_score_list)
        for item in score_list:
            if item[1] == max_score:
                poem.append(item[0])
                print(str(item[0]))
                for i in dataset:
                    if item[0] == i[0]:
                        dataset.remove(i)
                        break
                break
    return poem


# %%

# Print the poem
def print_poem(poems, trained_model):
    markov_model = markovify.NewlineText(poems)
    lyrics = generate_lyrics(markov_model, poems)
    rhyme_list = rhyme_index(lyrics)
    vectors = compose_poem(rhyme_list, poems, trained_model)
    poem = vectors_into_poem(vectors, lyrics, rhyme_list)

    f = open("poem_output.txt", "w", encoding="utf-8")
    for line in poem:
        f.write(line)
        f.write("\n")


# %%

# Initialized the settings
depth = 5
max_syl_num = 8
poems = open("poems.txt", "r").read()

if "trained.lstm" not in os.listdir('.'):
    print("Start to train the model.")
    model = train_model(poems, depth)

else:
    print("The trained model is loaded.")
    model = create_LSTM(depth)
    model.load_weights("trained.lstm")

print_poem(poems, model)
