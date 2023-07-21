#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import os
import pickle


# In[2]:


from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras.layers import Input,Dense,LSTM,Embedding,Dropout,add


# In[3]:


model = VGG16()
model = Model(inputs = model.inputs,outputs = model.layers[-2].output)
print(model.summary())


# In[4]:


from tqdm.notebook import tqdm


# In[5]:


BASE_DIR = '/Users/91887/imageClassification/archive (1)'
WORKING_DIR = '/Users/91887/imageClassification'


# In[ ]:


features = {}
directory = os.path.join(BASE_DIR,'Images')

for img_name in tqdm(os.listdir(directory)):
    #load the image from file 
    img_path = directory+'/'+img_name
    image = load_img(img_path,target_size = (224,224))
    #image pixel to numpy array 
    image = img_to_array(image)
    #reshapre data for model
    image =  image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
    #preprocess image for vgg
    image = preprocess_input(image)
    #Extract the Features
    feature = model.predict(image,verbose=0)
    #get Image ID
    image_id = img_name.split('.')[0]
    #store the Features
    features[image_id] = feature


# In[ ]:


# Store Features in pickle 

pickle.dump(features,open(os.path.join(WORKING_DIR,'features.pkl'),'wb'))


# In[6]:


# load Features from pickle 

with open(os.path.join(WORKING_DIR,'features.pkl'),'rb') as f:
    features = pickle.load(f)


# In[7]:


with open(os.path.join(BASE_DIR,'captions.txt'),'r') as f:
    next(f)
    captions_doc = f.read()


# In[8]:


# Create mapping of images to captions

# Creating Dict name mapping 

mapping = {}

#process lines 

for line in tqdm(captions_doc.split('\n')):
    #Split the line by comma
    tokens = line.split(',')
    if len(line)<2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    #remove exe from image_id
    image_id = image_id.split('.')[0]
    #convert caption list into the string 
    caption =" ".join(caption)
    #Create a list 
    if image_id not in mapping:
        mapping[image_id] = []
    mapping[image_id].append(caption)


# In[9]:


def load_descriptions(doc):
    # Create mapping of images to captions

    # Creating Dict name mapping 

    mapping = {}

    #process lines 

    for line in tqdm(captions_doc.split('\n')):
        #Split the line by comma
        tokens = line.split(',')
        if len(line)<2:
            continue
        image_id, caption = tokens[0], tokens[1:]
        #remove exe from image_id
        image_id = image_id.split('.')[0]
        #convert caption list into the string 
        caption =" ".join(caption)
        #Create a list 
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption)
    return mapping


# In[10]:


descriptions = load_descriptions(captions_doc)


# In[11]:


import string


# In[12]:


def clean(descriptions):
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for key,captions in descriptions.items():
        for i in range(len(captions)):
            caption = captions[i]
            #Preprocessing steps
            caption = caption.lower()
            caption = caption.replace('[^A-Za-z]', '')  #Delete special character and digit
            caption = caption.replace('\s+',' ')
            # add Start and End tags to the caption
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption


# In[13]:


clean(descriptions)


# In[14]:


# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
    all_captions = []
    for key in descriptions:
        for caption in descriptions[key]:
            all_captions.append(caption)
    return all_captions


# In[15]:


vocabulary = to_vocabulary(descriptions)
print('Vocabulary Size: %d' % len(vocabulary))


# In[19]:


vocabulary[:10]


# In[16]:


# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
        lines = list()
        for key, desc_list in descriptions.items():
            for desc in desc_list:
                lines.append(key + ' ' + desc)
        data = '\n'.join(lines)
        file = open(filename, 'w')
        file.write(data)
        file.close()


# In[17]:


save_descriptions(descriptions, 'descriptions.txt')


# In[18]:


image_ids = list(descriptions.keys())
split = int(len(image_ids)*0.80)
split


# In[19]:


train = image_ids[:split]
test = image_ids[split:]


# In[20]:


from pickle import load


# In[21]:


# load doc into memory
def load_doc(filename):
        # open the file as read only
        file = open(filename, 'r')
        # read all text
        text = file.read()
        # close the file
        file.close()
        return text


# In[22]:


# load a pre-defined list of photo identifiers
def load_set(filename):
        doc = load_doc(filename)
        dataset = list()
        # process line by line
        for line in doc.split('\n'):
            # skip empty lines
            if len(line) < 1:
                continue
            # get the image identifier
            identifier = line.split('.')[0]
            dataset.append(identifier)
        return set(dataset)


# In[23]:


# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
        # load document
        doc = load_doc(filename)
        descriptions = dict()
        for line in doc.split('\n'):
            # split line by white space
            tokens = line.split()
            # split id from description
            image_id, image_desc = tokens[0], tokens[1:]
            # skip images not in the set
            if image_id in dataset:
                # create list
                if image_id not in descriptions:
                    descriptions[image_id] = list()
                # wrap description in tokens
                desc = ' '.join(image_desc)
                # store
                descriptions[image_id].append(desc)
        return descriptions


# In[24]:


# load photo features
def load_photo_features(filename, dataset):
        # load all features
        all_features = load(open(filename, 'rb'))
        # filter features
        features = {k: all_features[k] for k in dataset}
        return features


# In[25]:


print('Dataset: %d' % len(train))


# In[26]:


# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)


# In[27]:


print('Descriptions: train=%d' % len(train_descriptions))


# In[28]:


train_features = load_photo_features('features.pkl', train)
print('Photos: train=%d' % len(train_features))


# In[29]:


train_descriptions


# In[30]:


# load doc into memory
def load_doc(filename):
        # open the file as read only
        file = open(filename, 'r')
        # read all text
        text = file.read()
        # close the file
        file.close()
        return text


# In[31]:


# load a pre-defined list of photo identifiers
def load_set(filename):
        doc = load_doc(filename)
        dataset = list()
        # process line by line
        for line in doc.split('\n'):
            # skip empty lines
            if len(line) < 1:
                continue
            # get the image identifier
            identifier = line.split('.')[0]
            dataset.append(identifier)
        return set(dataset)


# In[32]:


# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
        # load document
        doc = load_doc(filename)
        descriptions = dict()
        for line in doc.split('\n'):
            # split line by white space
            tokens = line.split()
            # split id from description
            image_id, image_desc = tokens[0], tokens[1:]
            # skip images not in the set
            if image_id in dataset:
                # create list
                if image_id not in descriptions:
                    descriptions[image_id] = list()
                # wrap description in tokens
                desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
                # store
                descriptions[image_id].append(desc)
        return descriptions


# In[33]:


# load photo features
def load_photo_features(filename, dataset):
        # load all features
        all_features = load(open(filename, 'rb'))
        # filter features
        features = {k: all_features[k] for k in dataset}
        return features


# In[34]:


# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
        all_desc = list()
        for key in descriptions.keys():
            [all_desc.append(d) for d in descriptions[key]]
        return all_desc


# In[35]:


# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
        lines = to_lines(descriptions)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines)
        return tokenizer


# In[36]:


tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)


# In[37]:


def max_length(descriptions):
        lines = to_lines(descriptions)
        return max(len(d.split()) for d in lines)


# In[38]:


def create_sequences(tokenizer, max_length, desc_list, photo):
        X1, X2, y = list(), list(), list()
        # walk through each description for the image
        for desc in desc_list:
            # encode the sequence
            seq = tokenizer.texts_to_sequences([desc])[0]
            # split one sequence into multiple X,y pairs
            for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                # encode output sequence
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                # store
                X1.append(photo)
                X2.append(in_seq)
                y.append(out_seq)
        return np.array(X1), np.array(X2), np.array(y)


# In[39]:


def define_model(vocab_size, max_length):
        # feature extractor model
        inputs1 = Input(shape=(4096,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)
        # sequence model
        inputs2 = Input(shape=(max_length,))
        se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)
        # decoder model
        decoder1 = add([fe2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(vocab_size, activation='softmax')(decoder2)
        # tie it together [image, seq] [word]
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        # summarize model
        print(model.summary())
        return model


# In[40]:


def data_generator(descriptions, photos, tokenizer, max_length):
        # loop for ever over images
        while 1:
            for key, desc_list in descriptions.items():
                # retrieve the photo feature
                photo = photos[key][0]
                in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo)
                yield [[in_img, in_seq], out_word]


# In[41]:


tokenizer = create_tokenizer(train_descriptions)


# In[42]:


vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)


# In[43]:


# train the model
model = define_model(vocab_size, max_length)
# train the model, run epochs manually and save after each epoch
epochs = 10
steps = len(train_descriptions)
for i in range(epochs):
        # create the data generator
        generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
        # fit for one epoch
        model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
        model.save('model_' + str(i) + '.h5')


# In[44]:


print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = load_photo_features('features.pkl', test)
print('Photos: test=%d' % len(test_features))


# In[46]:


from tensorflow.keras.models import load_model


# In[47]:


filename = 'model_9.h5'
model = load_model(filename)


# In[59]:


# map an integer to a word
def word_for_id(integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None


# In[77]:


# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
        # seed the generation process
        in_text = 'startseq'
        # iterate over the whole length of the sequence
        for i in range(max_length):
            # integer encode input sequence
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            # pad input
            sequence = pad_sequences([sequence], maxlen=max_length)
            # predict next word
            yhat = model.predict([photo,sequence], verbose=0)
            # convert probability to integer
            yhat = np.argmax(yhat)
            # map integer to word
            word = word_for_id(yhat, tokenizer)
            # stop if we cannot map the word
            if word is None:
                break
            # append as input for generating the next word
            in_text += ' ' + word
            # stop if we predict the end of the sequence
            if word == 'endseq':
                break
        return in_text


# In[61]:


def evaluate_model(model, descriptions, photos, tokenizer, max_length):
        actual, predicted = list(), list()
        # step over the whole set
        for key, desc_list in descriptions.items():
            # generate description
            yhat = generate_desc(model, tokenizer, photos[key], max_length)
            # store actual and predicted
            references = [d.split() for d in desc_list]
            actual.append(references)
            predicted.append(yhat.split())
        # calculate BLEU score
        print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
        print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
        print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
        print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


# In[62]:


import numpy as np


# In[64]:


from nltk.translate.bleu_score import corpus_bleu


# In[66]:


evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)


# In[79]:


def extract_features(filename):
        model = VGG16()
        model = Model(inputs = model.inputs,outputs = model.layers[-2].output)
        # load the photo
        image = load_img(filename, target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # get features
        feature = model.predict(image, verbose=0)
        return feature
    
# load the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(vocabulary)
# pre-define the max sequence length (from training)
max_length = 35
# load the model
model = load_model('model_9.h5')
# load and prepare the photograph
photo = extract_features('sample.jpg')
# generate description
description = generate_desc(model, tokenizer, photo, max_length)
print(description)


# In[81]:


# Remove the startseq and endseq
query = description
stopwords = ['startseq','endseq']
querywords = query.split()

resultwords  = [word for word in querywords if word.lower() not in stopwords]
result = ' '.join(resultwords)

print(result)


# In[83]:


photo=extract_features('sample2.jpg')
description = generate_desc(model, tokenizer, photo, max_length)
print(description)


# In[84]:


# Remove the startseq and endseq
query = description
stopwords = ['startseq','endseq']
querywords = query.split()

resultwords  = [word for word in querywords if word.lower() not in stopwords]
result = ' '.join(resultwords)

print(result)


# In[85]:


def predict_captions(image_name):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(vocabulary)
    # pre-define the max sequence length (from training)
    max_length = 35
    # load the model
    model = load_model('model_9.h5')
    # load and prepare the photograph
    photo = extract_features(image_name)
    # generate description
    description = generate_desc(model, tokenizer, photo, max_length)
    query = description
    stopwords = ['startseq','endseq']
    querywords = query.split()
    resultwords  = [word for word in querywords if word.lower() not in stopwords]
    result = ' '.join(resultwords)
    return result


# In[86]:





# In[ ]:




