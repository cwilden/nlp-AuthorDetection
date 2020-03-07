import math
import random

random.seed(1)

#Input: n is the size of n grams
# text is list of word/strings
def get_ngrams(n, text):
    #tokenize the text
    textList = []
    for line in text.split():
        textList.append(line)

    #insert start token
    textList.insert(0, '<s>')
    startcount = 1
    if(n > 2):
        #ngrams = 3
        while(startcount != n - 1):
            textList.insert(0, '<s>')
            startcount += 1

    #insert stop token
    textList.append('</s>')
    stopcount = 1
    while(stopcount != n - 1):
        textList.append('</s>')
        stopcount += 1

    #create word: context tuple
    for i, word in enumerate(textList):
        if(i >= n - 1):
        # should be n - 1 size
            prefix = ()
            size = 1
            while(size != n):
                prefix = prefix + (textList[i - size],)
                size += 1
            yield (word, prefix)

#NGramLM class
class NGramLM:

    #initialize variables
    def __init__(self, n):
        self.n = n
        self.ngram_counts = dict()
        self.context_counts = dict()
        self.vocabulary = []

    #update class function
    def update(self, text):
        #generate ngrams from text
        self.training_data = get_ngrams(self.n, text)

        #populate vocabulary and dictionaries
        self.vocabulary.append('<s>')
        for line in self.training_data:
            (w, cont) = line
            if(line not in self.ngram_counts.keys()):
                self.ngram_counts[line] = 1
            else:
                self.ngram_counts[line] += 1
            if(cont not in self.context_counts.keys()):
                self.context_counts[cont] = 1
            else:
                self.context_counts[cont] += 1
            if w not in self.vocabulary:
                self.vocabulary.append(w)

    #return word probability given context
    def word_prob(self, word, context, delta):
        #check for out of vocabulary word
        if (word not in self.vocabulary):
            word = '<unk>'

        #check for out of vocabulary words in context
        updatedContext = []
        for v in context:
            if (v not in self.vocabulary):
                updatedContext.append('<unk>')
            else:
                updatedContext.append(v)
        upCont = tuple(context)

        #get word count
        wordcount = self.ngram_counts.get((word, upCont))
        if (wordcount == None):
            wordcount = 0

        #get context count
        contextcount = self.context_counts.get(upCont)

        #if context not seen
        if (upCont not in self.context_counts.keys()):
            rv = 1 / len(self.vocabulary)
            return math.log(rv)
        else:
            #return (wc + delta) / (contextcount + delta*|V|)
            rv = (wordcount + delta) / (contextcount + delta*len(self.vocabulary))
            if (rv != 0.0):
                return math.log(rv)
            else:
                return 0

    # generate random word
    def random_word(self, context, delta):
        # sort alphabetically V
        sortedV = sorted(self.vocabulary)

        # obtain w | context probabilities
        probabilities = []
        word = ''
        uniqueProbs = []
        for i, word in enumerate(sortedV):
            prob = self.word_prob(word, context, delta)
            probabilities.insert(i, prob)
            if prob not in uniqueProbs:
                uniqueProbs.append(prob)

        # normalize probability distribution
        normprobs = [float(i) / sum(uniqueProbs) for i in uniqueProbs]

        # dictionary of probability : [list of words]
        zones = dict()
        for j, p in enumerate(sorted(uniqueProbs)):
            words = []
            for i, w in enumerate(sortedV):
                if (probabilities[i] == p):
                    words.append(w)
            zones[normprobs[j]] = words

        # sort the probabilities
        sortedZones = sorted(zones.keys())

        # generate random r in probability distribution
        rnum = random.uniform(0, sortedZones[-1])

        # randomly sample word in that range where r is
        maxprob = 0
        returnWord = ''
        ind = 0
        for prob, words in zones.items():
            if (prob > rnum):
                maxprob = prob
                key = list(zones.keys())[ind]
                returnWord = random.sample(zones.get(key), 1)
                break
            ind += 1
        return returnWord[0]

    # return word with highest probability
    def likeliest_word(self, context, delta):
        # sort alphabetically V
        sortedV = sorted(self.vocabulary)

        # obtain probabilities of w | context
        probabilities = []
        uniqueProbs = []
        zones = dict()
        for i, word in enumerate(sortedV):
            prob = self.word_prob(word, context, delta=0)
            probabilities.insert(i, prob)
            if prob not in uniqueProbs:
                uniqueProbs.append(prob)

        # get max probability
        highestProb = max(uniqueProbs)

        # find words with highest probability
        for j, p in enumerate(sorted(uniqueProbs)):
            words = []
            for i, w in enumerate(sortedV):
                if (probabilities[i] == p):
                    words.append(w)
            zones[p] = words
        possiblewords = zones.get(highestProb)

        # pick a word of the highest probability and return
        index = random.randint(0, len(possiblewords) - 1)
        if (index != len(possiblewords) - 1):
            index += 1
        rvword = possiblewords[index]
        return rvword


#Input: corpus with rare words
#Output: corpus with <unk> tokens
def mask_rare(corpus):

    #get counts of each word
    wordCountDict = dict()
    for word in corpus.split():
        if word in wordCountDict:
            wordCountDict[word] += 1
        else:
            wordCountDict[word] = 1

    #find rare words and add to a list
    unkWords = []
    for key, val in wordCountDict.items():
        if(val == 1):
            unkWords.append(key)

    #generate new corpus with <unk> at locations of rare words
    rCorpus = ''
    for word in corpus.split():
        if word in unkWords:
            rCorpus += '<unk>' + ' '
        else:
            rCorpus += word + ' '
    return rCorpus


#Create NgramLM model instance
def create_ngramlm(n, corpus_path):
    #initialize model
    lm = NGramLM(n)

    #mask rare text
    file = open(corpus_path, "r")
    text = ''
    for line in file:
        text += line
    unkText = mask_rare(text)

    #update model
    lm.update(unkText)
    return lm


#Generate probability for text sentence
def text_prob(model, text):
    probabilities = 0
    #generate ngrams
    ngramsobj = get_ngrams(model.n, text)
    for word in ngramsobj:
        #obtain word probability
        prob = model.word_prob(word=word[0], context=word[1], delta=0.5)
        probabilities += prob
    return probabilities


#Generate random text
def random_text(model, max_length, delta=0):
    ngram = model.n
    tokens = []
    #initialize with <s> tokens
    for i in range(ngram - 1):
        tokens.append('<s>')

    #while ith generated word < max_length
    for gen_word in range(max_length):
        #get previous context
        cont = []
        index = len(tokens) - 1
        for i in range(ngram - 1):
            cont.append(tokens[index])
            index -= 1
        #get random word and append to token list
        word = model.random_word(tuple(cont), delta=0.5)
        tokens.append(word)
        if(word == '</s>'):
            break

    #convert tokens to string
    rvString = ''
    for w in tokens:
        rvString += w + ' '
    return rvString


#Generate likeliest text
def likeliest_text(model, max_length, delta):
    ngram = model.n
    tokens = []
    #initialize with <s> tokens
    for i in range(ngram - 1):
        tokens.append('<s>')
    #for word in generated sentence
    for gen_word in range(max_length):
        cont = []
        index = len(tokens) - 1
        for i in range(ngram - 1):
            cont.append(tokens[index])
            index -= 1
        #get likeliest word and append to tokens
        word = model.likeliest_word(tuple(cont), delta=0.5)
        tokens.append(word)
        if (word == '</s>'):
            break

    #convert tokens to string
    rvString = ''
    for w in tokens:
        rvString += w + ' '
    return rvString

#Calculate perplexity
def perplexity(model, corpus_path):
    #open file
    testfile = open(corpus_path, "r")
    text = ''
    tokenCount = 0
    all_probs = 0
    #calculate text probability and add to total probability
    for line in testfile:
        text += line
        lineprob = text_prob(model, line)
        all_probs += lineprob

    #get count of words in test
    for line in text.split():
        tokenCount+=1

    #calculate l
    l = (1/tokenCount)*all_probs

    #calculate perplexity
    perplex = 2**-l
    return perplex

#Create Warpeace LM
LM = create_ngramlm(3, "warpeace.txt")

#First and Second Sentences
example1 = 'God has given it to me, let him who touches it beware!'
logp = text_prob(LM, example1)
example2 = 'Where is the prince, my Dauphin?'
logp2 = text_prob(LM, example2)
print(logp)
print(logp2)

#Perplexity
LM1 = create_ngramlm(3, "shakespeare.txt")
p = perplexity(LM1, "sonnets.txt")
print(p)

LM2 = create_ngramlm(3, "warpeace.txt")
p2 = perplexity(LM2, "sonnets.txt")
print(p2)

#Generate Random five sentences
rparagraph = ''
for i in range(5):
    text = random_text(LM, 10, delta=0)
    rparagraph += text + '\n'
print(rparagraph)

#Likeliest sentence
print("bigram")
biLM = create_ngramlm(2, "shakespeare.txt")
text = likeliest_text(biLM, 10, delta=0)
print(text)

print("trigram")
triLM = create_ngramlm(3, "shakespeare.txt")
tritext = likeliest_text(triLM, 10, delta=0)
print(tritext)

print("4gram")
fLM = create_ngramlm(4, "shakespeare.txt")
ftext = likeliest_text(fLM, 10, delta=0)
print(ftext)

print("5gram")
fvLM = create_ngramlm(5, "shakespeare.txt")
fvtext = likeliest_text(fvLM, 10, delta=0)
print(fvtext)
