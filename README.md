# nlp-AuthorProbability
Generate likeliest text and calculate probability of belonging to an Author using Ngram Language Model.

In this project we created a simple Language model that assigns log probabilities to sequences of words, the ngrams.
The LM can predict sequences of words similarity to a given corpus. The LM can then generate likeliest text based on the training data. 
In practice, we don't use raw probability as our metric for evaluating language models, instead we use the variant called perplexity. The perplexity of a LM on a test set is an inverse probability of the test set normalized by the number of words.
We found that if the n is greater than 4 then the generation results are more coherent. 

