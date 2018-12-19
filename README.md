PyTorch LSTM neural networks that classify byte sequences as either UTF-8 or Windows-1252.

A training and an evaluation set was created from the News Crawl 2009 corpus (available http://www.statmt.org/wmt11/translation-task.html). This corpus is created from English-language newswire data.

Chardet (version 4.0.0), an existing Python character encoding detector, was used for comparison.

Select results:

32 hidden unit model accuracy: 0.9770

Chardet accuracy: 0.7340
