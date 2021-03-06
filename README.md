# Tweet for Me: RNN-based Keyword Oriented Post Generator
Final project for NTHU ISA562100 自然語言處理實作Natural Language Processing Lab.

## Introduction
隨著智慧型手機以及行動網路的普及，社群媒體已經漸漸成為了許多人們生活，交流的重心。它利 用人們想得到「認同」的心理，促使人們不斷的發文。藉由這個情況，我們思考如果能從大量的數 據中train出一個model，只需要給定一個關鍵字或是主題，他就可以自動產生適當的語句。不但能省 下人們大量的時間，還能確保語句內容一定程度上符合大眾的價值觀。
在這個project，我們從Twitter上爬取幾萬則貼文藉著hashtag分門別類，丟入我們的noisy channel mo del(其中包含statistic model和遞迴神經網路(RNN))進行訓練。我們就可以透過主題(關鍵字)的選擇， 產生一段與之有關的對話。最後再透過Flask包裝。

## Methodology

### Dataset

Emotion-English Dataset: https://github.com/huseinzol05/NLP-Dataset

### Model

1. Channel Model: Statistical model
2. Language Model: Recurrent Neural Network(RNN)

![rnn_structure](https://github.com/sellylu/TweetforMe/blob/master/image/rnn_struc.png "RNN Structure")

4. Keyword categorization: WordNet

![Result Demostration](https://github.com/sellylu/TweetforMe/blob/master/image/result.png "Result Demostration")

## Reference

1. Emotion-English Dataset: https://github.com/huseinzol05/NLP-Dataset
2. WordNet: https://wordnet.princeton.edu/
3. LSTM Network: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
4. The effectiveness of RNN: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
5. Twitter API: https://developer.twitter.com/en/docs/api-reference-index
6. Flask Framework: http://flask.pocoo.org/
7. Multi-layer RNN for character-level language models in Torch: https://github.com/karpathy/char-rnn
8. Mikolov, T., Karafiát, M., Burget, L., Černocký, J., &amp; Khudanpur, S. (2010). Recurrent neural network based language model. In Eleventh Annual Conference of the International Speech Communication Association.
9. Noisy Channel: Ch14, Segaran, Toby, and Jeff Hammerbacher. Beautiful data: the stories behind elegant data solutions. "O'Reilly Media, Inc.", 2009.

## Acknowledgement

Thanks to Prof. Jason S. Chang giving the course and TAs helping us go through all the labs. Thanks to my teammate [@chuangyc](https://github.com/chuangyc) work so hard to complete this project together withn very short time.
