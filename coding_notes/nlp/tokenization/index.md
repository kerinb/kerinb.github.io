---
layout: default
title: Tokenization with Python
---
<br>

<h1>
	Text Tokenization in Python
</h1>

<h3> 
	<p style="width:100%;max-width:600px;">
		In the area of Natural Language Processing and Text Processing in general, Tokenization
		is the process of taking a corpus of text data and splitting it out into its 
		constituent parts - i.e. tokens. These tokens can be thought of as the components of a 
		sentence, e.g. words. Or on a higher level, the components of the corpus, think more
		along the lines of sentences or paragraphs. 
		
		In the following examples, we will use the python Natural Language Toolkit (NLTK)[1]. <!-- REFERENCE: https://www.nltk.org/_modules/nltk/tokenize.html -->
	</p>
</h3>

<h2>
	Word Level Tokenization:
</h2>

<h3> 
	<p style="width:100%;max-width:600px;">
		<b>Illustration #1.1: word_tokenize</b> - This method splits a corpus into its individual words preserving punctuation.<br>
		<!-- In jupyter, create a simple example - import data etc, split and show split data. -->
		<br> The word_tokenize function used in the above snippet splits the input data into
		its contituent words and punctuation, i.e. commas and other punctuation marks and 
		special characters in the data will be included as individual tokens. This method is a wrapper 
		for that in turn calls the TreebankWordTokenizer class - The TreebankWordTokenizer is discussed more below.
	</p>
	
	<br>
	
	<p style="width:100%;max-width:600px;">
		<b>Illustration #1.2: PunktWordTokenizer</b> - Split a corpus into its individual words while splitting the punctuation marks from words also.
		<!-- In jupyter, create a simple example - import data etc, split and show split data. -->
	</p>
	
	<br>
	
	<p style="width:100%;max-width:600px;">
		<b>Illustration #1.3: wordpunct_tokenize</b> - EXPLANATION -  It seperates the punctuation from the words.
		<!-- In jupyter, create a simple example - import data etc, split and show split data. -->
		EXPLANATION. 
	</p>
	
	<br>
	
	<p style="width:100%;max-width:600px;">
		<b>Illustration #1.4: TreebankWordTokenizer </b> - EXPLANATION - These tokenizers work by separating the words using punctuation and spaces. 
		And as mentioned in the code outputs above, it does not discard the punctuation, allowing a user to decide what to do with the punctuations at the time of pre-processing.
		<!-- In jupyter, create a simple example - import data etc, split and show split data. -->
		EXPLANATION. 
	</p>
		
	<br>
	
	<p style="width:100%;max-width:600px;">
		<b>Illustration #1.4: RegexpTokenizer  </b> - EXPLANATION 
		<!-- In jupyter, create a simple example - import data etc, split and show split data. -->
		EXPLANATION. 
	</p>
	
	<br>
	
	<p style="width:100%;max-width:600px;">
		<b>Illustration #4.1: TweetTokenizer</b> - 
		<br>
		<!-- In jupyter, create a simple example - import data etc, split and show split data. -->
		<br>
		explanation
	</p>
</h3> 

<h2>
	Sentence Level Tokenization:
</h2>

<h3> 
	<p style="width:100%;max-width:600px;">
		<b>Illustration #2.1: sent_tokenize</b> - The sent_tokenize method uses the PunktSentenceTokenizer 
		from the nltk.tokenize.punkt module. 
		<br>
		<!-- In jupyter, create a simple example - import data etc, split and show split data. -->
		<br>
		explanation
	</p>
	
	<br>
	
	<p style="width:100%;max-width:600px;">
		<b>Illustration #2.2: None English Sentence Tokenization</b> - As I have illuded to above, it 
		is possible to tokenize a sentence using the sent_tokenize method. To do so, we simply add the
		langauge as a paramenter to the method. This can be seen below.
		<br>
		<!-- In jupyter, create a simple example - import data etc, split and show split data. -->
		<br>
		explanation
		
		The second way to use a non=English sentence tokenizer is to explicitly instantiate the language
		tokenizer we want to use. This can be shown below where we load the tokenizer we need from the nltk.data module.
		<br>
		<!-- In jupyter, create a simple example - import data etc, split and show split data. -->
		<br>
		explanation
	</p>
</h3> 

<h2>
	Whitespace Tokenizer VS str.split(' '):
</h2>

<h3> 
	<p style="width:100%;max-width:600px;">
		<b>Illustration #3.1: WhitespaceTokenizer</b> - 
		<br>
		<!-- In jupyter, create a simple example - import data etc, split and show split data. -->
		<br>
		explanation
	</p>
	
	<br>
	
	<p style="width:100%;max-width:600px;">
		<b>Illustration #2.2: str.split(' ')</b> - 
		<br>
		<!-- In jupyter, create a simple example - import data etc, split and show split data. -->
		<br>
		explanation
	</p>
</h3> 

<h2>
	Multi-Word Epxression Tokenization:
</h2>

<h3> 
	<p style="width:100%;max-width:600px;">
		<b>Illustration #5.1: MWETokenizer </b> - 
		<br>
		<!-- In jupyter, create a simple example - import data etc, split and show split data. -->
		<br>
		explanation
	</p>
</h3>