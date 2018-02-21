# Re-using these materials

The content of the notebook is released under the [Creative Commons Attributional Share-Alike license](https://creativecommons.org/licenses/by-sa/4.0/legalcode).  The code may be used indepdently of the rest of the notebook, and is licensed under the BSD 3-clause license.

# PythonForLinguistsTalk
Notebooks and sample data from my Python for Linguists talk at UTA (Feb 16th, 2018).

The .ipynb file is what the talk was given directly from, with a bit of touching up.  The .tex file was automatically generated from the .ipynb file using nbconvert, and compiled into the .pdf file also in this repository.

# Running this notebook

To run this notebook, you will need to install the following Python libraries:
* Jupyter (to run/view the notebook itself)
* Gensim
* spaCy
  * You'll need to download two of spaCy's language models: `en_core_web_sm` and `en_core_web_lg`.  Installation instructions are [here.](https://spacy.io/models/)
* Numpy
* Scikit-learn (goes by sklearn when installing)
* Matplotlib
* Natural Language Toolkit (NLTK)
* tqdm
* Pandas

Open Jupyter and navigate to this .ipynb file, then open it.  Every major section is designed to be able to run independently, minus the "Setup Code" section, which should always be run first.

For the first part of the talk, you'll also need to download `glen carrig.txt` from the Github repository and have it in the same folder as this notebook.

For the second part of the talk, you'll need to download the [Blog Authorship Corpus](http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm) and unzip its files into a folder named "blogs" (case-sensitive) in the same folder as this notebook.
