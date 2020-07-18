# Introduction
Alzheimer’s is a widespread, irreversible, progressive neurodegenerative disease, with a complex genetic architecture. The key goal of this project is to seek out disease risk genes and classify them as Alzheimer's Disease associated and unassociated.

Various machine learning algorithms have been used to predict candidate genes. Previous prediction methods can be roughly divided into five types-

1. Methods studying protein-protein interaction networks

2. Gene functional annotations

3. Sequence-based features patterns

4. Machine learning and network topological features

5. Information about tissue-specific networks

These methods predict associated genes or biomarkers. However, there are few reports on brain gene expression data. Accordingly, the <a href = "https://bmcneurol.biomedcentral.com/articles/10.1186/s12883-017-1010-3">research paper</a> by Huang et al. on *Revealing Alzheimer’s disease genes spectrum in the whole-genome by machine learning* was used as a reference for this project.

<img src = "https://github.com/isha-git/Alzheimers-Disease/blob/master/Images/ResearchPaper.PNG" width = "800">

The aim is to divide the genes into five classes, namely C1-AD: probable pathogenic genes, C2-AD: high confidence genes, C3-AD: related genes, and C4-AD: possibly associated genes.

# Libraries
1. Numpy

2. Scipy <br>
3. Sklearn <br>
4. Pandas <br>
5. Pylab <br>
6. Matplotlib <br>
7. Itertools <br>

Environment- Python 3.6, Windows 10

# Dataset
The dataset used in the above-mentioned research paper was taken from the <a href = "http://www.alzgene.org/"> AlzGene archive </a>. The training features include number of positive and negative Alzheimer's cases in control studies and family-based studies for 335 genes.

<img src = "https://github.com/isha-git/Alzheimers-Disease/blob/master/Images/DatasetScreenshot.PNG" width = "500">

The lack of sufficient data samples make it difficult to train the model. Accordingly, regularization has been used to prevent overfitting. <br>
For training purposes, 33% of the data was used for testing.

# Results
The followed algorithms were trained on the given dataset-

1. Support Vector Machine using Radial Kernel

2. Support Vector Machine using Linear Kernel

3. Support Vector Machine using Polynomial Kernel

4. Support Vector Machine using Sigmoid kernel

5. Decision Trees

<img src = "https://github.com/isha-git/Alzheimers-Disease/blob/master/Images/Results.PNG" width = "800">

**Of these, desicion trees gave the best accuracy (88.29%). <br>
However, the highest Receiver Operating Characteristic (ROC) curve area of 0.78 was obtained using Support Vector Machine with Radial kernel.**

Note- The results on Support Vector Machine using R library were provided in the paper and were not reproduced by us.

# Team Members
The project was developed with <a href = "https://github.com/jagriti04">Jagriti</a> and <a href="https://github.com/shripriyamaheshwari">Shripriya Maheshwari</a> for the Machine Learning course (CS314b) at the Indian Instiute of Information Technology, Design and Manufacturing, Jabalpur, India.
