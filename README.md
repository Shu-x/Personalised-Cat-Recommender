# Personalised Cat Recommender
<p align="center">
<img width="800" alt="image" src="https://github.com/Shu-x/Personalised-Cat-Recommender/assets/100437979/c2a986a7-783c-4481-890d-b82ae8e6eba9"><br>
The author's foster cats.</a>
</p>

## About
A multi-modal, personalised recommender system that helps potential adopters (a.k.a users) find their ideal cat. 

We explore and evaluate various models that incorporate the following
1. User interactions (such as likes, clicks, and dwell time)
2. Cat images
3. Text description of the cat's traits and adoption writeup

While text and image models have traditionally been confined in their own domains (i.e. using an image model to recommend similar-looking items, or a text model to recommend products with similar descriptions), we experiment with using **both text AND image features** to see if the combination can produce better recommendations.

## Pre-requisites
Python version==3.9

Install requirements via `pip install -r requirements.txt`

## Project Directory
```
.
├── data                                    <- input data
│   ├── auxiliary                           <- cat profiles
│   ├── train_val_test                      <- user-item interactions
├── models                                  <- trained and serialised models
├── notebooks                               
│   ├── personalised_cat_recommender.ipynb  <- main file with code walkthrough, results and discussion
├── output                                  <- model recommendations
└── scripts                                 <- helper scripts
```
