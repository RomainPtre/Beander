# Welcome to beander ☕
## The coffee matchmaker
https://beander.streamlit.app/

This application will provide the user (e.g., coffee roasters) with recommendations of coffee producers (e.g., coffee farms) based on a composite aromatic profile that the user can draw interactively 🎨

## 1) Main objectives of the project

To take advantage of machine learning approaches in order to suggests personnalized recommendations of coffee.

## 2) How does it work?

The dataset we used (source: https://github.com/jldbc/coffee-quality-database) was scrapped from the Coffee Quality Institute's review pages in January 2018 (https://database.coffeeinstitute.org/).

We then followed three different approaches that we applied on 5 selected features :

- Euclidean Distances : Provides reliable live calculations of the coffee ressemblance but at the cost of computational power.
- K-Means : Pre-trained Unsupervised Machine Learning approaches that provides fast segmentation of the data, but can be not suited to the most complexe data structures.
- Agglomerative clustering : Pre-trained Unsupervised Machine Learning that provides with a segmentation suited to more complexe data at the cost of computational power. We have added a K-NN analyses that will dynamically predict to which cluster the user input belongs to.

## 3) How to use it?

- Go to https://beander.streamlit.app/ , draw your dream coffee and click the 'Go fetch, Beander!' button-
- Deploy the app locally by cloning the repo and running the command line `python -m streamlit run ./demo_app/app.py` from the root folder 'Beander'. 

## 4) Next steps

We are currently working on implementing a broader aromatic profile to the application, along with geographical data to add more output infos.

## 5) Credits

Romain Pintore, Pietro Scalisi & Isa Marie-Magdelaine
