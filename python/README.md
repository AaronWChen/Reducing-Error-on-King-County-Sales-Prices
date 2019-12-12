This folder contains Python code needed to perform the data analysis.

Inside the joblib folder is a pickled (joblib) specific random state version that can be opened and used to generate the same results as the ones presented. This is also faster than starting from scratch.


To generate your own model: 
1. Fork and clone the parent repository
2. Run the prep_data.py script with a new random state (it defaults to 50)
3. Run the predict_price.py script to perform regression