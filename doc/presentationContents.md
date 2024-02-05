# What have we done
## Cleaned Dataset
- Reading the data into a dataframe
- removing Visitor_Score
- transform sales column: 0 -> 0; rest = 1
- replaced potentially problematic chars (for example spaces)
- merged phone-columns with their non-phone-equivalents (if those existed)
- created dummy values for categorical features
- replaced NaN with 0 where applicable
- transformed floats to int where applicable (years, number of children etc.)
- removed features that looked too hard to deal with or seemed not worth it (given time constraint)
- removed highly interrelated features

## Feature Selection
- calculated a correlation matrix for sales variable
- features most related (positively or negatively) to the sales variable were selected for the model building

## Building the models
- generated synthetic data points for minority class 
- split data into train and test (80/20)
- creating confusion matrix, calculating accuracy

## Optimization
- Possibilities: GridSearch, RandomSearch, Bayesian Optimization
- here: Bayesian Optimization


# recommendations for better data collection
- do not allow all data types for the fields. example: in "Einreisejahr" there is a value "DFGKJDFS"
- also check the length of values (for example: "Geburtsjahr" should always consist of 4 digits)
- come up with better ideas for umlauts ("Nationalit_t" is not a good name for a column)
- allow only categorical values for fields like "Grund der Absage" -> many values are semantically the same but differ in syntax -> data cleaning necessary
- sometimes weird datatypes (example: "Geburtsjahr" is saved as a float) -> use less storage when changing to int
- use a consistent naming scheme (sometimes snake case with all lower case, sometimes first letter is upper case; randomly changing between german and english)
- Berufsausbildung_anerkannt: [nan 'Ja (z.b. von ZSBA)' 'Nein' 'weiß ich nicht'] -> should be 'ja', 'nein', 'null'

# Interesting findings
- Owner correlation with sales is 23% -> could be used to find out who should get a raise (also, super problematic when it comes to employer rights)