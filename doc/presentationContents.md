# recommendations for better data collection
- do not allow all data types for the fields. example: in "Einreisejahr" there is a value "DFGKJDFS"
- also check the length of values (for example: "Geburtsjahr" should always consist of 4 digits)
- come up with better ideas for umlauts ("Nationalit_t" is not a good name for a column)
- allow only categorical values for fields like "Grund der Absage" -> many values are semantically the same but differ in syntax -> data cleaning necessary
- sometimes weird datatypes (example: "Geburtsjahr" is saved as a float) -> use less storage when changing to int
- use a consistent naming scheme (sometimes snake case with all lower case, sometimes first letter is upper case; randomly changing between german and english)
- Berufsausbildung_anerkannt: [nan 'Ja (z.b. von ZSBA)' 'Nein' 'weiß ich nicht'] -> should be 'ja', 'nein', 'null'