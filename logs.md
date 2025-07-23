## V2.4 as of 23 July 2025

Vocab size: 82
Epochs: 5
Tokeniser: Char

Test 1

- "Run the settlement PDF for: F3-R09q-SM-30, F3-R26y-SM-55, F3-R55c-SM-74, F3-R81v-SM-26, F3-R82t-SM-20. SCD 2025-09-07, ASD 2025-09-10, max date 2025-11-17."
- {'function': 'reporter_Asaoka', 'args': {'ids': ['F3-R01p-SM-30', 'F3-R20b-SM-55', 'F3-R51f-SM-5', 'F3-R97i-SM-75', 'F3-R97y-SM-20'], 'SCD': '2025-09-01', 'ASD': '2025-09-01', 'max_date': '2025-09-09'}}

Test 2

- "What's the predicted settlement for F3-R10b-SM-11? Surcharge was Feb 10 2025, assess Feb 24 to Apr 15"
- {'function': 'Asaoka_data', 'args': {'ids': 'F3-R10b-SM-59', 'SCD': '2025-10-01', 'ASD': '2025-10-01', 'max_date': '2025-01-04'}}

Test 3

- "I need a document buddy for F3-R10z-SM-39, from start date 22nd July 2025, etc"
- {'function': 'reporter_Asaoka', 'args': {'ids': ['F3-R33f-SM-39'], 'SCD': '2025-12-29', 'ASD': '2025-02-1', 'max_date': '2025-03-1'}}

Overall Validation:
100 values tested
100% syntax
67% correct function identified
10% correct parameters identified

Remarks:

1. Complete Vocab size to 45x4. data set size would need to go 10k -> 100k.

2. Turn model into a classifier. classifies prompt into a specific model then extracts and or asks user for parameter values.

ex. What's the predicted settlement for F3-R10b-SM-11? Surcharge was Feb 10 2025, assess Feb 24 to Apr 15

classified to Asoka_data
[THEN] extract args from prompt using regex or similar.
[OR] extract args by asking user if needed.