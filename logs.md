## V2.4 as of 23 July 2025

Vocab size: 82
Epochs: 5
Tokeniser: Char

Test 1

- "Run the settlement PDF for: F3-R09q-SM-30, F3-R26y-SM-55, F3-R55c-SM-74, F3-R81v-SM-26, F3-R82t-SM-20. SCD 2025-09-07, ASD 2025-09-10, max date 2025-11-17."
- {'function': 'reporter_Asaoka', 'args': {'ids': ['F3-R01p-SM-59', 'F3-R20b-SM-59', 'F3-R51f-SM-5', 'F3-R97i-SM-5', 'F3-R97y-SM-5'], 'SCD': '2025-09-01', 'ASD': '2025-09-01', 'max_date': '2025-09-09'}}

Test 2

- "What's the predicted settlement for F3-R10b-SM-11? Surcharge was Feb 10 2025, assess Feb 24 to Apr 15"
- {'function': 'reporter_Asaoka', 'args': {'ids': ['F3-R01b-SM-5', 'F3-R111l-SM-5'], 'SCD': '2025-10-01', 'ASD': '2025-10-01', 'max_date': '2025-01-04'}}

Test 3

- "I need a document buddy for F3-R10z-SM-39, from start date 22nd July 2025, etc"
- {'function': 'reporter_Asaoka', 'args': {'ids': ['F3-R333f-SM-33', 'F3-R93f-SM-3'], 'SCD': '2025-12-29', 'ASD': '2025-02-1', 'max_date': '2025-03-1'}}

Overall Validation:
100 values tested
100% syntax
67% correct function identified
10% correct parameters identified

## 