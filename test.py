import pprint

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

#Asaoka Report .pdf generation
from helpers.reporter import reporter_Asaoka
with open("Asaoka_Report.pdf", "wb") as f:
    data = reporter_Asaoka(["F3-R06a-SM-30"], '2025-03-29', '2025-04-05', '2025-07-11')
    print(type(data))
    print(data)
    f.write(data)

#Settlement Plate overview, JSON output
from helpers.datasources import SM_overview
pprint.pp(SM_overview(['F3-R11a-SM-01','F3-R11a-SM-02','F3-R11a-SM-03','F3-R11a-SM-04','F3-R11a-SM-05','F3-R11a-SM-06',
                   'F3-R11a-SM-07','F3-R11a-SM-08','F3-R11a-SM-09','F3-R11a-SM-10','F3-R11a-SM-11','F3-R11a-SM-12',
                   'F3-R11a-SM-13','F3-R11a-SM-14','F3-R11a-SM-15','F3-R11a-SM-16','F3-R11a-SM-17','F3-R11a-SM-18',
                   'F3-R11a-SM-19','F3-R11a-SM-20','F3-R11a-SM-21','F3-R11a-SM-22','F3-R11a-SM-23','F3-R11a-SM-24',
                   'F3-R11a-SM-25','F3-R11a-SM-26','F3-R11b-SM-01','F3-R11b-SM-02','F3-R11b-SM-03','F3-R11b-SM-04',
                   'F3-R11b-SM-05','F3-R11b-SM-06']))

##OPTIONAL: Asoaka assessment details for single plate
from helpers.asaoka import Asaoka_data
print(Asaoka_data('F3-R06a-SM-30', '2025-03-29', '2025-04-05', '2025-07-11'))

#Settlement Plate plot .pdf output
from helpers.settlement_data import reporter_Settlement
with open("Combined Settlement Plot.pdf", "wb") as f:
    f.write(reporter_Settlement(["F3-R06a-SM-30"], '2025-04-05'))
