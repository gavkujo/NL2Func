from helpers.reporter import reporter_Asaoka
from helpers.asaoka import Asaoka_data
from helpers.settlement_data import reporter_Settlement
from helpers.datasources import SM_overview
import pprint


def Func1(id, SCD, ASD, max_date):
    """Process Asaoka data with given parameters."""
    # Implementation of Asaoka data processing
    try:
        data = Asaoka_data(id, SCD, ASD, max_date=None, asaoka_days=7, period=0, n=4)
        print(data)
    except Exception as e:
        print(">>> error <<<\n", e)
    return f"===USER DATA===\n Processed Asaoka data is given below for your analysis: \n {data}"  

def Func2(ids, SCD, ASD, max_date):
    """Generate a report for Asaoka data with given parameters."""
    # Implementation of reporter generation
    try:
        with open("pdf/asaoka_report.pdf", "wb") as f:
            f.write(reporter_Asaoka(ids, SCD, ASD, max_date, n=4, asaoka_days=7, dtick=500))
    except Exception as e:
        print(">>> error <<<\n", e)
    return "==PDF ALERT==:\n Instruction: Tell the user that a PDF with the data has been made and stored for them to download."

def Func3(ids, max_date):
    """Plot combined data for given ids and max_date."""
    # Implementation of plotting
    try:
        with open("pdf/Combined_settlement_plot.pdf", "wb") as f:
            f.write(reporter_Settlement(ids, max_date))
    except Exception as e:
        print(">>> error <<<\n", e)
    return f"==PDF ALERT==:\n Instruction: Tell the user that an IMAGE with the plotted graphs has been made and stored for them to download."

def Func4(ids):
    try:
        data = SM_overview(ids)
        pprint.pp(data)
    except Exception as e:
        print(">>> error <<<\n", e)
    return f"===USER DATA===\n Processed Settlement Plate data for all the plates are given below for your analysis: \n {data}" 


