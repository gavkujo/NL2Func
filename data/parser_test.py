import re
from dateparser import parse as date_parse

# Plate regex, with your constraints (1-80 for plate number)
PLATE_REGEX = r'F3\-R\d{2}[a-z]{1}\-SM\-(?:[0-9][0-9]?|80)'

class MissingSlot(Exception):
    def __init__(self, slot):
        self.slot = slot

def find_plates(text):
    """Extract plate IDs as list."""
    return re.findall(PLATE_REGEX, text)

def normalize_date_input(date_str):
    """Parse user date input and convert to YYYY-MM-DD, retry until valid."""
    dt = date_parse(date_str, settings={'PREFER_DATES_FROM': 'past'})
    if dt:
        return dt.date().isoformat()
    else:
        return None

def input_date_slot(slot_name):
    """Prompt user for date input and normalize it."""
    # Deprecated: input gathering moved to Dispatcher
    pass

def build_function_call(func_name, plates, dates):
    """Generate the final python function call string."""
    if func_name == 'Asaoka_data':
        id_ = plates[0] if plates else None
        return f"Asaoka_data(id='{id_}', SCD='{dates['SCD']}', ASD='{dates['ASD']}', max_date='{dates['max_date']}')"
    
    elif func_name == 'reporter_Asaoka':
        ids_list = ", ".join(f"'{p}'" for p in plates)
        return f"reporter_Asaoka(ids=[{ids_list}], SCD='{dates['SCD']}', ASD='{dates['ASD']}', max_date='{dates['max_date']}')"
    
    elif func_name == 'plot_combi_S':
        ids_list = ", ".join(f"'{p}'" for p in plates)
        return f"plot_combi_S(ids=[{ids_list}], max_date='{dates['max_date']}')"
    
    else:
        return f"# Unknown function: {func_name}"
    
def parse_and_build(user_text: str, func_name: str):
    """
    user_text: raw input string from user
    func_name: one of 'Asaoka_data', 'reporter_Asaoka', 'plot_combi_S'

    Returns:
        final python function call string

    Raises:
        ValueError if required plates or dates are missing or invalid.
    """
    plates = find_plates(user_text)
    if not plates:
        raise MissingSlot('plates')

    needed_slots = {
        'Asaoka_data': ['SCD', 'ASD', 'max_date'],
        'reporter_Asaoka': ['SCD', 'ASD', 'max_date'],
        'plot_combi_S': ['max_date'],
    }

    if func_name not in needed_slots:
        raise ValueError(f"Function '{func_name}' not supported.")

    # Try to extract dates from user_text
    dates = {}
    for slot in needed_slots[func_name]:
        # Try to find slot value in user_text
        import re
        pattern = re.compile(rf"{slot}[:=]?\s*([\w\-/]+)", re.IGNORECASE)
        match = pattern.search(user_text)
        if match:
            val = normalize_date_input(match.group(1))
            if val:
                dates[slot] = val
            else:
                raise MissingSlot(slot)
        else:
            raise MissingSlot(slot)

    # Return params as dict for Dispatcher
    if func_name == 'Asaoka_data':
        return {'id': plates[0], 'SCD': dates['SCD'], 'ASD': dates['ASD'], 'max_date': dates['max_date']}
    elif func_name == 'reporter_Asaoka':
        return {'ids': plates, 'SCD': dates['SCD'], 'ASD': dates['ASD'], 'max_date': dates['max_date']}
    elif func_name == 'plot_combi_S':
        return {'ids': plates, 'max_date': dates['max_date']}
    else:
        raise ValueError(f"Function '{func_name}' not supported.")

def main():
    user_text = input("Paste your command text: ")
    # Assuming you have your function classifier done — just mock here:
    func_name = input("Enter function name (Asaoka_data, reporter_Asaoka, plot_combi_S): ").strip()

    plates = find_plates(user_text)
    if not plates:
        print("No valid plates found in input. Please try again.")
        return

    print(f"Detected plates: {plates}")

    # Define needed date slots per function
    needed_slots = {
        'Asaoka_data': ['SCD', 'ASD', 'max_date'],
        'reporter_Asaoka': ['SCD', 'ASD', 'max_date'],
        'plot_combi_S': ['max_date'],
    }

    if func_name not in needed_slots:
        print(f"Function '{func_name}' not supported.")
        return

    dates = {}
    for slot in needed_slots[func_name]:
        dates[slot] = input_date_slot(slot)

    # Output the final function call ready to paste/run
    func_call = build_function_call(func_name, plates, dates)
    print("\nYour final function call:")
    print(func_call)

if __name__ == "__main__":
    main()


# —— example usage ——
examples = [
("this is prolly invalid but try anyway", "None"),
("I want a graph with the following plates: F3-R03a-SM-54. Only include data before July 22 2025.", "plot_combi_S"),
("Give me a snapshot of F3-R01d-SM-25 at a glance. From surcharge 24/06/24, ASD Apr 23, 2025, till max date 13/10/25.","Asaoka_data"),
("Generate a detailed PDF for the following plates: F3-R16b-SM-21, F3-R45c-SM-70, surcharge completed August 31, assessment from June 15 2025 until 16 Nov.","reporter_Asaoka"),
("Break down the Asaoka values for F3-R00b-SM-78 pls — dates are 02st March for surcharge, 16 May 2024 for assessment start, up to 17/08/24.","Asaoka_data"),
("yo lowkey thinkin bout settlements rn","None"),
("Hi system, initiate doc generation for F3-R22c-SM-77, F3-R37d-SM-23, F3-R36a-SM-22, F3-R10d-SM-35, F3-R38a-SM-41. Assessment phase Oct 19, 2024 - Aug 16, post surcharge 14 October.","reporter_Asaoka"),
("Give me a combined plot for plates F3-R31b-SM-54, F3-R40a-SM-63, F3-R44c-SM-57, assess until 03 February 2025.","plot_combi_S"),
("Provide a graph for settlement plates F3-R17c-SM-46 using data up to 10 Aug.","plot_combi_S"),
("Can I get the current stats for F3-R43b-SM-14? Looking for stuff like last settlement, GL, slope etc. Dates: SCD Jan 13, ASD Mar 22, 2025, till Sep 4, 2024.","Asaoka_data"),
("I wanna know everything measurable for F3-R21c-SM-58 — pairs, slope, r2, settlement. SCD=15st January, ASD=18 August 2024, max=02 Dec.","Asaoka_data"),
("literally no idea what this does","None"),
("Give me a snapshot of F3-R37b-SM-52 at a glance. From surcharge Oct 5, ASD 29 December, till max date Apr 5, 2024.","Asaoka_data"),
("Plot settlements for: F3-R15c-SM-33. Cutoff date: January 28 2024.","plot_combi_S"),
("science stuff please","None"),
("I wanna know everything measurable for F3-R07a-SM-01 — pairs, slope, r2, settlement. SCD=26/04/24, ASD=Feb 16, 2024, max=07st March.","Asaoka_data"),
("Multi-plate graph for: F3-R42b-SM-20, F3-R22a-SM-79, F3-R29a-SM-04, last data on 13/10/24.","plot_combi_S"),
("plot plates F3-R06c-SM-33 with max date Aug 27 thanks bby","plot_combi_S"),
("Do a single-plate assessment run on F3-R41d-SM-40 — show prediction, slope, intercept, whatever u got. Timeframe: 18 June–29-12-2024, SCD: July 10 2024.","Asaoka_data"),
("I wanna know everything measurable for F3-R32c-SM-37 — pairs, slope, r2, settlement. SCD=July 7 2025, ASD=August 7 2024, max=22 Feb.","Asaoka_data"),
("Graph time 📈: plates F3-R01c-SM-39, F3-R07c-SM-32, stop on 08 Jul.","plot_combi_S"),
("What are the model results for F3-R00c-SM-77? I'm talkin m, b, r2, predicted value. Use dates 07 August 2025, August 19, 28 Aug.","Asaoka_data"),
("Provide a graph for settlement plates F3-R36d-SM-52, F3-R43a-SM-51, F3-R20a-SM-58, F3-R07b-SM-73 using data up to Jan 29, 2025.","plot_combi_S"),
("Need a doc for F3-R34a-SM-08, F3-R24d-SM-80, F3-R19d-SM-70 from 21/07/25 to 17-06-2024. Surcharge wrapped on 10th April.","reporter_Asaoka"),
("Plot this group: F3-R31d-SM-64, F3-R08b-SM-69, F3-R39c-SM-19. End data range on 30-07-2024.","plot_combi_S")
]

