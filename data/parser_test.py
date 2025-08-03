import re
from dateparser import parse as date_parse

# Plate regex, with your constraints (1-80 for plate number)
PLATE_REGEX = r'F3\-R\d{2}[a-z]{1}\-SM\-(?:[0-9][0-9]?|80)'

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

class MissingSlot(Exception):
    def __init__(self, slot):
        self.slot = slot

def find_plates(text):
    """Extract plate IDs as list."""
    return re.findall(PLATE_REGEX, text)

# a helper month→num map to keep things clean
MONTH_MAP = {
    'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
    'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
    'sep': '09', 'sept': '09', 'oct': '10', 'nov': '11', 'dec': '12'
}

def normalize_date_input(date_str):
    """Parse user date input and convert to YYYY-MM-DD, retry until valid."""
    date_str = date_str.strip()
    date_patterns = [
        r"(\d{2})[\-/](\d{2})[\-/](\d{4})",        # DD-MM-YYYY or DD/MM/YYYY
        r"(\d{4})[\-/](\d{2})[\-/](\d{2})",        # YYYY-MM-DD or YYYY/MM/DD
        r"(\d{2})[\-/](\d{2})",                    # DD-MM or DD/MM (no year)
        # **Move DD Month [Year] *before* Month Year!**
        r"(\d{1,2})\s+([A-Za-z]+)\s*(\d{4})?",     # 16 Aug OR 16 Aug 2024
        r"([A-Za-z]+)\s+(\d{2,4})",                # Aug 24 or August 2024
        r"([A-Za-z]+)\s+(\d{1,2})",                # Aug 16
        r"(\d{1,2})\s+([A-Za-z]+)"                 # 16 August
    ]

    for pat in date_patterns:
        m = re.search(pat, date_str)
        if not m:
            continue

        # 1) DD-MM-YYYY
        if pat == date_patterns[0]:
            d, mth, y = m.groups()
            return f"{y}-{mth.zfill(2)}-{d.zfill(2)}"

        # 2) YYYY-MM-DD
        if pat == date_patterns[1]:
            y, mth, d = m.groups()
            return f"{y}-{mth.zfill(2)}-{d.zfill(2)}"

        # 3) DD-MM (no year → assume current year)
        if pat == date_patterns[2]:
            d, mth = m.groups()
            return f"2025-{mth.zfill(2)}-{d.zfill(2)}"

        # 4) DD Month [Year]
        if pat == date_patterns[3]:
            d, month, y = m.groups()
            mon = month.lower()[:3]
            mnum = MONTH_MAP.get(mon, '01')
            year = y if y and len(y)==4 else '2025'
            return f"{year}-{mnum}-{d.zfill(2)}"

        # 5) Month Year
        if pat == date_patterns[4]:
            month, year = m.groups()
            mon = month.lower()[:3]
            mnum = MONTH_MAP.get(mon, '01')
            y = year if len(year)==4 else f"20{year[-2:]}"
            return f"{y}-{mnum}-01"

        # 6) Month DD
        if pat == date_patterns[5]:
            month, d = m.groups()
            mon = month.lower()[:3]
            mnum = MONTH_MAP.get(mon, '01')
            return f"2025-{mnum}-{d.zfill(2)}"

        # 7) DD Month
        if pat == date_patterns[6]:
            d, month = m.groups()
            mon = month.lower()[:3]
            mnum = MONTH_MAP.get(mon, '01')
            return f"2025-{mnum}-{d.zfill(2)}"

    # fallback to dateparser for the wild ones
    dt = date_parse(date_str, settings={'PREFER_DATES_FROM': 'past'})
    return dt.date().isoformat() if dt else None

def input_date_slot(slot_name):
    """Prompt user for date input and normalize it."""
    while True:
        raw = input(f"Enter {slot_name} (date): ")
        val = normalize_date_input(raw)
        if val:
            return val
        print("Invalid date format. Please try again.")

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
    # Only extract plates from the original user input (not slot-filling context)
    # The dispatcher always appends slot-filling lines as '\nSCD: ...', etc.
    # So, split user_text into two parts: the original prompt, and any appended slot-filling lines
    # We'll treat the first paragraph as the original user input, and any lines matching 'slot: value' as slot-filling context
    # (This is robust because the dispatcher always appends in this format)

    # Split lines
    lines = user_text.splitlines()
    # The original user input is always the first line (or block before any slot-filling lines)
    orig_lines = []
    slot_lines = []
    for line in lines:
        if re.match(r'^(SCD|ASD|max_date):', line.strip(), re.IGNORECASE):
            slot_lines.append(line)
        else:
            orig_lines.append(line)
    orig_text = '\n'.join(orig_lines)

    plates = find_plates(orig_text)
    if not plates:
        raise MissingSlot('plates')

    needed_slots = {
        'Asaoka_data': ['SCD', 'ASD', 'max_date'],
        'reporter_Asaoka': ['SCD', 'ASD', 'max_date'],
        'plot_combi_S': ['max_date'],
    }

    if func_name not in needed_slots:
        raise ValueError(f"Function '{func_name}' not supported.")

    # Only check for slot values in the slot-filling context (never extract from original user input)
    slot_values = {}
    for slot in needed_slots[func_name]:
        # Look for a line like 'SCD: ...' (case-insensitive)
        found = False
        for sline in slot_lines:
            m = re.match(rf'^{slot}:\s*(.+)$', sline.strip(), re.IGNORECASE)
            if m:
                slot_values[slot] = m.group(1).strip()
                found = True
                break
        if not found:
            raise MissingSlot(slot)

    # Return params dict with extracted values
    if func_name == 'Asaoka_data':
        return {'id': plates[0], 'SCD': slot_values['SCD'], 'ASD': slot_values['ASD'], 'max_date': slot_values['max_date']}
    elif func_name == 'reporter_Asaoka':
        return {'ids': plates, 'SCD': slot_values['SCD'], 'ASD': slot_values['ASD'], 'max_date': slot_values['max_date']}
    elif func_name == 'plot_combi_S':
        return {'ids': plates, 'max_date': slot_values['max_date']}
    else:
        raise ValueError(f"Function '{func_name}' not supported.")
def main_test_examples():
    print("\n--- Testing parse_and_build on example usage ---\n")
    for i, (text, func_name) in enumerate(examples, 1):
        print(f"Example {i}: {func_name}")
        print(f"Input: {text}")
        if func_name == "None":
            print("  [SKIP] No function to parse.\n")
            continue
        try:
            plates = find_plates(text)
            print(f"  Extracted plates: {plates}")
            needed_slots = {
                'Asaoka_data': ['SCD', 'ASD', 'max_date'],
                'reporter_Asaoka': ['SCD', 'ASD', 'max_date'],
                'plot_combi_S': ['max_date'],
            }
            missing = []
            for slot in needed_slots[func_name]:
                if slot != 'id':
                    try:
                        raise MissingSlot(slot)
                    except MissingSlot as ms:
                        missing.append(ms.slot)
            print(f"  Missing slots (should be prompted): {missing}\n")
        except Exception as e:
            print(f"  [ERROR] {e}\n")


if __name__ == "__main__":
    main_test_examples()



