import random
import json
from datetime import datetime, timedelta

# ------------------------
# Utility 💡
# ------------------------

def random_plate():
    prefix = f"F3-R{str(random.randint(0, 45)).zfill(2)}{random.choice('abcd')}-SM-"
    suffix = str(random.randint(1, 80)).zfill(2)
    return prefix + suffix

def random_plate_list():
    return [random_plate() for _ in range(random.randint(1, 5))]

def random_textual_date():
    formats = [
        "%b %d", "%B %d", "%d %B", "%d %b", "%d %B %Y", "%B %d %Y",
        "%b %d, %Y", "%dth %B", "%dst %B", "%d/%m/%y", "%d-%m-%Y"
    ]
    base = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 730))
    return base.strftime(random.choice(formats)).replace(" 0", " ")

# ------------------------
# HANDCRAFTED TEMPLATES ✨
# ------------------------

reporter_Asaoka_templates = [
    "Generate a detailed PDF for the following plates: {plates}, surcharge completed {scd}, assessment from {asd} until {max_date}.",
    "Hey, I need a report of {plates}. Surcharge on {scd}, monitor from {asd} to {max_date} plsss 🙏",
    "Can you prep the Asaoka file for {plates}? Use {scd} as the start and check until {max_date}, assessment kicks in from {asd}.",
    "I wanna see the whole documentation for these plates: {plates}. Start at {scd}, assess from {asd}, end by {max_date}.",
    "Formal request: settlement report required for plates {plates}. SCDate: {scd}, ASD: {asd}, Deadline: {max_date}.",
    "Yo quick PDF run for plates {plates}? The surcharge was {scd}, assessment kicks off {asd} and ends {max_date}.",
    "Please compile a PDF report of {plates} beginning with surcharge {scd}, assessing from {asd} through {max_date}.",
    "Hey GPT overlord, can you document {plates}? SCD {scd}, ASD {asd}, wrap it up by {max_date}.",
    "Settlement overview for: {plates}. From {asd} (assessment), after surcharge {scd}, up till {max_date}.",
    "Need a doc for {plates} from {asd} to {max_date}. Surcharge wrapped on {scd}.",
    "I'm gonna need a full report for {plates} pls. We had surcharge on {scd}, assessed starting {asd}, end it {max_date}.",
    "Plot & PDF me the following: {plates}. Timeline: surcharge done on {scd}, assess from {asd}, cutoff {max_date}.",
    "Urgent: generate Asaoka report for {plates}. SCD: {scd}, ASD: {asd}, End: {max_date}.",
    "Hi system, initiate doc generation for {plates}. Assessment phase {asd} - {max_date}, post surcharge {scd}.",
    "Y’all I need the assessment report (PDF) for: {plates}. Timeframes are {scd}, {asd}, and {max_date}.",
    "Compile and export the full data visualization (PDF format) for plates {plates}. SCD = {scd}, ASD = {asd}, end = {max_date}.",
    "Bruh run that full settlement PDF on these: {plates}. Surcharge? {scd}. Assessment: {asd}–{max_date}.",
    "Help me get that report doc again for {plates}. Start on {asd}, end {max_date}, surcharge was {scd}."
]

Asaoka_data_templates = [
    "Can I get the current stats for {plate}? Looking for stuff like last settlement, GL, slope etc. Dates: SCD {scd}, ASD {asd}, till {max_date}.",
    "How’s {plate} performing lately? Use surcharge {scd}, assessment from {asd}, up to {max_date}. I wanna know the settlement trend.",
    "Tell me the predicted settlement and DOC for {plate} using these: SCD {scd}, ASD {asd}, and max {max_date}.",
    "Get the analysis data for {plate}. Key dates: surcharge on {scd}, assessment started {asd}, check until {max_date}.",
    "Yo what's the latest GL and measured settlement for {plate}? Use {scd}, {asd} and go up to {max_date}.",
    "Break down the Asaoka values for {plate} pls — dates are {scd} for surcharge, {asd} for assessment start, up to {max_date}.",
    "I just need the main metrics like slope, intercept, and r² for {plate}. Use these dates: SCD {scd}, ASD {asd}, max {max_date}.",
    "Run the Asaoka model for {plate}. What’s the settlement prediction and DOC? SCD: {scd}, ASD: {asd}, cutoff {max_date}.",
    "Give me a quick summary for plate {plate}. Start with surcharge {scd}, assess from {asd}, look till {max_date}.",
    "I wanna know everything measurable for {plate} — pairs, slope, r2, settlement. SCD={scd}, ASD={asd}, max={max_date}.",
    "Help me analyze {plate}. Use surcharge {scd}, assess between {asd} and {max_date}. Get me all numeric results.",
    "Give me a snapshot of {plate} at a glance. From surcharge {scd}, ASD {asd}, till max date {max_date}.",
    "Yo just crunch the numbers for {plate}. SCD = {scd}, ASD = {asd}, till {max_date}. What's the ground sayin?",
    "I’m looking for slope and prediction data for plate {plate}. Range: from {asd} to {max_date}, surcharge on {scd}.",
    "Need a full analysis output (not doc!) for {plate}. Like all the values and pairs, using {scd}, {asd}, and {max_date}.",
    "Do a single-plate assessment run on {plate} — show prediction, slope, intercept, whatever u got. Timeframe: {asd}–{max_date}, SCD: {scd}.",
    "Give me the Asaoka breakdown for just {plate}. Input dates: surcharge = {scd}, assessment = {asd}, end = {max_date}.",
    "What are the model results for {plate}? I'm talkin m, b, r2, predicted value. Use dates {scd}, {asd}, {max_date}."
]

plot_combi_S_templates = [
    "Plot settlements for: {plates}. Cutoff date: {max_date}.",
    "I need a chart of the following plates: {plates}. Latest date = {max_date}.",
    "Generate a graph of settlement data for {plates} till {max_date}.",
    "Give me a combined plot for plates {plates}, assess until {max_date}.",
    "Can you visualize {plates}? Use {max_date} as max.",
    "Provide a graph for settlement plates {plates} using data up to {max_date}.",
    "Plot this group: {plates}. End data range on {max_date}.",
    "Build me a line graph for {plates}, up to {max_date} pls.",
    "Gimme the graph for {plates} till {max_date}!",
    "Visualize settlement data for these: {plates}. Use {max_date} as the cutoff.",
    "I want a graph with the following plates: {plates}. Only include data before {max_date}.",
    "Make a combined plot of these plates: {plates}. End at {max_date}.",
    "settlement plot pls. for plates {plates}. latest date is {max_date}",
    "Graph time 📈: plates {plates}, stop on {max_date}.",
    "Yo, make that plot for {plates} — cutoff {max_date}!",
    "Multi-plate graph for: {plates}, last data on {max_date}.",
    "plot plates {plates} with max date {max_date} thanks bby",
    "Visual plot needed for {plates}. Cap it at {max_date}."
]

none_templates = [
    "idk bro just run something", "this is prolly invalid but try anyway", "can u cook up magic?", 
    "tell me something about the ground", "yo lowkey thinkin bout settlements rn", "gimme a vibe check",
    "get report... or not", "literally no idea what this does", "testing 123", 
    "settle me", "show me the dirt idk", "what's the most slay plate rn?", 
    "plot the vibe not the data", "this ain't it but try", "open something idc",
    "data go brrr", "science stuff please", "assess the universe or smth"
]

# ------------------------
# Main Dataset Generator 🚀
# ------------------------

def generate_entry(template, plates=None, plate=None, scd=None, asd=None, max_date=None):
    return template.format(
        plates=", ".join(plates or []),
        plate=plate or "",
        scd=scd or "",
        asd=asd or "",
        max_date=max_date or ""
    )

def generate_dataset(n=15000):
    dataset = []
    total_templates = (
        reporter_Asaoka_templates +
        Asaoka_data_templates +
        plot_combi_S_templates +
        none_templates
    )
    templates_per_type = {
        "reporter_Asaoka": reporter_Asaoka_templates,
        "Asaoka_data": Asaoka_data_templates,
        "plot_combi_S": plot_combi_S_templates,
        "None": none_templates
    }

    weights = {
        "reporter_Asaoka": 0.3,
        "Asaoka_data": 0.3,
        "plot_combi_S": 0.25,
        "None": 0.15
    }

    for fn, templates in templates_per_type.items():
        count = int(n * weights[fn])
        for _ in range(count):
            t = random.choice(templates)
            if fn == "reporter_Asaoka":
                entry = generate_entry(t, plates=random_plate_list(), scd=random_textual_date(), asd=random_textual_date(), max_date=random_textual_date())
            elif fn == "Asaoka_data":
                entry = generate_entry(t, plate=random_plate(), scd=random_textual_date(), asd=random_textual_date(), max_date=random_textual_date())
            elif fn == "plot_combi_S":
                entry = generate_entry(t, plates=random_plate_list(), max_date=random_textual_date())
            else:  # None
                entry = t
            dataset.append({"input": entry, "output": fn})

    random.shuffle(dataset)
    return dataset

def save_dataset(path="intent_dataset.json", total=15000):
    data = generate_dataset(total)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(data)} entries to {path}")

# ------------------------
# Run Script 🧃
# ------------------------

if __name__ == "__main__":
    save_dataset("intent_dataset.json", 15000)
