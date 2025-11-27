import pdfplumber
import re

# Use pdf plumber to extract the text from the text-based pdf

start_page = 15   # Chapter 1 starts here (0-indexed → page 11 in the PDF)
end_page = 241

start_index = 3   # Chapter 1 starts here (0-indexed → page 11 in the PDF)
end_index = 7

def fix_drop_caps_regex(text):
    # Handle cases like:
    # "T\nhe", "T \nhe", "T\n he", "\nT\nhe", start-of-text "T\nhe", etc.
    # This will collapse an isolated capital letter followed by newline(s)/spaces
    # into the normal word: "T\nhe" -> "The"
    text = re.sub(r"([A-Z])\s*\n\s*([a-z])", r"\1\2", text)
    # Also cover cases where the drop-cap was separated by multiple newlines
    text = re.sub(r"([A-Z])\s*(?:\n\s*){2,}([a-z])", r"\1\2", text)
    return text

with pdfplumber.open("pdf_data/the_courage_to_be_desliked.pdf") as pdf:
    text = ""
    for i in range(start_page, end_page):
        page = pdf.pages[i]
        extracted = page.extract_text()
        if extracted:
            extracted = fix_drop_caps_regex(extracted)
            text += extracted + "\n"

    index = ''
    for i in range(start_index, end_index):
        page = pdf.pages[i]
        extracted = page.extract_text()
        if extracted:
            extracted = fix_drop_caps_regex(extracted)
            index += extracted + "\n"



#print(text)

index = index.split("\n")

#print("\n\n",index)

values_to_remove = ['Contents', 'Authors’ Note', 'Introduction','']

index_clean = [x for x in index if x not in values_to_remove]

print(index_clean)

for chapter in index_clean:
    print(chapter)
    text = text.replace(chapter, f"\n*****\n{chapter}\n*****\n")

print(text)

# Open (or create) a file in write mode
with open("text_data/the_courage_to_be_desliked.txt", "w") as file:
    file.write(text)