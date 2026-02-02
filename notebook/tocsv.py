import json
import csv
import pandas as pd

def extract_tabular_data(text):
    """Extract tabular data from the OCR text"""
    
    # Split into sections based on the document structure
    sections = {
        'cereals_hiver': [],
        'cereals_ete': [], 
        'cultures_industrielles': [],
        'legumes_secs': [],
        'cultures_maraicheres': [],
        'fourrages': [],
        'agrumes': [],
        'vignobles': [],
        'arboriculture': []
    }
    
    current_section = None
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Detect section headers
        if 'Céréales d\'hiver' in line:
            current_section = 'cereals_hiver'
            continue
        elif 'Céréales d\'été' in line:
            current_section = 'cereals_ete'
            continue
        elif 'Cultures industrielles' in line:
            current_section = 'cultures_industrielles'
            continue
        elif 'Légumes secs' in line:
            current_section = 'legumes_secs'
            continue
        elif 'Cultures maraîchères' in line:
            current_section = 'cultures_maraicheres'
            continue
        elif 'Fourrages' in line:
            current_section = 'fourrages'
            continue
        elif 'Agrumes' in line:
            current_section = 'agrumes'
            continue
        elif 'Vignobles' in line:
            current_section = 'vignobles'
            continue
            
        # Add data to current section
        if current_section and line and not any(keyword in line for keyword in 
                                              ['Direction', 'SERIE', 'Récapitulatif', '2018', '2019', 'Taux']):
            # Basic cleaning
            cleaned_line = ' '.join(line.split())
            if len(cleaned_line.split()) > 3:  # Only add lines that look like data
                sections[current_section].append(cleaned_line)
    
    return sections

# Load and process
with open('output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

text = data['document']['text']
sections = extract_tabular_data(text)

# Create separate CSV files for each section
for section_name, section_data in sections.items():
    if section_data:
        filename = f'{section_name}.csv'
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Data'])
            for row in section_data:
                writer.writerow([row])
        print(f"Created {filename}")

print("Processing complete!")