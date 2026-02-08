import pdfplumber
import pandas as pd
import re
from typing import List, Dict, Tuple, Any
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class PDFDataExtractor:
    def __init__(self, pdf_files: List[str]):
        """Initialize with list of PDF file paths"""
        self.pdf_files = pdf_files
        self.tables_data = {
            'table1_summary': None,
            'table2_land_use': None,
            'table3_winter_cereals': None,
            'table4_summer_cereals': None
        }
        
    def extract_all_tables(self):
        """Extract all tables from all PDFs"""
        all_tables = []
        
        for pdf_file in self.pdf_files:
            print(f"Processing {pdf_file}...")
            try:
                tables = self.extract_from_pdf(pdf_file)
                all_tables.extend(tables)
            except Exception as e:
                print(f"  Warning: Error processing {pdf_file}: {e}")
                continue
            
        # Organize extracted tables
        self.organize_tables(all_tables)
        return self.tables_data
    
    def extract_from_pdf(self, pdf_path: str) -> List[pd.DataFrame]:
        """Extract tables from a single PDF"""
        tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        # Extract text for analysis
                        text = page.extract_text() or ""
                        
                        # Look for table indicators
                        if any(keyword in text for keyword in ['Récapitulatif', 'SUPERFICIES', 'CEREALES', 'CEREALES D\'HIVER', 'CEREALES D\'ETE']):
                            # Extract tables from page
                            page_tables = page.extract_tables({
                                'vertical_strategy': 'text',
                                'horizontal_strategy': 'text',
                                'intersection_tolerance': 20,
                                'intersection_x_tolerance': 10
                            })
                            
                            for table in page_tables:
                                if table and len(table) > 1:  # At least header + 1 data row
                                    try:
                                        df = self.clean_table(table)
                                        if len(df) > 0:
                                            tables.append(df)
                                    except Exception as e:
                                        print(f"    Warning: Could not clean table on page {page_num}: {e}")
                                        continue
                    except Exception as e:
                        print(f"    Warning: Error on page {page_num}: {e}")
                        continue
                        
        except Exception as e:
            print(f"  Error opening PDF {pdf_path}: {e}")
            
        return tables
    
    def clean_table(self, table_data: List[List]) -> pd.DataFrame:
        """Clean and convert raw table data to DataFrame"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(table_data)
            
            # Remove completely empty rows and columns
            df = df.replace('', np.nan)
            df = df.dropna(how='all')
            df = df.dropna(axis=1, how='all')
            
            if df.empty:
                return pd.DataFrame()
            
            # Reset index
            df = df.reset_index(drop=True)
            
            # Clean cell values - handle both strings and other types
            df = df.applymap(lambda x: self.clean_cell(x))
            
            return df
            
        except Exception as e:
            print(f"    Error in clean_table: {e}")
            return pd.DataFrame()
    
    def clean_cell(self, cell_value: Any) -> str:
        """Clean a single cell value"""
        if pd.isna(cell_value):
            return ''
        
        try:
            # Convert to string and strip whitespace
            text = str(cell_value).strip()
            
            # Remove extra whitespace and newlines
            text = ' '.join(text.split())
            
            return text
        except:
            return ''
    
    def organize_tables(self, tables: List[pd.DataFrame]):
        """Organize extracted tables based on content patterns"""
        
        for i, df in enumerate(tables):
            if df.empty:
                continue
                
            try:
                text_representation = df.to_string()
                
                # Table 1: Récapitulatif des superficies
                if any(keyword in text_representation for keyword in ['Récapitulatif', 'Taux d\'accroissement', '2019/2018']):
                    self.tables_data['table1_summary'] = self.structure_table1(df)
                    
                # Table 2: SUPERFICIES DES TERRES UTILISEES
                elif any(keyword in text_representation for keyword in ['SUPERFICIES DES TERRES', 'Cultures herbacées', 'Prairies naturelles']):
                    self.tables_data['table2_land_use'] = self.structure_table2(df)
                    
                # Table 3: CEREALES D'HIVER
                elif any(keyword in text_representation for keyword in ['CEREALES D\'HIVER', 'BLE DUR', 'BLE TENDRE', 'ORGE', 'AVOINE']):
                    self.tables_data['table3_winter_cereals'] = self.structure_table3(df)
                    
                # Table 4: CEREALES D'ETE
                elif any(keyword in text_representation for keyword in ['CEREALES D\'ETE', 'MAIS', 'SORGHO']):
                    self.tables_data['table4_summer_cereals'] = self.structure_table4(df)
                    
            except Exception as e:
                print(f"  Warning: Error organizing table {i}: {e}")
                continue
    
    def structure_table1(self, df: pd.DataFrame) -> pd.DataFrame:
        """Structure Table 1: Récapitulatif des superficies"""
        try:
            df_clean = df.copy()
            
            # Find header row with years
            header_row_idx = -1
            for idx in range(min(10, len(df_clean))):
                row_text = ' '.join(df_clean.iloc[idx].astype(str).fillna('').tolist())
                if '2018' in row_text and '2019' in row_text:
                    header_row_idx = idx
                    break
            
            if header_row_idx >= 0:
                # Use found row as header
                df_clean.columns = df_clean.iloc[header_row_idx]
                df_clean = df_clean.iloc[header_row_idx + 1:]
            
            # Keep only rows with cereal names
            cereal_keywords = ['Céréales', 'Blé', 'Orge', 'Avoine', 'Triticale', 'Maïs', 'Sorgho']
            
            # Create a mask for rows containing cereal keywords
            mask = pd.Series(False, index=df_clean.index)
            for idx, row in df_clean.iterrows():
                row_text = ' '.join(row.astype(str).fillna('').tolist())
                if any(keyword in row_text for keyword in cereal_keywords):
                    mask[idx] = True
            
            df_clean = df_clean[mask]
            
            if df_clean.empty:
                return df_clean
            
            # Clean column names
            df_clean.columns = [self.clean_cell(col) for col in df_clean.columns]
            
            # Clean numerical values column by column
            for col in df_clean.columns:
                if col:  # Skip empty column names
                    df_clean[col] = df_clean[col].apply(lambda x: self.clean_numeric_single(x))
            
            return df_clean.reset_index(drop=True)
            
        except Exception as e:
            print(f"  Error in structure_table1: {e}")
            return pd.DataFrame()
    
    def structure_table2(self, df: pd.DataFrame) -> pd.DataFrame:
        """Structure Table 2: Land use data"""
        try:
            df_clean = df.copy()
            
            # Find rows starting with wilaya numbers (1, 2, 3, etc.)
            wilaya_rows = []
            for idx in range(len(df_clean)):
                first_cell = str(df_clean.iloc[idx, 0]) if len(df_clean.columns) > 0 else ''
                if re.match(r'^\s*\d+\s+[A-Z]', first_cell):
                    wilaya_rows.append(idx)
            
            if wilaya_rows:
                # Get header from row before first wilaya
                if wilaya_rows[0] > 0:
                    header_row = wilaya_rows[0] - 1
                    df_clean.columns = df_clean.iloc[header_row]
                    df_clean = df_clean.iloc[wilaya_rows[0]:]
                else:
                    # No header found, use default
                    df_clean.columns = [f'Col_{i}' for i in range(len(df_clean.columns))]
                
                # Keep only wilaya rows
                mask = df_clean.iloc[:, 0].astype(str).str.contains(r'^\s*\d+\s+', na=False)
                df_clean = df_clean[mask]
                
                # Clean numerical values
                for col in df_clean.columns[1:]:  # Skip first column
                    df_clean[col] = df_clean[col].apply(lambda x: self.clean_numeric_single(x))
                
                return df_clean.reset_index(drop=True)
            
            return df_clean
            
        except Exception as e:
            print(f"  Error in structure_table2: {e}")
            return pd.DataFrame()
    
    def structure_table3(self, df: pd.DataFrame) -> pd.DataFrame:
        """Structure Table 3: Winter cereals data - simplified"""
        try:
            df_clean = df.copy()
            
            # Find rows with wilaya data
            wilaya_rows = []
            for idx in range(len(df_clean)):
                first_cell = str(df_clean.iloc[idx, 0]) if len(df_clean.columns) > 0 else ''
                if re.match(r'^\s*\d+\s+[A-Z\-]', first_cell):
                    wilaya_rows.append(idx)
            
            if wilaya_rows:
                # Get all wilaya rows
                df_wilayas = df_clean.iloc[wilaya_rows].copy()
                
                # Simple extraction: just get first few columns
                if len(df_wilayas.columns) > 3:
                    # Take first 4 columns as basic info
                    df_wilayas = df_wilayas.iloc[:, :4]
                    df_wilayas.columns = ['Wilaya', 'Col1', 'Col2', 'Col3']
                    
                    # Clean values
                    for col in df_wilayas.columns[1:]:
                        df_wilayas[col] = df_wilayas[col].apply(lambda x: self.clean_numeric_single(x))
                
                return df_wilayas.reset_index(drop=True)
            
            return df_clean
            
        except Exception as e:
            print(f"  Error in structure_table3: {e}")
            return pd.DataFrame()
    
    def structure_table4(self, df: pd.DataFrame) -> pd.DataFrame:
        """Structure Table 4: Summer cereals data - simplified"""
        try:
            df_clean = df.copy()
            
            # Find rows with wilaya data
            wilaya_rows = []
            for idx in range(len(df_clean)):
                first_cell = str(df_clean.iloc[idx, 0]) if len(df_clean.columns) > 0 else ''
                if re.match(r'^\s*\d+\s+[A-Z\-]', first_cell):
                    wilaya_rows.append(idx)
            
            if wilaya_rows:
                df_wilayas = df_clean.iloc[wilaya_rows].copy()
                
                # Simple extraction
                if len(df_wilayas.columns) > 1:
                    df_wilayas = df_wilayas.iloc[:, :3]  # First 3 columns
                    df_wilayas.columns = ['Wilaya', 'Col1', 'Col2']
                    
                    # Clean values
                    for col in df_wilayas.columns[1:]:
                        df_wilayas[col] = df_wilayas[col].apply(lambda x: self.clean_numeric_single(x))
                
                return df_wilayas.reset_index(drop=True)
            
            return df_clean
            
        except Exception as e:
            print(f"  Error in structure_table4: {e}")
            return pd.DataFrame()
    
    def clean_numeric_single(self, value: Any):
        """Clean and convert a single numeric value"""
        try:
            if pd.isna(value):
                return None
            
            text = str(value).strip()
            if text == '' or text.lower() in ['nan', 'none', 'null']:
                return None
            
            # Remove spaces between digits (e.g., "1 000" -> "1000")
            text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
            
            # Replace comma with dot for decimal
            text = text.replace(',', '.')
            
            # Remove any non-numeric characters except dots, minus, and parentheses
            # Keep parentheses for negative numbers (accounting style)
            text = re.sub(r'[^\d\.\-\(\)]', '', text)
            
            # Handle accounting negative numbers: (123) -> -123
            if text.startswith('(') and text.endswith(')'):
                text = '-' + text[1:-1]
            
            # Try to convert to float or int
            if '.' in text:
                return float(text) if text.replace('.', '').replace('-', '').isdigit() else None
            else:
                return int(text) if text.replace('-', '').isdigit() and text != '' else None
                
        except Exception as e:
            return None
    
    def save_to_excel(self, output_file: str = 'agriculture_data.xlsx'):
        """Save all tables to an Excel file with multiple sheets"""
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                for table_name, df in self.tables_data.items():
                    if df is not None and not df.empty:
                        # Use short sheet names
                        sheet_name = table_name.replace('table', '').replace('_', ' ').title()
                        sheet_name = sheet_name[:31]  # Excel sheet name limit
                        
                        # Write to Excel
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                print(f"\nData saved to {output_file}")
                
        except Exception as e:
            print(f"Error saving to Excel: {e}")
    
    def display_summary(self):
        """Display summary of extracted data"""
        print("\n" + "=" * 60)
        print("EXTRACTION SUMMARY")
        print("=" * 60)
        
        extracted_count = 0
        for table_name, df in self.tables_data.items():
            if df is not None and not df.empty:
                extracted_count += 1
                print(f"\n{table_name}:")
                print(f"  Rows: {len(df)}")
                print(f"  Columns: {len(df.columns)}")
                print(f"  First few rows:")
                print(df.head(3).to_string())
                print(f"\n  Columns: {list(df.columns)}")
            else:
                print(f"\n{table_name}: Not extracted or empty")
        
        print(f"\nTotal tables extracted: {extracted_count}")
        print("=" * 60)


# Alternative simpler version for direct extraction
def extract_simple_tables(pdf_files):
    """Simpler extraction focusing on finding tables"""
    all_data = {}
    
    for pdf_file in pdf_files:
        print(f"\nProcessing {pdf_file}...")
        
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    
                    # Check for specific table indicators
                    if 'Récapitulatif' in text:
                        print(f"  Found 'Récapitulatif' on page {page_num + 1}")
                        tables = page.extract_tables()
                        for i, table in enumerate(tables):
                            if table:
                                df = pd.DataFrame(table)
                                df = df.replace('', np.nan).dropna(how='all').dropna(axis=1, how='all')
                                if not df.empty:
                                    key = f"{os.path.basename(pdf_file)}_recap_page{page_num+1}_table{i}"
                                    all_data[key] = df
                    
                    elif 'SUPERFICIES DES TERRES' in text:
                        print(f"  Found 'SUPERFICIES DES TERRES' on page {page_num + 1}")
                        tables = page.extract_tables()
                        for i, table in enumerate(tables):
                            if table:
                                df = pd.DataFrame(table)
                                df = df.replace('', np.nan).dropna(how='all').dropna(axis=1, how='all')
                                if not df.empty:
                                    key = f"{os.path.basename(pdf_file)}_landuse_page{page_num+1}_table{i}"
                                    all_data[key] = df
                    
                    elif 'CEREALES D\'HIVER' in text:
                        print(f"  Found 'CEREALES D\'HIVER' on page {page_num + 1}")
                        tables = page.extract_tables()
                        for i, table in enumerate(tables):
                            if table:
                                df = pd.DataFrame(table)
                                df = df.replace('', np.nan).dropna(how='all').dropna(axis=1, how='all')
                                if not df.empty:
                                    key = f"{os.path.basename(pdf_file)}_winter_page{page_num+1}_table{i}"
                                    all_data[key] = df
                    
                    elif 'CEREALES D\'ETE' in text:
                        print(f"  Found 'CEREALES D\'ETE' on page {page_num + 1}")
                        tables = page.extract_tables()
                        for i, table in enumerate(tables):
                            if table:
                                df = pd.DataFrame(table)
                                df = df.replace('', np.nan).dropna(how='all').dropna(axis=1, how='all')
                                if not df.empty:
                                    key = f"{os.path.basename(pdf_file)}_summer_page{page_num+1}_table{i}"
                                    all_data[key] = df
                                    
        except Exception as e:
            print(f"  Error processing {pdf_file}: {e}")
            continue
    
    return all_data


def main():
    # List your PDF files
    pdf_files = [
        'SERIE-B-2016.pdf',
        'SERIE-B-2017.pdf', 
        'SERIE-B-2018.pdf',
        'SERIE-B-2019.pdf'
    ]
    
    # Check if files exist
    existing_files = []
    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            existing_files.append(pdf_file)
            print(f"Found: {pdf_file}")
        else:
            print(f"Warning: File {pdf_file} not found")
    
    if not existing_files:
        print("No PDF files found. Please update the file names in the script.")
        return
    
    print(f"\nFound {len(existing_files)} PDF file(s)")
    
    # Option 1: Use the main extractor
    print("\n" + "=" * 60)
    print("OPTION 1: Using main extractor")
    print("=" * 60)
    
    extractor = PDFDataExtractor(existing_files)
    tables_data = extractor.extract_all_tables()
    extractor.display_summary()
    extractor.save_to_excel('agriculture_data_extracted.xlsx')
    
    # Option 2: Use simpler extractor
    print("\n" + "=" * 60)
    print("OPTION 2: Using simpler extractor")
    print("=" * 60)
    
    simple_data = extract_simple_tables(existing_files)
    
    if simple_data:
        print(f"\nFound {len(simple_data)} tables in simpler extraction")
        
        # Save all simple tables to Excel
        with pd.ExcelWriter('simple_extracted_tables.xlsx', engine='openpyxl') as writer:
            for sheet_name, df in simple_data.items():
                # Truncate sheet name if too long
                safe_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=safe_name, index=False)
        
        print("Simple extraction saved to 'simple_extracted_tables.xlsx'")
        
        # Show sample of each table
        for key, df in list(simple_data.items())[:5]:  # Show first 5
            print(f"\n{key}:")
            print(f"  Shape: {df.shape}")
            print(df.head(2).to_string())
    else:
        print("No tables found in simpler extraction")


if __name__ == "__main__":
    # Install required packages if not installed
    required_packages = ['pdfplumber', 'pandas', 'openpyxl', 'numpy']
    
    import subprocess
    import sys
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    main()