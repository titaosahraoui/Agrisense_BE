import camelot
import pandas as pd
import os
from pathlib import Path
import logging

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def extract_tables_camelot(pdf_path, output_dir="extracted_tables", flavor='lattice'):
    """
    Extract tables from PDF using Camelot
    
    Parameters:
    -----------
    pdf_path : str
        Path to the PDF file
    output_dir : str
        Directory to save extracted CSV files
    flavor : str
        'lattice' for tables with borders, 'stream' for tables without borders
    """
    logger = setup_logging()
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Validate PDF exists
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return
    
    logger.info(f"Processing PDF: {pdf_path}")
    logger.info(f"Using flavor: {flavor}")
    
    try:
        # Read PDF tables
        tables = camelot.read_pdf(
            pdf_path,
            pages='all',  # Extract from all pages
            flavor=flavor,
            suppress_stdout=False,
            strip_text='\n',
            edge_tol=500,  # Adjust for table edge detection
            row_tol=10,    # Adjust for row detection
        )
        
        logger.info(f"Found {len(tables)} tables")
        
        # Save each table to CSV and Excel
        summary_data = []
        for i, table in enumerate(tables):
            try:
                # Get parsing report
                report = table.parsing_report
                page = report.get('page', 'unknown')
                
                # Sanitize PDF name for filename
                pdf_name = Path(pdf_path).stem
                safe_pdf_name = "".join([c for c in pdf_name if c.isalpha() or c.isdigit() or c in (' ', '-', '_')]).strip()
                safe_pdf_name = safe_pdf_name.replace(' ', '_')
                
                # Generate robust filename: {pdf_name}_p{page}_t{index}.csv
                base_name = f"{safe_pdf_name}_p{page}_t{i+1:03d}"
                csv_path = os.path.join(output_dir, f"{base_name}.csv")
                excel_path = os.path.join(output_dir, f"{base_name}.xlsx")
                
                # Save to CSV
                table.df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                
                # Save to Excel (preserves formatting better)
                table.df.to_excel(excel_path, index=False)
                
                # Save raw data as JSON for debugging
                json_path = os.path.join(output_dir, f"{base_name}_raw.json")
                table.df.to_json(json_path, orient='records', force_ascii=False)
                
                # Collect summary info
                summary_data.append({
                    'table_num': i + 1,
                    'page': report.get('page', 'N/A'),
                    'accuracy': report.get('accuracy', 0),
                    'whitespace': report.get('whitespace', 0),
                    'order': report.get('order', 0),
                    'rows': len(table.df),
                    'columns': len(table.df.columns),
                    'csv_file': csv_path,
                    'excel_file': excel_path
                })
                
                logger.info(f"  Table {i+1}: {len(table.df)} rows × {len(table.df.columns)} cols "
                           f"(Accuracy: {report.get('accuracy', 0):.1%})")
                
                # Log first few rows for verification
                logger.debug(f"  First row: {table.df.iloc[0].tolist()}")
                
            except Exception as e:
                logger.error(f"Error processing table {i+1}: {str(e)}")
                continue
        
        # Create summary report
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(output_dir, "extraction_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            
            # Also save summary as markdown for readability
            markdown_path = os.path.join(output_dir, "SUMMARY.md")
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write("# PDF Table Extraction Summary\n\n")
                f.write(f"**Source PDF:** {pdf_path}\n")
                f.write(f"**Extraction Method:** Camelot ({flavor})\n")
                f.write(f"**Total Tables Found:** {len(tables)}\n\n")
                
                f.write("## Table Details\n\n")
                f.write("| Table | Page | Rows | Columns | Accuracy | Files |\n")
                f.write("|-------|------|------|---------|----------|-------|\n")
                
                for item in summary_data:
                    f.write(f"| {item['table_num']} | {item['page']} | {item['rows']} | {item['columns']} | "
                           f"{item['accuracy']:.1%} | [CSV]({os.path.basename(item['csv_file'])}) • "
                           f"[Excel]({os.path.basename(item['excel_file'])}) |\n")
            
            logger.info(f"Summary saved to: {summary_path}")
            logger.info(f"Markdown summary: {markdown_path}")
            
            # Print overall statistics
            avg_accuracy = summary_df['accuracy'].mean()
            logger.info(f"\n📊 Extraction Complete:")
            logger.info(f"   Total tables: {len(tables)}")
            logger.info(f"   Average accuracy: {avg_accuracy:.1%}")
            logger.info(f"   Output directory: {os.path.abspath(output_dir)}")
    
    except Exception as e:
        logger.error(f"Failed to process PDF: {str(e)}")
        # Try alternative flavor if lattice fails
        if flavor == 'lattice':
            logger.info("Trying stream flavor as fallback...")
            extract_tables_camelot(pdf_path, output_dir + "_stream", flavor='stream')

def cleanup_dataframe(df):
    """
    Clean up extracted dataframe
    """
    # Remove empty rows and columns
    df = df.dropna(how='all').dropna(axis=1, how='all')
    
    # Remove duplicate header rows
    df = df[~df.apply(lambda row: row.astype(str).str.contains('Unnamed').any(), axis=1)]
    
    # Clean column names
    df.columns = [str(col).strip().replace('\n', ' ') for col in df.columns]
    
    # Remove extra whitespace from all cells
    df = df.map(lambda x: str(x).strip() if pd.notnull(x) else x)
    
    return df

def batch_process_pdfs(pdf_folder, output_base_dir="extracted_data"):
    """
    Process multiple PDFs in a folder
    """
    logger = setup_logging()
    pdf_folder = Path(pdf_folder)
    
    if not pdf_folder.exists():
        logger.error(f"Folder not found: {pdf_folder}")
        return
    
    pdf_files = list(pdf_folder.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        output_dir = Path(output_base_dir) / pdf_file.stem
        logger.info(f"\nProcessing: {pdf_file.name}")
        extract_tables_camelot(str(pdf_file), str(output_dir))

def main():
    """
    Main execution function
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract tables from PDF using Camelot')
    parser.add_argument('pdf_path', help='Path to PDF file or folder')
    parser.add_argument('-o', '--output', default='extracted_tables', 
                       help='Output directory (default: extracted_tables)')
    parser.add_argument('-f', '--flavor', choices=['lattice', 'stream'], default='lattice',
                       help='Extraction flavor: lattice for bordered tables, stream for borderless (default: lattice)')
    parser.add_argument('-b', '--batch', action='store_true',
                       help='Process all PDFs in a folder')
    
    args = parser.parse_args()
    
    if args.batch:
        batch_process_pdfs(args.pdf_path, args.output)
    else:
        extract_tables_camelot(args.pdf_path, args.output, args.flavor)

if __name__ == "__main__":
    # Install requirements first:
    # pip install camelot-py[cv] pandas openpyxl
    
    # Example usage:
    # python script.py SERIE-B-2019-8-16.pdf -o results -f lattice
    # python script.py . -b -o all_results  # batch process
    
    main()