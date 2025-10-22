import pandas as pd
import os
import glob
from datetime import datetime, timedelta
import numpy as np


def clean_date_format(date_value):
    """
    Clean date format and remove time portion
    """
    try:
        if pd.isna(date_value) or date_value == "" or date_value == " ":
            return date_value

        # Convert to string if it's not already
        date_str = str(date_value).strip()

        # If it's already in the correct format, return as is
        if '/' in date_str and len(date_str.split('/')) == 3:
            parts = date_str.split('/')
            if len(parts[0]) == 2 and len(parts[1]) == 2 and len(parts[2]) == 4:
                return date_str

        # Remove time portion if present
        if ' 00:00:00' in date_str:
            date_str = date_str.replace(' 00:00:00', '')
        elif ' 0:00:00' in date_str:
            date_str = date_str.replace(' 0:00:00', '')
        elif ' 00:00' in date_str:
            date_str = date_str.replace(' 00:00', '')
        elif ' 12:00:00 am' in date_str:
            date_str = date_str.replace(' 12:00:00 am', '')

        # Handle Excel serial numbers - only if it's actually a number
        if date_str.replace('.', '').replace('-', '').isdigit():
            try:
                serial = float(date_str)
                if 30000 <= serial <= 50000:
                    base_date = datetime(1899, 12, 30)
                    date_result = base_date + timedelta(days=serial)
                    return date_result.strftime('%d/%m/%Y')
            except (ValueError, TypeError):
                pass  # Not actually a number, continue with date parsing

        # Try to parse common date formats
        date_formats = [
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%m/%d/%Y',
            '%Y/%m/%d',
            '%d-%m-%Y',
            '%m-%d-%Y'
        ]

        for fmt in date_formats:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                return date_obj.strftime('%d/%m/%Y')
            except ValueError:
                continue

        return date_str
    except Exception as e:
        print(f"Error processing date '{date_value}': {str(e)}")
        return date_value


def clean_date_column(df, date_column_name):
    """
    Clean a specific date column in the dataframe
    """
    try:
        if date_column_name in df.columns:
            print(f"  Cleaning date column: {date_column_name}")

            # Check if column has any non-null values
            if df[date_column_name].isna().all():
                print(f"  Warning: Column '{date_column_name}' is completely empty")
                return df

            # Apply date cleaning
            df[date_column_name] = df[date_column_name].apply(clean_date_format)

            # Show conversion results
            non_null_dates = df[date_column_name].dropna()
            if len(non_null_dates) > 0:
                print(f"  Sample converted dates: {non_null_dates.head(3).tolist()}")
        else:
            print(f"  Warning: Column '{date_column_name}' not found in dataframe")

        return df
    except Exception as e:
        print(f"  Error cleaning date column '{date_column_name}': {str(e)}")
        return df


def round_numbers(value):
    """
    Round numbers to 2 decimal places, leave other types unchanged
    """
    try:
        if isinstance(value, (int, float)) and not pd.isna(value):
            return round(float(value), 2)
        return value
    except:
        return value


def clean_currency_cells(df):
    """
    Clean currency symbols from all cells in the DataFrame and round numbers
    """
    try:
        currency_symbols = ['$', '£', '€', '¥', '₩', '₹', '₽', '₺', '₴', 'zł', 'kr', 'Fr']

        for column in df.columns:
            # Convert column name to string for safe comparison
            col_str = str(column)

            # Skip date columns - safer check without .lower() on potentially non-string columns
            if any(keyword in col_str for keyword in ['date', 'Date', 'DATE']):
                continue

            # For numeric columns, round to 2 decimal places
            if pd.api.types.is_numeric_dtype(df[column]):
                df[column] = df[column].apply(round_numbers)
            # For object/string columns, check if they contain numeric data
            elif df[column].dtype == 'object':
                # Sample the column to see if it contains mostly numeric data
                sample_data = df[column].dropna().head(10)
                if len(sample_data) > 0:
                    # Check if sample contains mostly numeric-looking data
                    numeric_count = 0
                    for val in sample_data:
                        str_val = str(val).strip()
                        # Remove currency symbols temporarily for checking
                        clean_val = ''.join(c for c in str_val if c not in currency_symbols)
                        try:
                            float(clean_val)
                            numeric_count += 1
                        except (ValueError, TypeError):
                            pass

                    # If most samples are numeric, clean the column
                    if numeric_count >= len(sample_data) * 0.8:  # 80% are numeric
                        df[column] = df[column].astype(str).apply(
                            lambda x: ''.join(char for char in str(x) if char not in currency_symbols)
                        )
                        # Try to convert to numeric
                        try:
                            df[column] = pd.to_numeric(df[column], errors='ignore')
                            if pd.api.types.is_numeric_dtype(df[column]):
                                df[column] = df[column].apply(round_numbers)
                        except:
                            pass

        return df
    except Exception as e:
        print(f"  Error in clean_currency_cells: {str(e)}")
        return df


def find_date_column(df):
    """
    Find the most likely date column in the dataframe
    """
    # Convert all column names to strings for comparison
    str_columns = [str(col) for col in df.columns]

    # Look for columns with 'date' in the name (case insensitive)
    for i, col_name in enumerate(str_columns):
        if 'date' in col_name.lower():
            return df.columns[i]

    # If no date column found, use the first column
    if len(df.columns) > 0:
        return df.columns[0]

    return None


def process_excel_files():
    """
    Process all Excel files, convert dates, round numbers, and clean currencies
    """
    excel_files = glob.glob("*.xlsx")

    if not excel_files:
        print("No Excel files found in current directory!")
        return

    print(f"Found {len(excel_files)} Excel files to process:")
    for file in excel_files:
        print(f"  - {file}")

    for excel_file in excel_files:
        try:
            print(f"\nProcessing: {excel_file}")

            # Read the Excel file
            df = pd.read_excel(excel_file)
            print(f"  Original shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")

            # Find date column
            date_column = find_date_column(df)
            if date_column is None:
                print("  No columns found in dataframe, skipping...")
                continue

            print(f"  Using date column: {date_column}")

            # Clean the date column
            df = clean_date_column(df, date_column)

            # Clean currency symbols and round numbers
            df_cleaned = clean_currency_cells(df)

            # Create CSV filename - overwrite existing CSV
            csv_filename = excel_file.replace('.xlsx', '.csv')

            # Save as CSV - overwrites existing file
            df_cleaned.to_csv(csv_filename, index=False)
            print(f"  Successfully converted to: {csv_filename}")

            # Show samples of the converted data
            print(f"  First few rows:")
            print(df_cleaned.head().to_string(index=False))

        except Exception as e:
            print(f"  ERROR processing {excel_file}: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\nConversion completed!")


def fix_existing_csv_files():
    """
    Fix existing CSV files that have the time format issue - overwrites original files
    """
    csv_files = glob.glob("*.csv")

    if not csv_files:
        print("No CSV files found in current directory!")
        return

    print(f"Found {len(csv_files)} CSV files to fix:")
    for file in csv_files:
        print(f"  - {file}")

    for csv_file in csv_files:
        try:
            print(f"\nFixing: {csv_file}")

            # Read the CSV file
            df = pd.read_csv(csv_file)
            print(f"  Original shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")

            # Find date column
            date_column = find_date_column(df)
            if date_column is None:
                print("  No columns found in dataframe, skipping...")
                continue

            print(f"  Using date column: {date_column}")

            # Clean the date column
            df = clean_date_column(df, date_column)

            # Overwrite the original CSV file
            df.to_csv(csv_file, index=False)
            print(f"  Successfully fixed: {csv_file}")

            # Show samples of the fixed data
            print(f"  First few rows after fixing:")
            print(df.head().to_string(index=False))

        except Exception as e:
            print(f"  ERROR fixing {csv_file}: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\nCSV file fixing completed!")


if __name__ == "__main__":
    print("Choose an option:")
    print("1. Convert Excel files to CSV")
    print("2. Fix existing CSV files")
    print("3. Both")

    choice = input("Enter your choice (1, 2, or 3): ").strip()

    if choice == "1":
        process_excel_files()
    elif choice == "2":
        fix_existing_csv_files()
    elif choice == "3":
        process_excel_files()
        fix_existing_csv_files()
    else:
        print("Invalid choice. Running both operations.")
        process_excel_files()
        fix_existing_csv_files()