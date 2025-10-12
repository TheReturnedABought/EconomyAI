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
        date_str = str(date_value)

        # Remove time portion if present (like "2000-01-04 00:00:00")
        if ' 00:00:00' in date_str:
            date_str = date_str.replace(' 00:00:00', '')
        elif ' 0:00:00' in date_str:
            date_str = date_str.replace(' 0:00:00', '')
        elif ' 00:00' in date_str:
            date_str = date_str.replace(' 00:00', '')
        elif ' 0:00' in date_str:
            date_str = date_str.replace(' 0:00', '')

        # Handle Excel serial numbers
        if date_str.replace('.', '').replace('-', '').isdigit():
            serial = float(date_str)
            if 30000 <= serial <= 50000:
                base_date = datetime(1899, 12, 30)
                date_result = base_date + timedelta(days=serial)
                return date_result.strftime('%d/%m/%Y')

        # Try to parse the remaining date string
        for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                return date_obj.strftime('%d/%m/%Y')
            except:
                continue

        return date_str
    except Exception as e:
        print(f"Error processing date {date_value}: {str(e)}")
        return date_value


def excel_serial_to_date(serial):
    """
    Convert Excel serial number to datetime object
    """
    try:
        if pd.isna(serial) or serial == "" or serial == " ":
            return serial

        # Convert to integer if it's a float with no decimal
        if isinstance(serial, float) and serial.is_integer():
            serial = int(serial)

        # Handle Excel serial numbers
        if isinstance(serial, (int, float)) and 30000 <= serial <= 50000:
            # Excel incorrectly considers 1900 as a leap year
            base_date = datetime(1899, 12, 30)
            date_result = base_date + timedelta(days=serial)
            return date_result.strftime('%d/%m/%Y')

        # If it's already a string, use the clean_date_format function
        if isinstance(serial, str):
            return clean_date_format(serial)

        return serial
    except:
        return serial


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
    currency_symbols = ['$', '£', '€', '¥', '₩', '₹', '₽', '₺', '₴', 'zł', 'kr', 'Fr']

    for column in df.columns:
        # First, round numeric columns to 2 decimal places
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].apply(round_numbers)
        # For string columns, remove currency symbols and try to convert to numbers
        elif df[column].dtype == 'object':
            df[column] = df[column].astype(str).apply(
                lambda x: ''.join(char for char in x if char not in currency_symbols)
            )
            # Try to convert to numeric after cleaning
            try:
                converted = pd.to_numeric(df[column], errors='ignore')
                # If successful conversion, round to 2 decimal places
                if pd.api.types.is_numeric_dtype(converted):
                    df[column] = converted.apply(round_numbers)
            except:
                pass

    return df


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

            # Convert first column to dates (assuming it's the date column)
            first_column = df.columns[0]
            print(f"  Converting date column: {first_column}")

            # Apply both conversion functions to ensure all dates are cleaned
            df[first_column] = df[first_column].apply(excel_serial_to_date)
            df[first_column] = df[first_column].apply(clean_date_format)

            # Clean currency symbols and round numbers
            df_cleaned = clean_currency_cells(df)

            # Create CSV filename
            csv_filename = excel_file.replace('.xlsx', '.csv')

            # Save as CSV
            df_cleaned.to_csv(csv_filename, index=False)
            print(f"  Successfully converted to: {csv_filename}")

            # Show samples of the converted data
            print(f"  First few rows:")
            print(df_cleaned.head().to_string(index=False))

        except Exception as e:
            print(f"  ERROR processing {excel_file}: {str(e)}")

    print("\nConversion completed!")


def fix_existing_csv_files():
    """
    Fix existing CSV files that have the time format issue
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

            # Clean the date column (assuming it's the first column)
            date_column = df.columns[0]
            print(f"  Cleaning date column: {date_column}")
            df[date_column] = df[date_column].apply(clean_date_format)

            # Save the fixed CSV (you can overwrite or create a new file)
            fixed_filename = csv_file.replace('.csv', '_fixed.csv')
            df.to_csv(fixed_filename, index=False)
            print(f"  Successfully fixed and saved as: {fixed_filename}")

            # Show samples of the fixed data
            print(f"  First few rows after fixing:")
            print(df.head().to_string(index=False))

        except Exception as e:
            print(f"  ERROR fixing {csv_file}: {str(e)}")

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