import pandas as pd

# 1. Define your file path
input_file = 'output.csv'  # Replace with your actual file path
output_file = 'output_cleaned.csv'


# 2. Create a translation table for English digits to Devanagari
# This maps '0' -> '०', '1' -> '१', etc.
en_to_dev_map = str.maketrans('0123456789', '०१२३४५६७८९')

def clean_voter_field(value):
    """Removes all spaces and converts English digits to Devanagari."""
    if pd.isna(value):
        return value
    
    # Convert to string to handle any data type
    val_str = str(value)
    
    # Step 1: Remove all spaces
    val_str = val_str.replace(" ", "")
    
    # Step 2: Translate English digits to Devanagari digits
    val_str = val_str.translate(en_to_dev_map)
    
    return val_str

# 3. Load the data
df = pd.read_csv(input_file)

# 4. Apply cleaning to voter_id and picture columns
if 'voter_id' in df.columns:
    df['voter_id'] = df['voter_id'].apply(clean_voter_field)

if 'picture' in df.columns:
    df['picture'] = df['picture'].apply(clean_voter_field)

# 5. Save the cleaned file
# We use 'utf-8-sig' so the Nepali characters show correctly in Excel
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"Process complete. Cleaned file saved as: {output_file}")

# Optional: Verify the first few rows
df[['voter_id', 'picture']].head()