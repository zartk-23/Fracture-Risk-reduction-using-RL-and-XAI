#%%
import pandas as pd

# Load raw data with error handling
try:
    raw_data = pd.read_csv(r"C:\Users\zarth\Downloads\Bone_Health.csv")
    print("File loaded successfully.")
except PermissionError as e:
    print(f"PermissionError: {e}. Please close the file if it’s open or check permissions.")
    alt_path = input("Enter an alternative file path: ")
    try:
        raw_data = pd.read_csv(alt_path)
        print("File loaded from alternative path.")
    except Exception as e2:
        print(f"Error with alternative path: {e2}. Aborting.")
        exit()
except Exception as e:
    print(f"Unexpected error: {e}. Check the file path and try again.")
    exit()

# Select and convert relevant columns to strings to ensure consistent handling
data = raw_data[["Your Age? (e.g., 25, 60)", "What is your gender?",
                 "How many hours per week do you engage in physical activity? (e.g., walking, exercise, sports)",
                 "How many servings of calcium-rich foods (e.g., milk, yogurt, cheese) do you consume per week?",
                 "On an average, how many hours do you sleep per night?",
                 "How often do you stumble, lose balance, or have minor falls per month? (e.g., tripping but not falling fully)",
                 "Do you currently smoke?"]].astype(str)

# Clean and convert numerical columns
numeric_cols = ["Your Age? (e.g., 25, 60)", 
                "How many hours per week do you engage in physical activity? (e.g., walking, exercise, sports)",
                "How many servings of calcium-rich foods (e.g., milk, yogurt, cheese) do you consume per week?",
                "On an average, how many hours do you sleep per night?"]
for col in numeric_cols:
    data[col] = data[col].str.strip().replace(r'^\s*$', '0')  # Replace empty or whitespace-only with 0
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)  # Convert to numeric, fill NA with 0

# Convert MicroTrauma to numerical
def microtrauma_to_num(x):
    x = x.lower().strip()
    if "0-1" in x: return 0.5
    elif "2-3" in x: return 2.5
    elif "4 or more" in x: return 4.5
    return 0

data["How often do you stumble, lose balance, or have minor falls per month? (e.g., tripping but not falling fully)"] = data[
    "How often do you stumble, lose balance, or have minor falls per month? (e.g., tripping but not falling fully)"].apply(microtrauma_to_num)

# Convert Gender and Smoking to binary
data["What is your gender?"] = data["What is your gender?"].map({"male": 1, "female": 0, "Male": 1, "Female": 0}).fillna(0)
data["Do you currently smoke?"] = data["Do you currently smoke?"].map({"yes": 1, "no": 0, "Yes": 1, "No": 0}).fillna(0)

# Rename columns
data.columns = ["Age", "Gender", "Activity", "Calcium", "Sleep", "MicroTrauma", "Smoking"]

# Drop rows with any NA values (e.g., due to invalid entries) and verify data rows
data = data.dropna()
print(f"Number of data rows after cleaning: {len(data)}")  # Should be 120 or less if invalid

# Save to CSV
data.to_csv("bone_health.csv", index=False)
print(f"Saved {len(data)} rows to bone_health.csv")
# %%
