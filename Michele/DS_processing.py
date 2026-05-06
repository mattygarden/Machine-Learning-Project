import pandas as pd

def group_problem(text):
    # Convert to uppercase to catch "PLUMBING" and "Plumbing" identically
    text = str(text).upper()
    
    # 1. Noise Complaints
    if 'NOISE' in text: 
        return 'Noise'
        
    # 2. Vehicles and Parking
    elif 'PARKING' in text or 'VEHICLE' in text or 'DRIVEWAY' in text or 'TAXI' in text or 'FOR HIRE' in text: 
        return 'Vehicles & Parking'
        
    # 3. Housing and Indoor Building Issues
    elif 'HEAT' in text or 'PAINT' in text or 'DOOR' in text or 'FLOOR' in text or 'ELECTRIC' in text or 'APPLIANCE' in text or 'ELEVATOR' in text or 'MOLD' in text or 'BOILER' in text: 
        return 'Housing & Buildings'
        
    # 4. Water, Plumbing, and Sewers
    elif 'WATER' in text or 'SEWER' in text or 'PLUMBING' in text or 'LEAK' in text: 
        return 'Water & Plumbing'
        
    # 5. Street and Infrastructure
    elif 'STREET' in text or 'SIDEWALK' in text or 'HIGHWAY' in text or 'TRAFFIC' in text or 'CURB' in text or 'SIGN' in text or 'BRIDGE' in text: 
        return 'Street & Infrastructure'
        
    # 6. Sanitation and Garbage
    elif 'SANITARY' in text or 'DIRTY' in text or 'DUMPING' in text or 'COLLECTION' in text or 'DISPOSAL' in text or 'LITTER' in text or 'SWEEPING' in text: 
        return 'Sanitation & Garbage'
        
    # 7. Trees and Parks
    elif 'TREE' in text or 'PARK' in text or 'WOOD' in text or 'STUMP' in text or 'PLANT' in text: 
        return 'Trees & Parks'
        
    # 8. Animals and Pests
    elif 'ANIMAL' in text or 'DOG' in text or 'RODENT' in text or 'PIGEON' in text or 'BEE' in text: 
        return 'Animals & Pests'
        
    # 9. General / Everything Else
    else: 
        return 'Other'
    



def group_location(text):
    text = str(text).upper()
    
    # 1. Street and Sidewalks (Catches 'Curb', 'Highway', 'Crosswalk', etc.)
    if any(word in text for word in ['STREET', 'SIDEWALK', 'CURB', 'HIGHWAY', 'ROAD', 'INTERSECTION', 'CROSSWALK', 'LANE', 'OVERPASS', 'ISLAND']):
        return 'Street/Sidewalk'
        
    # 2. Residential (Catches 'Apartment', 'House', 'Dwelling', 'Residence')
    elif any(word in text for word in ['RESIDENT', 'HOUSE', 'APT', 'APARTMENT', 'DWELLING', 'LOFT', 'HOME']):
        return 'Residential'
        
    # 3. Commercial and Business (Catches 'Store', 'Restaurant', 'Bar', 'Vendor', etc.)
    elif any(word in text for word in ['COMMERCIAL', 'STORE', 'CLUB', 'BAR', 'RESTAURANT', 'BUSINESS', 'DELI', 'BAKERY', 'FOOD', 'VENDOR', 'CATERING', 'TATTOO', 'RETAIL', 'OFFICE', 'SHOP', 'COMERCIAL']):
        return 'Commercial/Business'
        
    # 4. Public Spaces and Transit (Catches 'Park', 'Subway', 'School', 'Bus', 'Ferry')
    elif any(word in text for word in ['PARK', 'PLAYGROUND', 'GARDEN', 'SUBWAY', 'TERMINAL', 'AIRPORT', 'BUS', 'SCHOOL', 'GOVERNMENT', 'FERRY', 'SHELTER', 'POOL', 'BRIDGE', 'STATION']):
        return 'Public/Transit'
        
    # 5. Everything Else (Catches 'Mixed Use', 'Lot', 'Alley', 'Other', NaN)
    else:
        return 'Other/Mixed'
    


def Process_train_DS(df):

    # Apply the function to create a brand new, clean column
    df['Problem_Grouped'] = df['Problem (formerly Complaint Type)'].apply(group_problem)

    # Apply the function to create the clean column
    df['Location_Grouped'] = df['Location Type'].apply(group_location)

    # Create a new column: 1 if it has text, 0 if it is missing (NaN)
    df['Is_Landmark'] = df['Landmark'].notna().astype(int)

    # Create a new column: 1 if it involves a TLC vehicle, 0 if it is missing (NaN)
    df['Is_Taxi'] = df['Vehicle Type'].notna().astype(int)


    # 1. Fill nulls with 'Unknown' safely
    df['Incident Zip'] = df['Incident Zip'].fillna('Unknown')

    # 2. Convert to string, then safely strip ONLY the trailing '.0' 
    df['Incident Zip'] = df['Incident Zip'].astype(str).str.replace(r'\.0$', '', regex=True)



    # The exact format from your screenshot: MM/DD/YYYY HH:MM:SS AM/PM
    date_format = '%m/%d/%Y %I:%M:%S %p'

    # Convert strings to actual pandas datetime objects
    df['Created Date'] = pd.to_datetime(df['Created Date'], format=date_format, errors='coerce')
    df['Closed Date'] = pd.to_datetime(df['Closed Date'], format=date_format, errors='coerce')

    # Drop any rows where the Created Date was corrupted or missing
    df = df.dropna(subset=['Created Date'])

    # Find the absolute latest date in the entire dataset (Data Dump Date)
    max_date = max(df['Created Date'].max(), df['Closed Date'].max())

    # Calculate how old every ticket was at the exact moment the data was pulled
    df['Age at Data Dump'] = max_date - df['Created Date']

    # Identify open tickets that are less than 24 hours old
    unknown_outcomes = df['Closed Date'].isna() & (df['Age at Data Dump'] <= pd.Timedelta(days=1))
    print(f"Dropped {unknown_outcomes.sum()} rows because their 24-hour window hasn't expired yet.")

    # Filter out those unknown rows
    df = df[~unknown_outcomes].copy()

    # Calculate time to close
    df['Time to Close'] = df['Closed Date'] - df['Created Date']

    # Create Y: 1 if closed within 24 hours, else 0
    df['Y'] = (df['Time to Close'] <= pd.Timedelta(days=1)).astype(int)

    

    # Extracting numerical time features from the Created Date
    df['Created_Hour'] = df['Created Date'].dt.hour           # Returns 0-23
    df['Created_DayOfWeek'] = df['Created Date'].dt.dayofweek # Returns 0-6 (0=Monday, 6=Sunday)
    df['Created_Month'] = df['Created Date'].dt.month         # Returns 1-12

    # You can even create custom binary features!
    # 1 if it's Saturday (5) or Sunday (6), else 0
    df['Is_Weekend'] = df['Created_DayOfWeek'].isin([5, 6]).astype(int)



    features_to_drop = [
    "Unique Key",
    "Created Date",
    "Closed Date",
    "Agency Name",
    "Problem (formerly Complaint Type)",
    "Problem Detail (formerly Descriptor)",
    "Additional Details",
    "Location Type",
    "Incident Address",
    "Street Name",
    "Cross Street 1",
    "Cross Street 2",
    "Intersection Street 1",
    "Intersection Street 2",
    "Address Type",
    "City",
    "Landmark",
    "Facility Type",
    "Community Board",
    "Council District",
    "BBL",
    "Park Facility Name",
    "Park Borough",
    "Vehicle Type",
    "Taxi Company Borough",
    "Taxi Pick Up Location",
    "Bridge Highway Name",
    "Bridge Highway Direction",
    "Bridge Highway Segment",
    "Road Ramp",
    "X Coordinate (State Plane)",
    "Y Coordinate (State Plane)",
    "Latitude",
    "Longitude",
    "Location",
    "Age at Data Dump",
    "Time to Close"
    ]

    df = df.drop(columns=features_to_drop, errors='ignore')

    # Remove 'Y' and immediately stick it on the end, just for easier visualization
    df['Y'] = df.pop('Y')

    return df




def Process_test_DS(df):

    # Apply the function to create a brand new, clean column
    df['Problem_Grouped'] = df['Problem (formerly Complaint Type)'].apply(group_problem)

    # Apply the function to create the clean column
    df['Location_Grouped'] = df['Location Type'].apply(group_location)

    # Create a new column: 1 if it has text, 0 if it is missing (NaN)
    df['Is_Landmark'] = df['Landmark'].notna().astype(int)

    # Create a new column: 1 if it involves a TLC vehicle, 0 if it is missing (NaN)
    df['Is_Taxi'] = df['Vehicle Type'].notna().astype(int)


    # 1. Fill nulls with 'Unknown' safely
    df['Incident Zip'] = df['Incident Zip'].fillna('Unknown')
    
    # 2. Convert to string, then safely strip ONLY the trailing '.0' 
    df['Incident Zip'] = df['Incident Zip'].astype(str).str.replace(r'\.0$', '', regex=True)



    # The exact format from your screenshot: MM/DD/YYYY HH:MM:SS AM/PM
    date_format = '%m/%d/%Y %I:%M:%S %p'

    # Convert strings to actual pandas datetime objects
    df['Created Date'] = pd.to_datetime(df['Created Date'], format=date_format, errors='coerce')

    # Drop any rows where the Created Date was corrupted or missing
    df = df.dropna(subset=['Created Date'])
    

    # Extracting numerical time features from the Created Date
    df['Created_Hour'] = df['Created Date'].dt.hour           # Returns 0-23
    df['Created_DayOfWeek'] = df['Created Date'].dt.dayofweek # Returns 0-6 (0=Monday, 6=Sunday)
    df['Created_Month'] = df['Created Date'].dt.month         # Returns 1-12

    # You can even create custom binary features!
    # 1 if it's Saturday (5) or Sunday (6), else 0
    df['Is_Weekend'] = df['Created_DayOfWeek'].isin([5, 6]).astype(int)



    features_to_drop = [
    "Unique Key",
    "Created Date",
    "Agency Name",
    "Problem (formerly Complaint Type)",
    "Problem Detail (formerly Descriptor)",
    "Additional Details",
    "Location Type",
    "Incident Address",
    "Street Name",
    "Cross Street 1",
    "Cross Street 2",
    "Intersection Street 1",
    "Intersection Street 2",
    "Address Type",
    "City",
    "Landmark",
    "Facility Type",
    "Community Board",
    "Council District",
    "BBL",
    "Park Facility Name",
    "Park Borough",
    "Vehicle Type",
    "Taxi Company Borough",
    "Taxi Pick Up Location",
    "Bridge Highway Name",
    "Bridge Highway Direction",
    "Bridge Highway Segment",
    "Road Ramp",
    "X Coordinate (State Plane)",
    "Y Coordinate (State Plane)",
    "Latitude",
    "Longitude",
    "Location",
    ]

    df = df.drop(columns=features_to_drop, errors='ignore')

    return df