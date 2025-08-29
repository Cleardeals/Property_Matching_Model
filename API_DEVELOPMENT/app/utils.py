# utils.py
import pandas as pd
import numpy as np
import math
import re
import json
import time
import os
from geopy.geocoders import Nominatim
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from google.cloud import bigquery

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on Earth using the Haversine formula.
    Parameters:
        lat1, lon1: Latitude and longitude of the first point (in degrees).
        lat2, lon2: Latitude and longitude of the second point (in degrees).
    Returns:
        Distance in kilometers.
    """
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371  # Earth's radius in km
    return c * R

def get_or_create_geo_mapping(df):
    """
    Loads geocoding from a file or creates it by calling an API.
    """
    json_path = '../geo_mapping.json'
    if os.path.exists(json_path):
        print(f"✅ Found existing geocoding file at '{json_path}'. Loading...")
        with open(json_path, 'r') as f:
            return json.load(f)

    print(f"⚠️ No geocoding file found. Creating a new one using geopy...")
    print("This will take several minutes as it respects API rate limits.")

    geolocator = Nominatim(user_agent="property_matcher_app_v6")
    unique_locations = df['Location'].dropna().unique()
    geo_mapping = {}

    city_prefix_map = {
        'A': 'Ahmedabad', 'P': 'Pune', 'G': 'Gandhinagar',
        'S': 'Surat', 'V': 'Vadodara', 'B': 'Vadodara'
    }

    for location_str in unique_locations:
        if not isinstance(location_str, str) or '-' not in location_str:
            continue

        parts = location_str.split('-', 1)
        prefix = parts[0].strip()
        area = parts[1].strip()
        city = city_prefix_map.get(prefix, 'Ahmedabad')

        query = f"{area}, {city}, India"
        try:
            location_data = geolocator.geocode(query, timeout=10)
            if location_data:
                geo_mapping[location_str] = (location_data.latitude, location_data.longitude)
                print(f"✅ Found: {query} -> ({location_data.latitude:.4f}, {location_data.longitude:.4f})")
            else:
                print(f"⚠️ Not Found: {query}")
            time.sleep(1)
        except Exception as e:
            print(f"❌ Error for '{query}': {e}")

    with open(json_path, 'w') as f:
        json.dump(geo_mapping, f, indent=4)
    print(f"✅ Geocoding complete. Saved to '{json_path}'.")
    return geo_mapping


def run_data_pipeline():
    """
    Loads, cleans, and preprocesses the property data, using manually created geo_mapping.json.
    Keeps original Location prefixes, trims whitespace, and ensures all locations are in geo_mapping.json.
    """
    # --- NEW: Query BigQuery using the native client ---
    print("✅ Connecting to BigQuery to fetch latest property data...")
    try:
        # Define your project and table ID
        project_id = "cleardeals-459513"
        table_id = f"{project_id}.cleardeals_dataset.Customer_Data"
        
        # 1. Create a BigQuery client
        client = bigquery.Client(project=project_id)
        
        # 2. Define your SQL query
        sql_query = f"SELECT * FROM `{table_id}`"
        
        # 3. Run the query and convert the result to a pandas DataFrame
        df = client.query(sql_query).to_dataframe()
        
        print(f"✅ Successfully loaded {len(df)} rows from BigQuery.")
    except Exception as e:
        print(f"❌ Error fetching data from BigQuery: {e}")
        return None, None, None, None   

    features_to_use = [
        'BHK', 'Property_Price', 'City1', 'Property_On_Floor', 'Property_Facing',
        'Age_Of_Property', 'Super_Built_up_Construction_Area', 'Carpet_Construction_Area',
        'Bathroom', 'Furniture_Details', 'Property_Status', 'Current_Status', 'Location',
        'Parking_Details', 'No_Of_Lift_Per_Block', 'Service_Expiry_Date', 'Tag',
        'Property_Type', 'Residential_Property', 'Commercial_Property_Type'
    ]
    missing_cols = [col for col in features_to_use if col not in df.columns]
    if missing_cols:
        print(f"⚠️ Warning: Columns {missing_cols} not found in PropertyData.csv. Proceeding without them.")
        features_to_use = [col for col in features_to_use if col in df.columns]
    df = df[features_to_use]
    print(f"✅ Selected {len(features_to_use)} features for modeling.")

    # Clean Location values
    print("\n🧹 Cleaning Location values...")
    df['Location'] = df['Location'].astype(str).str.strip()  # Trim whitespace
    invalid_locations = df[df['Location'] == '-']
    if not invalid_locations.empty:
        print(f"⚠️ Found {len(invalid_locations)} rows with Location = '-'. Setting to NaN.")
        df.loc[df['Location'] == '-', 'Location'] = np.nan

    # Clean Property_Type-related columns
    if 'Property_Type' in df.columns:
        print("\n🧹 Cleaning Property_Type, Residential_Property, Commercial_Property_Type...")
        if 'Commercial_Property_Type' in df.columns:
            df.loc[df['Property_Type'] == 'Residential', 'Commercial_Property_Type'] = np.nan
        if 'Residential_Property' in df.columns:
            df.loc[df['Property_Type'] == 'Commercial', 'Residential_Property'] = np.nan
        print("Inconsistent Residential properties with Commercial_Property_Type:")
        print(df[(df['Property_Type'] == 'Residential') & (df['Commercial_Property_Type'].notna())][['Tag', 'Property_Type', 'Residential_Property', 'Commercial_Property_Type']])
        print("Inconsistent Commercial properties with Residential_Property:")
        print(df[(df['Property_Type'] == 'Commercial') & (df['Residential_Property'].notna())][['Tag', 'Property_Type', 'Residential_Property', 'Commercial_Property_Type']])

    # Status Calculation
    print("\n🗓️ Calculating property status (Active/Expired/Sold)...")
    sold_statuses = ['Sold-CD', 'Sold-Others', 'Rented-CD']
    current_date = pd.to_datetime('2025-08-23')
    df['Service_Expiry_Date'] = pd.to_datetime(df['Service_Expiry_Date'], errors='coerce', dayfirst=True)
    
    print("Null Service_Expiry_Date count:", df['Service_Expiry_Date'].isna().sum())
    print("Sample Service_Expiry_Date values:")
    print(df[['Tag', 'Service_Expiry_Date']].head())

    conditions = [
        df['Property_Status'].isin(sold_statuses),
        df['Service_Expiry_Date'] < current_date
    ]
    choices = ['Sold', 'Expired']
    df['Calculated_Status'] = np.select(conditions, choices, default='Active')
    print(f"✅ Status calculation complete. Status distribution:")
    print(df['Calculated_Status'].value_counts())

    print("\nChecking for duplicate Tags...")
    if df['Tag'].duplicated().any():
        print(f"Found {df['Tag'].duplicated().sum()} duplicate Tags. Assigning unique Tags...")
        duplicate_tags = df[df['Tag'].duplicated(keep=False)]
        #duplicate_tags[['Tag', 'Calculated_Status', 'Property_Price', 'Location']].to_csv('duplicate_tags.csv')
        #print("Exported duplicate Tags to 'duplicate_tags.csv' for review.")
        df['Tag'] = df['Tag'].astype(str) + '_' + df.index.astype(str)
        print("Assigned unique Tags by appending index.")

    # Load manually created geo_mapping.json
    print("\n📍 Loading geo_mapping.json...")
    try:
        with open('geo_mapping.json', 'r') as f:
            geo_mapping = json.load(f)
    except FileNotFoundError:
        print("❌ Error: 'geo_mapping.json' not found.")
        return None, None, None, None

    # Validate locations
    valid_locations = df['Location'].astype(str).str.strip()
    missing_locations = set(valid_locations) - set(geo_mapping.keys()) - {'nan'}

    if missing_locations:
        # 2. Log the error/warning
        print(f"⚠️ WARNING: {len(missing_locations)} locations not in geo_mapping.json: {missing_locations}")
        print("These properties will be excluded from the current run.")
        
        # 3. Exclude the properties
        df = df[~df['Location'].isin(missing_locations)]
        print(f"✅ Excluded invalid properties. {len(df)} properties remaining for processing.")

    df[['Latitude', 'Longitude']] = df['Location'].apply(
        lambda x: pd.Series(geo_mapping.get(str(x).strip(), (np.nan, np.nan)) if str(x).strip() != 'nan' else (np.nan, np.nan))
    )
    print("Coordinate assignment complete. Sample coordinates:")
    print(df[['Location', 'Latitude', 'Longitude']].head())

    print("\n⚙️ Starting data cleaning process...")
    def clean_price(price):
        if not isinstance(price, str): return np.nan
        price_str = price.lower()
        try:
            numbers = re.findall(r'[\d\.]+', price_str)
            if not numbers: return np.nan
            value = float(numbers[0])
            if 'cr' in price_str: return value * 100
            return value
        except (ValueError, IndexError): return np.nan

    def clean_area(area):
        if not isinstance(area, str): return np.nan
        area_str = area.lower()
        try:
            numbers = re.findall(r'[\d\.]+', area_str)
            if not numbers: return np.nan
            value = float(numbers[0])
            if 'yard' in area_str: return value * 9
            return value
        except (ValueError, IndexError): return np.nan

    def clean_floor(floor):
        if not isinstance(floor, str): return np.nan
        floor_str = floor.lower().replace('g', '0')
        try:
            numbers = re.findall(r'\d+', floor_str)
            if numbers: return int(numbers[0])
            return np.nan
        except (ValueError, IndexError): return np.nan

    def clean_age(age):
        if not isinstance(age, str): return np.nan
        age_str = age.lower()
        if 'new' in age_str or 'under' in age_str: return 0
        try:
            numbers = [int(s) for s in re.findall(r'\d+', age_str)]
            if numbers: return sum(numbers) / len(numbers)
            return np.nan
        except (ValueError, IndexError): return np.nan

    df.replace('-', np.nan, inplace=True)
    df.replace('NA', np.nan, inplace=True)
    df['Property_Price'] = df['Property_Price'].apply(clean_price)
    df['Super_Built_up_Construction_Area'] = df['Super_Built_up_Construction_Area'].apply(clean_area)
    df['Carpet_Construction_Area'] = df['Carpet_Construction_Area'].apply(clean_area)
    df['Property_On_Floor'] = df['Property_On_Floor'].apply(clean_floor)
    df['Age_Of_Property'] = df['Age_Of_Property'].apply(clean_age)
    df['BHK'] = pd.to_numeric(df['BHK'].astype(str).str.extract(r'(\d+)', expand=False), errors='coerce')
    df['Bathroom'] = pd.to_numeric(df['Bathroom'], errors='coerce')
    df['No_Of_Lift_Per_Block'] = pd.to_numeric(df['No_Of_Lift_Per_Block'], errors='coerce')
    df.loc[df['Bathroom'] > 20, 'Bathroom'] = np.nan

    df['Price_Per_SqFt'] = df['Property_Price'] / df['Carpet_Construction_Area'].clip(lower=1)

    numerical_cols = ['Property_Price', 'Super_Built_up_Construction_Area', 'Carpet_Construction_Area',
                      'Property_On_Floor', 'Age_Of_Property', 'BHK', 'Bathroom', 'No_Of_Lift_Per_Block',
                      'Latitude', 'Longitude', 'Price_Per_SqFt']
    for col in numerical_cols:
        df[col].fillna(df[col].median(), inplace=True)

    categorical_cols = ['City1', 'Property_Facing', 'Furniture_Details', 'Property_Status',
                       'Current_Status', 'Parking_Details', 'Calculated_Status',
                       'Property_Type', 'Residential_Property', 'Commercial_Property_Type']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col].replace('nan', 'None', inplace=True)
            df[col].fillna('None', inplace=True)

    print("✅ Data cleaning and imputation complete.")
    print("Property_Type distribution:")
    print(df['Property_Type'].value_counts())
    print("Residential_Property distribution:")
    print(df['Residential_Property'].value_counts())
    print("Commercial_Property_Type distribution:")
    print(df['Commercial_Property_Type'].value_counts())

    numerical_features = [col for col in numerical_cols if col in df.columns]
    categorical_features = [col for col in categorical_cols if col != 'Calculated_Status' and col in df.columns]
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )
    
    return df, numerical_cols, categorical_cols, preprocessor