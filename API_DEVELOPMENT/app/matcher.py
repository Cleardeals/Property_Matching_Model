# matcher.py
import pandas as pd
import numpy as np
import json
import os
from sklearn.neighbors import NearestNeighbors
from keras.models import load_model
import joblib
from . import utils

class PropertyMatcher:
    def __init__(self, model_assets_dir='model_assets'):
        """
        Initializes the matcher by loading all necessary production artifacts.
        """
        print("--- 🧠 Initializing Property Matcher ---")
        try:
            self.encoder_model = load_model(os.path.join(model_assets_dir, 'property_encoder.h5'))
            self.preprocessor = joblib.load(os.path.join(model_assets_dir, 'preprocessor.joblib'))
            print("✅ Successfully loaded pre-trained model assets.")
        except Exception as e:
            print(f"❌ Critical error loading model assets: {e}")
            raise

    def create_and_save_mapping(self):
        """
        Uses the pre-trained model to generate a fresh mapping of active to expired properties.
        This is the main weekly update logic.
        """
        print("\n--- 🔄 Starting Weekly Matching Update ---")

        # --- Step 1: Load and process the latest data ---
        df_cleaned, _, _, _ = utils.run_data_pipeline()
        if df_cleaned is None:
            print("❌ Data pipeline failed. Aborting matching update.")
            return {"status": "error", "message": "Data pipeline failed."}

        # --- Step 2: Generate embeddings for the new data ---
        X_processed = self.preprocessor.transform(df_cleaned)
        all_embeddings = self.encoder_model.predict(X_processed)
        print(f"✅ Generated new embeddings for {len(df_cleaned)} properties.")

        # --- Step 3: Perform the Matching Logic ---
        query_mask = df_cleaned['Calculated_Status'] == 'Active'
        database_mask = df_cleaned['Calculated_Status'] == 'Expired'

        query_indices = df_cleaned[query_mask].index
        database_indices = df_cleaned[database_mask].index
        
        query_positions = df_cleaned.index.get_indexer(query_indices)
        database_positions = df_cleaned.index.get_indexer(database_indices)
        
        active_embeddings = all_embeddings[query_positions]
        database_embeddings = all_embeddings[database_positions]

        active_to_expired_mapping = {}
        
        if len(database_indices) > 0 and len(query_indices) > 0:
            K = 20
            min_similarity_threshold = 0.6
            max_distance_km = 5.0
            price_tolerance = 0.40
            max_matches = 10
            
            knn = NearestNeighbors(n_neighbors=min(K, len(database_embeddings)), metric='cosine')
            knn.fit(database_embeddings)

            print("\n--- Computing KNN for Active Properties ---")
            distances, indices = knn.kneighbors(active_embeddings)

            print("\n--- Creating Mapping for All Active Properties ---")
            for i, active_idx in enumerate(query_indices):
                active_tag = str(df_cleaned.loc[active_idx, 'Tag'])
                active_price = df_cleaned.loc[active_idx, 'Property_Price']
                active_lat = df_cleaned.loc[active_idx, 'Latitude']
                active_lon = df_cleaned.loc[active_idx, 'Longitude']
                active_location = df_cleaned.loc[active_idx, 'Location']
                active_property_type = df_cleaned.loc[active_idx, 'Property_Type'] if 'Property_Type' in df_cleaned.columns else None
                
                lower_bound = active_price * (1 - price_tolerance)
                upper_bound = active_price * (1 + price_tolerance)
                
                filtered_matches = []
                for j, candidate_idx in enumerate(indices[i]):
                    expired_position = candidate_idx
                    expired_idx = database_indices[expired_position]
                    expired_tag = str(df_cleaned.loc[expired_idx, 'Tag'])
                    expired_price = df_cleaned.loc[expired_idx, 'Property_Price']
                    expired_lat = df_cleaned.loc[expired_idx, 'Latitude']
                    expired_lon = df_cleaned.loc[expired_idx, 'Longitude']
                    
                    dist_km = utils.haversine(active_lat, active_lon, expired_lat, expired_lon)
                    similarity = 1 - distances[i][j]
                    
                    if (lower_bound <= expired_price <= upper_bound and 
                        dist_km <= max_distance_km and 
                        similarity >= min_similarity_threshold):
                        filtered_matches.append({
                            'expired_tag': expired_tag,
                            'similarity': float(similarity),
                            'distance_km': float(dist_km)
                        })
                
                filtered_matches.sort(key=lambda x: x['similarity'], reverse=True)
                active_to_expired_mapping[active_tag] = filtered_matches[:max_matches]

        # --- Step 4: Overwrite the mapping file ---
        with open('active_to_expired_mapping.json', 'w') as f:
            json.dump(active_to_expired_mapping, f, indent=4)
        print("\n✅ New 'active_to_expired_mapping.json' has been created.")
        return {"status": "success", "message": f"Mapping file updated with {len(active_to_expired_mapping)} active properties."}