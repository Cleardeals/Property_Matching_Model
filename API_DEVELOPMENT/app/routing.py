# routing.py
import pandas as pd
import json
from collections import defaultdict
from google.cloud import bigquery
import datetime

def run_routing_process():
    """
    Loads low-scored leads, checks against historical assignments in BigQuery
    to prevent duplicates, routes new leads, and appends them to the BigQuery table.
    """
    print("--- 🚀 Starting Daily Routing for Low-Scored Leads ---")
    
    project_id = "cleardeals-459513"
    client = bigquery.Client(project=project_id, location= "US")

    try:
        # Load the master match file (from GCS in production)
        with open('active_to_expired_mapping.json', 'r') as f:
            active_to_expired_matches = json.load(f)
        
        # Query BigQuery for the scored leads
        leads_table_id = f"{project_id}.lead_scoring.daily_scored_leads_final"
        sql_query_leads = f"SELECT * FROM `{leads_table_id}`"
        daily_leads = client.query(sql_query_leads).to_dataframe()
        print(f"✅ Successfully loaded {len(daily_leads)} scored leads for today.")

        # --- THIS IS THE FIX ---
        # Explicitly set the location to match your dataset's location.
        
        # Corrected table name with underscore
        assignments_table_id = f"{project_id}.Property_Matching.Routed_Lead_Assignments"
        sql_query_assignments = f"SELECT assignment_id FROM `{assignments_table_id}`"
        
        # Check if the table exists and is not empty before querying
        try:
            existing_assignments_df = client.query(sql_query_assignments).to_dataframe()
            existing_assignments = set(existing_assignments_df['assignment_id'])
        except Exception:
            # If the table is empty or doesn't exist, start with an empty set
            existing_assignments = set()
            
        print(f"✅ Found {len(existing_assignments)} existing assignments to use for deduplication.")

        leads_table_id = f"{project_id}.lead_scoring.daily_scored_leads_final"
        sql_query_leads = f"SELECT * FROM `{leads_table_id}`"
        daily_leads = client.query(sql_query_leads).to_dataframe()
        print(f"✅ Successfully loaded {len(daily_leads)} scored leads for today.")

    except Exception as e:
        print(f"❌ Error during data loading: {e}")
        return {"status": "error", "message": str(e)}

    # --- Step 1: Filter for Low-Scoring Leads ---
    LOW_SCORE_THRESHOLD = 0.2
    low_scored_leads = daily_leads[daily_leads['predicted_score'] < LOW_SCORE_THRESHOLD].copy()
    print(f"✅ Filtered down to {len(low_scored_leads)} low-scoring leads (score < {LOW_SCORE_THRESHOLD}).")

    if low_scored_leads.empty:
        return {"status": "success", "message": "No low-scoring leads to process.", "routed_leads_count": 0}

    # --- NEW: Clean and Standardize Tags ---
    print("🧹 Standardizing property tags for matching...")
    
    # Clean the keys in the mapping file by removing the '_index' suffix
    cleaned_matches = {key.split('_')[0]: value for key, value in active_to_expired_matches.items()}
    
    # Clean the tags in the incoming leads data
    low_scored_leads['property_tag'] = low_scored_leads['property_tag'].apply(lambda tag: str(tag).split('_')[0])

    # --- Step 2: Route Leads ---
    new_routed_leads = []
    
    for _, lead in low_scored_leads.iterrows():
        active_prop_tag = lead['property_tag']
        lead_identifier = str(lead['standardized_phone'])
        lead_name = lead['customer_name']

        if active_prop_tag in cleaned_matches:
            candidate_expired_props = cleaned_matches[active_prop_tag]
            for match in candidate_expired_props:
                expired_tag = match['expired_tag'].split('_')[0] # Also clean the expired tag
                assignment_id = f"{lead_identifier}-{expired_tag}"
                
                if assignment_id not in existing_assignments:
                    new_routed_leads.append({
                        'assignment_id': assignment_id,
                        'lead_phone': lead_identifier,
                        'lead_name': lead_name,
                        'original_active_property_tag': active_prop_tag,
                        'routed_to_expired_property_tag': expired_tag,
                        'similarity_score': match['similarity'],
                        'assignment_timestamp': datetime.datetime.utcnow()
                    })
                    existing_assignments.add(assignment_id)

    # --- Step 3: Save New Assignments to BigQuery ---
    if new_routed_leads:
        routing_results_df = pd.DataFrame(new_routed_leads)
        
        print(f"\n✅ Lead routing complete. Generated {len(routing_results_df)} new notifications.")
        
        try:
            # --- THIS IS THE FIX ---
            # Use the native BigQuery client to upload the DataFrame
            job_config = bigquery.LoadJobConfig(
                # Specify that you want to append to the table if it exists
                write_disposition="WRITE_APPEND",
            )
            
            job = client.load_table_from_dataframe(
                routing_results_df, assignments_table_id, job_config=job_config
            )
            job.result()  # Wait for the job to complete
            
            print(f"✅ Successfully appended {len(routing_results_df)} new assignments to BigQuery.")
        except Exception as e:
            print(f"❌ Error writing to BigQuery: {e}")
            return {"status": "error", "message": f"Error writing to BigQuery: {e}"}
            
        return {
            "status": "success",
            "message": f"Successfully routed and saved {len(routing_results_df)} new leads.",
            "routed_leads_count": len(routing_results_df)
        }
    else:
        return {"status": "success", "message": "No new, unique leads were available to be routed.", "routed_leads_count": 0}
