 import tmscoring
import argparse
import os
import csv
 
def compute_tm_score(pdb1, pdb2, output_file="tm_scores.csv"):
    """Compute the TM-score between two PDB structures and save results."""
    alignment = tmscoring.TMscoring(pdb1, pdb2)
    alignment.optimise()
    tm_score = alignment.tmscore(**alignment.get_current_values())
 
    print(f'TM-score ({os.path.basename(pdb1)} vs {os.path.basename(pdb2)}): {tm_score}')
 
    # Save results to CSV file
    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([os.path.basename(pdb1), os.path.basename(pdb2), tm_score])
 
    return tm_score
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute TM-score for all PDB file pairs")
    parser.add_argument("pdb_dir", help="Directory containing all PDB files")
    args = parser.parse_args()
 
    pdb_dir = args.pdb_dir
 
    # Get all PDB files in the directory and sort them
    pdb_files = sorted([f for f in os.listdir(pdb_dir) if f.endswith(".pdb")])
 
    output_file = "tm_scores_chr3_scl.csv"
 
    # Clear previous results
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["PDB1", "PDB2", "TM-score"])  # Write header
 
    # Loop through each pair
    for i in range(len(pdb_files)):  # Start from 1st PDB
        pdb1 = os.path.join(pdb_dir, pdb_files[i])
        for j in range(len(pdb_files)):  # Compare with all PDBs
            if i != j:  # Skip self-comparison
                pdb2 = os.path.join(pdb_dir, pdb_files[j])
                compute_tm_score(pdb1, pdb2, output_file)
 
