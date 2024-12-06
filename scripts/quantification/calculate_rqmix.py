# scripts/quantification/calculate_rqmix.py

import pandas as pd
from pathlib import Path

def calculate_rqmix(sample_dir: Path, output_dir: Path) -> None:
    """
    Calculate RQmix values for all samples and save results.
    
    Args:
        sample_dir: Directory containing quantification results
        output_dir: Directory to save RQmix results
    """
    # Create output directory
    rqmix_dir = output_dir / "rqmix"
    rqmix_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize results dataframe
    results = []
    
    # Process each quantification file
    for quant_file in sample_dir.glob("*_quantification.csv"):
        try:
            # Get sample name
            sample_name = quant_file.stem.replace("_quantification", "")
            print(f"Processing {sample_name}")
            
            # Read quantification data
            df = pd.read_csv(quant_file)
            
            # Convert concentration from g/L to ug/L
            df['conc_ug_L'] = df['conc_M'] * 1e6
            
            # Calculate individual RQs
            df['RQ_daphnia'] = df['conc_ug_L'] / df['daphnia_LC50_48_hr_ug/L']
            df['RQ_algae'] = df['conc_ug_L'] / df['algae_EC50_72_hr_ug/L']
            df['RQ_pimephales'] = df['conc_ug_L'] / df['pimephales_LC50_96_hr_ug/L']
            
            # Replace inf and NA with 0
            df = df.replace([float('inf'), -float('inf')], 0)
            df = df.fillna(0)
            
            # Calculate RQmix for each endpoint
            rqmix_result = {
                'sample': sample_name,
                'RQmix_daphnia_LC50_48h': df['RQ_daphnia'].sum(),
                'RQmix_algae_EC50_72h': df['RQ_algae'].sum(),
                'RQmix_pimephales_LC50_96h': df['RQ_pimephales'].sum()
            }
            
            results.append(rqmix_result)
            
            # Save detailed results for this sample
            detailed_df = df[['identifier', 'conc_ug_L', 'RQ_daphnia', 'RQ_algae', 'RQ_pimephales']]
            detailed_df.to_csv(rqmix_dir / f"{sample_name}_rqmix_details.csv", index=False)
            
        except Exception as e:
            print(f"Error processing {quant_file.name}: {str(e)}")
    
    # Create and save final results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(rqmix_dir / "rqmix_summary.csv", index=False)
        print(f"✅ RQmix results saved in {rqmix_dir}")
    else:
        print("❌ No results were generated")

if __name__ == "__main__":
    # Définir les chemins
    sample_dir = Path("output/quantification/samples_quantification")
    output_dir = Path("output/quantification")
    
    # Calculer RQmix
    calculate_rqmix(sample_dir, output_dir)
