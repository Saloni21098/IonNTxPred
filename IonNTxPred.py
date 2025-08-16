#########################################################################
# IonNTxpred is developed for identifying modulator and non-modulator  #
# peptides that selectively modulate sodium, potassium, calcium, and    #
# other ion channels. It is developed by Prof G. P. S. Raghava's group. #                                                  
# Please cite : IonNTxpred                                             #
#########################################################################
def main():
    ## Import libraries
    import argparse  
    import warnings
    import os
    import re
    import numpy as np
    import pandas as pd
    from transformers import AutoTokenizer, EsmForSequenceClassification, EsmModel
    import torch
    from torch.utils.data import DataLoader, Dataset
    import joblib
    import shutil
    import subprocess
    from collections import defaultdict
    from Bio import SeqIO
    import uuid
    import zipfile
    import urllib.request
    from tqdm.auto import tqdm
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description='Please provide following arguments. Please make the suitable changes in the envfile provided in the folder.')

    ## Read Arguments from command
    parser.add_argument("-i", "--input", type=str, required=True, help="Input: Peptide sequence in FASTA format or single sequence per line in single letter code")
    parser.add_argument("-o", "--output",type=str, default="output.csv", help="Output: File for saving results by default output.csv")
    parser.add_argument("-t","--threshold", type=float, default=0.5, help="Threshold: Value between 0 to 1 by default 0.5")
    parser.add_argument("-j", "--job", type=int, choices=[1, 2, 3, 4, 5], default=1, help="Job Type: 1: Prediction, 2: Design, 3: Protein Scanning, 4: Motif Scanning, 5: Blast Search")
    parser.add_argument("-c", "--channel", type=int, default=None, choices=[1, 2, 3, 4], help="Ion channel type: 1: Na+, 2: K+, 3: Ca+, 4: Other")
    parser.add_argument("-m", "--model", type=int, default= 1, choices=[1, 2], help="Model: 1: ESM2-t12, 2: Hybrid (ESM2-t12 + MERCI)")
    parser.add_argument("-w","--winleng", type=int, choices =range(8, 21), default=8, help="Window Length: 8 to 20 (scan mode only), by default 8")
    parser.add_argument("-wd", "--working", type=str, default=os.getcwd(), help="Working directory for intermediate files (optional).")
    parser.add_argument("-d","--display", type=int, choices = [1,2], default=2, help="Display: 1:Modulating ion, 2: All peptides, by default 2")


    args = parser.parse_args()

    nf_path = os.path.dirname(__file__)

    ################################### Model Calling ##########################################

    # Get the absolute path of the script
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(SCRIPT_DIR, "Model")
    ZIP_PATH = os.path.join(SCRIPT_DIR, "Model.zip")
    MODEL_URL = "https://webs.iiitd.edu.in/raghava/ionntxpred/download/Model.zip"


    # Check if the Model folder exists
    if not os.path.exists(MODEL_DIR):
        print('##############################')
        print("Downloading the model files...")
        print('##############################')

        try:
            # Download the ZIP file with the progress bar
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading") as t:
                urllib.request.urlretrieve(MODEL_URL, ZIP_PATH, reporthook=lambda block_num, block_size, total_size: t.update(block_size))

            print("Download complete. Extracting files...")

            # Extract the ZIP file
            with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
                zip_ref.extractall(SCRIPT_DIR)

            print("Extraction complete. Removing ZIP file...")

            # Remove the ZIP file after extraction
            os.remove(ZIP_PATH)
            print("Model setup completed successfully.")

        except urllib.error.URLError as e:
            print(f"Network error: {e}. Please check your internet connection.")
        except zipfile.BadZipFile:
            print("Error: The downloaded file is corrupted. Please try again.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    else:
        print('#################################################################')
        print("Model folder already exists. Skipping download.")
        print('#################################################################')
    
    # Function to check the sequence residue
    def readseq(file):
        with open(file) as f:
            records = f.read()
        records = records.split('>')[1:]
        seqid = []
        seq = []
        non_standard_detected = False  # Flag to track non-standard amino acids

        for fasta in records:
            array = fasta.split('\n')
            name, sequence = array[0].split()[0], ''.join(array[1:]).upper()
            
            # Check for non-standard amino acids
            filtered_sequence = re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '', sequence)
            if filtered_sequence != sequence:
                non_standard_detected = True
            
            seqid.append('' + name)
            seq.append(filtered_sequence)
        
        if len(seqid) == 0:
            f = open(file, "r")
            data1 = f.readlines()
            for each in data1:
                seq.append(each.replace('\n', ''))
            for i in range(1, len(seq) + 1):
                seqid.append("Seq_" + str(i))
        
        if non_standard_detected:
            print("Non-standard amino acids were detected. Processed sequences have been saved and used for further prediction.")
        else:
            print("No non-standard amino acids were detected.")
        
        df1 = pd.DataFrame(seqid)
        df2 = pd.DataFrame(seq)
        return df1, df2


    # Function to check the length of sequences and suggest a model
    def lenchk(file1):
        cc = []
        df1 = file1
        df1.columns = ['seq']
        
        # Analyze sequence lengths
        for seq in df1['seq']:
            cc.append(len(seq))
        
        # Check if any sequences are shorter than 7
        if any(length < 7 for length in cc):
            raise ValueError("Sequences with length < 7 detected. Please ensure all sequences have length at least 7. Prediction process stopped.")
        
        return df1


    # ESM2
    # Define a function to process sequences

    def process_sequences(df, df_2):
        df = pd.DataFrame(df, columns=['seq'])  # Assuming 'seq' is the column name
        df_2 = pd.DataFrame(df_2, columns=['SeqID'])
        # Process the sequences
        outputs = [(df_2.loc[index, 'SeqID'], row['seq']) for index, row in df.iterrows()]
        return outputs


    # Function to prepare dataset for prediction
    def prepare_dataset(sequences, tokenizer):
        seqs = [seq for _, seq in sequences]
        inputs = tokenizer(seqs, padding=True, truncation=True, return_tensors="pt")
        return inputs


    # Function to write output to a file
    def write_output(output_file, sequences, predictions, Threshold):
        with open(output_file, 'w') as f:
            f.write("SeqID,Seq,ESM Score,Prediction\n")
            for (seq_id, seq), pred in zip(sequences, predictions):
                clean_seq_id = str(seq_id).lstrip(">")  # Remove '>' if present
                final_pred = "Modulator" if pred >= Threshold else "Non-modulator"
                f.write(f"{clean_seq_id},{seq},{pred:.4f},{final_pred}\n")


    # Function to make predictions
    def make_predictions(model, inputs, device):
        # Move the model to the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        return probs


    # Main function for ESM model integration
    def run_esm_model(dfseq , df_2, output_file, Threshold):
        # Process sequences from the DataFrame
        sequences = process_sequences(dfseq, df_2)

        # Move the model to the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Prepare inputs for the model
        inputs = prepare_dataset(sequences, tokenizer)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Make predictions
        predictions = make_predictions(model, inputs, device)

        # Write the output to a file
        write_output(output_file, sequences, predictions, Threshold)


    # Function for generating pattern of a given length (protein scanning)
    def seq_pattern(file1, file2, num):
        df1 = pd.DataFrame(file1, columns=['Seq'])
        df2 = pd.DataFrame(file2, columns=['Name'])

        # Check input lengths
        if df1.empty or df2.empty:
            print("[ERROR] One of the input lists is empty.")
            return pd.DataFrame()

        if len(df1) != len(df2):
            print("[ERROR] Mismatched number of sequences and sequence IDs.")
            print(f"Sequences: {len(df1)}, IDs: {len(df2)}")
            return pd.DataFrame()

        cc, dd, ee, ff, gg = [], [], [], [], []

        for i in range(len(df1)):
            sequence = df1['Seq'][i]
            if not isinstance(sequence, str):
                print(f"[WARNING] Sequence at index {i} is not a string: {sequence}")
                continue

            for j in range(len(sequence)):
                xx = sequence[j:j+num]
                if len(xx) == num:
                    cc.append(df2['Name'][i])
                    dd.append('Pattern_' + str(j + 1))
                    ee.append(xx)
                    ff.append(j + 1)  # Start position (1-based)
                    gg.append(j + num)  # End position (1-based)

        if not cc:  # Check if any patterns were generated
            print(f"[WARNING] No patterns generated. Possibly all sequences are shorter than {num} residues.")
            return pd.DataFrame()

        df3 = pd.DataFrame({
            'SeqID': cc,
            'Pattern ID': dd,
            'Start': ff,
            'End': gg,
            'Seq': ee
        })

        return df3


    def generate_mutant(original_seq, residues, position):
        std = "ACDEFGHIKLMNPQRSTVWY"
        if all(residue.upper() in std for residue in residues):
            if len(residues) == 1:
                mutated_seq = original_seq[:position-1] + residues.upper() + original_seq[position:]
            elif len(residues) == 2:
                mutated_seq = original_seq[:position-1] + residues[0].upper() + residues[1].upper() + original_seq[position+1:]
            else:
                print("Invalid residues. Please enter one or two of the 20 essential amino acids.")
                return None
        else:
            print("Invalid residues. Please enter one or two of the 20 essential amino acids.")
            return None
        return mutated_seq


    def generate_mutants_from_dataframe(df, residues, position):
        mutants = []
        for index, row in df.iterrows():
            original_seq = row['seq']
            mutant_seq = generate_mutant(original_seq, residues, position)
            if mutant_seq:
                mutants.append((original_seq, mutant_seq, position))
        return mutants

    # Function for generating all possible mutants
    def all_mutants(file1,file2):
        std = list("ACDEFGHIKLMNPQRSTVWY")
        cc = []
        dd = []
        ee = []
        df2 = pd.DataFrame(file2)
        df2.columns = ['Name']
        df1 = pd.DataFrame(file1)
        df1.columns = ['Seq']
        for k in range(len(df1)):
            cc.append(df1['Seq'][k])
            dd.append('Original_'+'Seq'+str(k+1))
            ee.append(df2['Name'][k])
            for i in range(0,len(df1['Seq'][k])):
                for j in std:
                    if df1['Seq'][k][i]!=j:
                        #dd.append('Mutant_'+df1['Seq'][k][i]+str(i+1)+j+'_Seq'+str(k+1))
                        dd.append('Mutant_'+df1['Seq'][k][i]+str(i+1)+j)
                        cc.append(df1['Seq'][k][:i] + j + df1['Seq'][k][i + 1:])
                        ee.append(df2['Name'][k])
        xx = pd.concat([pd.DataFrame(ee),pd.DataFrame(dd),pd.DataFrame(cc)],axis=1)
        xx.columns = ['SeqID','Mutant_ID','Seq']
        return xx


    def run_blast_and_get_scores(fasta_input, output_dir, blast_bin_path, blast_db_path):
        os.makedirs(output_dir, exist_ok=True)
        blast_output = os.path.join(output_dir, "blast_results.txt")

        cmd = f"{blast_bin_path} -query {fasta_input} -db {blast_db_path} -out {blast_output} -evalue 1e-3 -outfmt 6"
        print(f"[INFO] Running BLASTP on DB: {blast_db_path}")

        result = os.system(cmd)
        if result != 0:
            print(f"[ERROR] BLASTP failed. Check binary path and DB setup.")
            return {}

        if not os.path.exists(blast_output) or os.path.getsize(blast_output) == 0:
            print(f"[ERROR] No output from BLAST.")
            return {}

        # Read BLAST output
        blast_columns = ['name', 'hit', 'identity', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9']
        blast_df = pd.read_csv(blast_output, sep="\t", names=blast_columns)
        print(blast_df)

        if blast_df.empty:
            return {}

        headers, sequences = [], []
        for record in SeqIO.parse(fasta_input, "fasta"):
            headers.append(record.id)
            sequences.append(str(record.seq))
        fasta_df = pd.DataFrame({'name': headers, 'Sequence': sequences})
        blast_df['name'] = blast_df['name'].astype(str)
        fasta_df['name'] = fasta_df['name'].astype(str)

        blast_scores = {}
        for name in fasta_df['name']:
            match = blast_df[blast_df['name'] == name]
            if not match.empty:
                hit_value = match['hit'].iloc[0]
                if hit_value.startswith("Pseq_"):
                    blast_scores[name] = 0.5
                elif hit_value.startswith("Nontoxic_"):
                    blast_scores[name] = -0.5
                else:
                    blast_scores[name] = 0.0  # Unknown hit type
            else:
                blast_scores[name] = 0.0  # No hit found

        return blast_scores


    # Function of MERCI
    def MERCI_Processor_p(merci_file,merci_processed,name):
        hh =[]
        jj = []
        kk = []
        qq = []
        filename = merci_file
        df = pd.DataFrame(name)
        zz = list(df[0])
        check = '>'
        with open(filename) as f:
            l = []
            for line in f:
                if not len(line.strip()) == 0 :
                    l.append(line)
                if 'COVERAGE' in line:
                    for item in l:
                        if item.lower().startswith(check.lower()):
                            hh.append(item)
                    l = []
        if hh == []:
            ff = [w.replace('>', '') for w in zz]
            for a in ff:
                jj.append(a)
                qq.append(np.array(['0']))
                kk.append('Non-modulator')
        else:
            ff = [w.replace('\n', '') for w in hh]
            ee = [w.replace('>', '') for w in ff]
            rr = [w.replace('>', '') for w in zz]
            ff = ee + rr
            oo = np.unique(ff)
            df1 = pd.DataFrame(list(map(lambda x:x.strip(),l))[1:])
            df1.columns = ['Name']
            df1['Name'] = df1['Name'].str.strip('(')
            df1[['Seq','Hits']] = df1.Name.str.split("(",expand=True)
            df2 = df1[['Seq','Hits']]
            df2.replace(to_replace=r"\)", value='', regex=True, inplace=True)
            df2.replace(to_replace=r'motifs match', value='', regex=True, inplace=True)
            df2.replace(to_replace=r' $', value='', regex=True,inplace=True)
            total_hit = int(df2.loc[len(df2)-1]['Seq'].split()[0])
            for j in oo:
                if j in df2.Seq.values:
                    jj.append(j)
                    qq.append(df2.loc[df2.Seq == j]['Hits'].values)
                    kk.append('Modulator')
                else:
                    jj.append(j)
                    qq.append(np.array(['0']))
                    kk.append('Non-modulator')
        df3 = pd.concat([pd.DataFrame(jj),pd.DataFrame(qq),pd.DataFrame(kk)], axis=1)
        df3.columns = ['Name','Hits','Prediction']
        df3.to_csv(merci_processed,index=None)


    def Merci_after_processing_p(merci_processed,final_merci_p):
        df5 = pd.read_csv(merci_processed)
        df5 = df5[['Name','Hits']]
        df5.columns = ['Subject','Hits']
        kk = []
        for i in range(0,len(df5)):
            if df5['Hits'][i] > 0:
                kk.append(0.5)
            else:
                kk.append(0)
        df5["MERCI Score (+ve)"] = kk
        df5 = df5[['Subject','MERCI Score (+ve)']]
        df5.to_csv(final_merci_p, index=None)


    def MERCI_Processor_n(merci_file,merci_processed,name):
        hh =[]
        jj = []
        kk = []
        qq = []
        filename = merci_file
        df = pd.DataFrame(name)
        zz = list(df[0])
        check = '>'
        with open(filename) as f:
            l = []
            for line in f:
                if not len(line.strip()) == 0 :
                    l.append(line)
                if 'COVERAGE' in line:
                    for item in l:
                        if item.lower().startswith(check.lower()):
                            hh.append(item)
                    l = []
        if hh == []:
            ff = [w.replace('>', '') for w in zz]
            for a in ff:
                jj.append(a)
                qq.append(np.array(['0']))
                kk.append('Non-modulator')
        else:
            ff = [w.replace('\n', '') for w in hh]
            ee = [w.replace('>', '') for w in ff]
            rr = [w.replace('>', '') for w in zz]
            ff = ee + rr
            oo = np.unique(ff)
            df1 = pd.DataFrame(list(map(lambda x:x.strip(),l))[1:])
            df1.columns = ['Name']
            df1['Name'] = df1['Name'].str.strip('(')
            df1[['Seq','Hits']] = df1.Name.str.split("(",expand=True)
            df2 = df1[['Seq','Hits']]
            df2.replace(to_replace=r"\)", value='', regex=True, inplace=True)
            df2.replace(to_replace=r'motifs match', value='', regex=True, inplace=True)
            df2.replace(to_replace=r' $', value='', regex=True,inplace=True)
            total_hit = int(df2.loc[len(df2)-1]['Seq'].split()[0])
            for j in oo:
                if j in df2.Seq.values:
                    jj.append(j)
                    qq.append(df2.loc[df2.Seq == j]['Hits'].values)
                    kk.append('Modulator')
                else:
                    jj.append(j)
                    qq.append(np.array(['0']))
                    kk.append('Non-modulator')
        df3 = pd.concat([pd.DataFrame(jj),pd.DataFrame(qq),pd.DataFrame(kk)], axis=1)
        df3.columns = ['Name','Hits','Prediction']
        df3.to_csv(merci_processed,index=None)


    def Merci_after_processing_n(merci_processed,final_merci_n):
        df5 = pd.read_csv(merci_processed)
        df5 = df5[['Name','Hits']]
        df5.columns = ['Subject','Hits']
        kk = []
        for i in range(0,len(df5)):
            if df5['Hits'][i] > 0:
                kk.append(-0.5)
            else:
                kk.append(0)
        df5["MERCI Score (-ve)"] = kk
        df5 = df5[['Subject','MERCI Score (-ve)']]
        df5.to_csv(final_merci_n, index=None)


    def BLAST_search(blast_result, name1):
        # Force single-column DataFrame
        name1.columns = [0]

        # If BLAST file has results
        if os.path.exists(blast_result) and os.stat(blast_result).st_size != 0:
            # Read BLAST output
            df1 = pd.read_csv(
                blast_result, sep="\t",
                names=['name', 'hit', 'identity', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9'],
                dtype=str
            )

            # Prepare sequence list
            df2 = pd.DataFrame()
            df2 = pd.concat([df2, name1])
            cc = []

            for i in df2[0]:
                seq_id = i.replace('>', '')
                match = df1[df1['name'] == seq_id]

                if not match.empty:
                    hit_value = match['hit'].iloc[0]
                    if hit_value.startswith("Pseq_"):
                        cc.append(0.5)
                    elif hit_value.startswith("Nontoxic_"):
                        cc.append(-0.5)
                    else:
                        cc.append(0.0)  # Unknown
                else:
                    cc.append(0.0)  # No hit
        else:
            # No BLAST output — all scores zero
            df2 = pd.concat([pd.DataFrame(), name1])
            cc = [0.0] * len(df2)

        # Create final DataFrame
        df6 = pd.DataFrame()
        df6['Seq ID'] = [i.replace('>', '') for i in df2.iloc[:, 0]]
        df6['BLAST Score'] = cc
        df6['Prediction'] = ["Modulator" if s > 0 else "Non-Modulator" for s in cc]

        return df6


    def write_fasta(df, filename):
        """
        Write sequences to a FASTA file.
        Expects df to have 'Seq ID' and 'Seq' columns.
        """
        with open(filename, "w") as f:
            for _, row in df.iterrows():
                f.write(f">{row['SeqID']}\n")
                f.write(f"{row['Seq']}\n")


    print('\n')
    print('##########################################################################################')
    print('#                                                                                        #')
    print('#  🧠 IonNTxPred: A Powerful Tool to Predict Ion Channel Modulators                      #')
    print('#                                                                                        #')
    print('#  This program is developed to identify **modulators** and **non-modulators** of        #')
    print('#  ion channels—specifically targeting sodium (Na⁺), potassium (K⁺), calcium (Ca²⁺),     #')
    print('#  and other channels.                                                          #')
    print('#                                                                                        #')
    print("#  🧬 Developed by Prof. G. P. S. Raghava's group at IIIT-Delhi                           #")
    print('#                                                                                        #')
    print('##########################################################################################')



    # Parameter initialization or assigning variable for command level arguments

    Sequence= args.input        # Input variable 
    
    # Output file 
    if args.output is None:
        result_filename = "output.csv"
    else:
        result_filename = args.output
            
    # Threshold 
    if args.threshold is None:
        Threshold = 0.3
    else:
        Threshold= float(args.threshold)

    # Job Type
    if args.job is None:
        Job = 1
    else:
        Job = args.job

    # Model
    if args.model is None:
        Model = 1
    else:
        Model = int(args.model)

    # Channel Type
    if args.channel is None:
        if Job == 1:
            Channel = None    # Not needed for prediction
        else:
            Channel = 1       # Default to 1 if required and not provided
    else:
        Channel = args.channel

    if Job != 1 and Channel is None:
        parser.error("--channel is required for job types 2, 3, and 4")


    # Display
    if args.display is None:
        dplay = 2
    else:
        dplay = int(args.display)

    # Window Length 
    if args.winleng == None:
        Win_len = int(8)
    else:
        Win_len = int(args.winleng)


    # Working Directory
    wd = args.working

    print('\nSummary of Parameters:')
    print('Input File: ', Sequence, '; Model: ', Model, '; Channel: ', Channel, '; Job: ', Job, '; Threshold: ', Threshold)
    print('Output File: ',result_filename,'; Display: ',dplay)

    #------------------ Read input file ---------------------
    f=open(Sequence,"r")
    len1 = f.read().count('>')
    f.close()

    with open(Sequence) as f:
            records = f.read()
    records = records.split('>')[1:]
    seqid = []
    seq = []
    for fasta in records:
        array = fasta.split('\n')
        name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '', ''.join(array[1:]).upper())
        seqid.append(name)
        seq.append(sequence)
    if len(seqid) == 0:
        f=open(Sequence,"r")
        data1 = f.readlines()
        for each in data1:
            seq.append(each.replace('\n',''))
        for i in range (1,len(seq)+1):
            seqid.append("Seq_"+str(i))

    seqid_1 = list(map(">{}".format, seqid))
    CM = pd.concat([pd.DataFrame(seqid_1),pd.DataFrame(seq)],axis=1)
    CM.to_csv(f"{wd}/Sequence_1", header=False, index=False)

    # CM.to_csv(f"{wd}/Sequence_1",header=False,index=None,sep="\n")
    f.close()

                
                #======================= Prediction Module starts from here =====================
    if Job == 1:
        print(f'\n======= You are using the Prediction Module of IonNTxPred. Your results will be stored in file: {wd}/{result_filename}\n')
        print("\n==================== Running Prediction Module ====================")

        channel_configs = {
            "na": {
                "model_path": f"{nf_path}/Model/saved_model_t33_na",
                "label": "Na"
            },
            "k": {
                "model_path": f"{nf_path}/Model/saved_model_t33_k",
                "label": "K"
            },
            "ca": {
                "model_path": f"{nf_path}/Model/saved_model_t33_ca",
                "label": "Ca"
            },
            "other": {
                "model_path": f"{nf_path}/Model/saved_model_t33_other",
                "label": "Other"
            }
        }

        with open(f"{wd}/Sequence_1", "w") as f:
            for s_id, s in zip(seqid_1, seq):
                f.write(f"{s_id}\n{s}\n")

        channel_results = []

        for channel, cfg in channel_configs.items():
            print(f"=== Processing Channel: {cfg['label']} ===")
            model_save_path = cfg["model_path"]

            tokenizer = AutoTokenizer.from_pretrained(model_save_path)
            model = EsmForSequenceClassification.from_pretrained(model_save_path)
            model.eval()

            if Model == 1:
                out_file1 = f"{wd}/model1_output_{channel}.csv"
                run_esm_model(seq, seqid_1, out_file1, Threshold)

                df_model1 = pd.read_csv(out_file1)
                df_model1.rename(columns={"ML Score": cfg["label"]}, inplace=True)
                df_model1.columns = ['SeqID', 'Seq', cfg["label"], f"{cfg['label']}_Prediction"]
                df_model1['SeqID'] = df_model1['SeqID'].astype(str).str.replace('>', '')
                df_final = df_model1

                # Clean up temporary file
                os.remove(out_file1)

            elif Model == 2:
                out_file2 = f"{wd}/model2_output_{channel}.csv"
                run_esm_model(seq, seqid_1, out_file2, Threshold)
                df_esm2 = pd.read_csv(out_file2)
                hybrid_col = f"{cfg['label']}_hybrid"
                df_esm2.rename(columns={df_esm2.columns[2]: hybrid_col}, inplace=True)
                df_esm2['SeqID'] = df_esm2['SeqID'].astype(str).str.replace('>', '')

                # ==== Generate temporary FASTA for BLAST ====
                fasta_input_path = os.path.join(wd, f"temp_{channel}_blast_input.fasta")
                with open(fasta_input_path, "w") as f_fasta:
                    for s_id, s in zip(seqid_1, seq):
                        f_fasta.write(f"{s_id}\n{s}\n")

                # ==== Run BLAST ====
                blast_bin_path =  f"{nf_path}/blast_binaries/linux/blastp"
                blast_db_path  = f"{nf_path}/BLAST/{channel}_all_db/{channel}_train"
                blast_output_dir = os.path.join(wd, f"blast_output_{channel}")

                blast_scores = run_blast_and_get_scores(
                    fasta_input=fasta_input_path,
                    output_dir=blast_output_dir,
                    blast_bin_path=blast_bin_path,
                    blast_db_path=blast_db_path
                )

                # ==== Compute hybrid scores ====
                results = []
                for _, row in df_esm2.iterrows():
                    seq_id = row['SeqID']
                    sequence = row['Seq']
                    esm_prob = row[hybrid_col]
                    blast_score = blast_scores.get(seq_id, 0.0)
                    hybrid_score = esm_prob + blast_score
                    prediction = "Modulator" if hybrid_score > Threshold else "Non-Modulator"

                    results.append({
                        'SeqID': seq_id,
                        'Seq': sequence,
                        f"{cfg['label']}_ESM_Score": round(esm_prob, 4),
                        f"{cfg['label']}_BLAST_Score": round(blast_score, 4),
                        hybrid_col: round(hybrid_score, 4),
                        f"{hybrid_col}_Prediction": prediction
                    })
                    # print(f"[DEBUG] {channel} | {seq_id} | ESM: {esm_prob:.4f} | BLAST: {blast_score:.4f} | Hybrid: {hybrid_score:.4f}")

                df_final = pd.DataFrame(results)

                # ==== Clean up ====
                os.remove(out_file2)
                if os.path.exists(blast_output_dir):
                    shutil.rmtree(blast_output_dir)
                if os.path.exists(fasta_input_path):
                    os.remove(fasta_input_path)

            # print(f"[INFO] df_final for channel '{channel}':\n", df_final.head())

            # Append final channel DataFrame
            channel_results.append(df_final)

        # Merge all channel results
        from functools import reduce
        final_df = reduce(lambda left, right: pd.merge(left, right, on=['SeqID', 'Seq']), channel_results)
        final_df.to_csv(f"{wd}/{result_filename}", index=False)

        # print("\n[DEBUG] Columns in final_df:", final_df.columns.tolist())

        # Optional display
        if dplay == 1:
            print(final_df[
                (final_df.get('Na_hybrid_Prediction') == "Modulator") |
                (final_df.get('K_hybrid_Prediction') == "Modulator") |
                (final_df.get('Ca_hybrid_Prediction') == "Modulator") |
                (final_df.get('Other_hybrid_Prediction') == "Modulator")
            ])
        elif dplay == 2:
            print(final_df)
    
        # Clean up temporary file
        os.remove(f"{wd}/Sequence_1")

            #======================= Design Module starts from here =====================
    if Job == 2:

        #=================================== Na+ Channel Only ================================== 
        if Channel == 1:

            #===================== Model 1: ESM Only =====================
            if Model == 1:
                print(f'\n======= You are using the Design Module of IonNTxPred. Your results will be stored in file: {wd}/{result_filename} =======\n')
                print('==== Predicting Modulating Activity using ESM2-t12 model: Processing sequences, please wait ...')

                # Generate mutants
                muts = all_mutants(seq, seqid_1)
                muts.to_csv(f'{wd}/muts.csv', index=None, header=None)  
                seq = muts['Seq'].tolist()
                seqid_1 = muts['Mutant_ID'].tolist()

                # Load tokenizer and model
                model_save_path = f"{nf_path}/Model/saved_model_t33_na"
                tokenizer = AutoTokenizer.from_pretrained(model_save_path)
                model = EsmForSequenceClassification.from_pretrained(model_save_path)
                model.eval()

                # Run model and generate output
                run_esm_model(seq, seqid_1, f"{wd}/{result_filename}", Threshold)
                df13 = pd.read_csv(f"{wd}/{result_filename}")
                df13.columns = ['MutantID', 'Sequence', 'ESM Score', "Prediction"]

                # Add original SeqIDs
                seqid = pd.DataFrame(muts["SeqID"])
                df14 = pd.concat([seqid, df13], axis=1)
                df14['SeqID'] = df14['SeqID'].str.replace('>', '')
                # Save final result with SeqID back to file
                df14.to_csv(f"{wd}/{result_filename}", index=False)
                # Display results
                if dplay == 1:
                    df15 = df14[df14['Prediction'] == "Modulator"]
                    print(df15)
                elif dplay == 2:
                    print(df14)

                # Clean up
                os.remove(f'{wd}/Sequence_1')
                os.remove(f'{wd}/muts.csv')

            #===================== Model 2: Hybrid (ESM + BLAST) =====================
            elif Model == 2:
                print(f'\n======= You are using the Design Module of IonNTxPred. Your results will be stored in file: {wd}/{result_filename} =======\n')
                print('==== Predicting Modulating Activity using Hybrid model: Processing sequences, please wait ...')

                # Step 1: Generate mutants 
                muts = all_mutants(seq, seqid_1)
                muts.to_csv(f'{wd}/muts.csv', index=None, header=None)
                seq = muts['Seq'].tolist()
                seqid_1 = muts['Mutant_ID'].tolist()

                # Step 2: Load model and tokenizer
                model_save_path = f"{nf_path}/Model/saved_model_t33_na"
                tokenizer = AutoTokenizer.from_pretrained(model_save_path)
                model = EsmForSequenceClassification.from_pretrained(model_save_path)
                model.eval()

                # Step 3: Run ESM model
                out_file2 = f"{wd}/model2_output_na.csv"
                run_esm_model(seq, seqid_1, out_file2, Threshold)
                df_esm2 = pd.read_csv(out_file2)

                hybrid_col = "Na_hybrid"
                df_esm2.rename(columns={df_esm2.columns[2]: hybrid_col}, inplace=True)
                df_esm2['SeqID'] = df_esm2['SeqID'].astype(str).str.replace('>', '')

                # Step 4: Prepare FASTA for BLAST
                fasta_input_path = os.path.join(wd, "temp_na_blast_input.fasta")
                with open(fasta_input_path, "w") as f_fasta:
                    for s_id, s in zip(seqid_1, seq):
                        f_fasta.write(f">{s_id}\n{s}\n")

                # Step 5: Run BLAST
                blast_bin_path =  f"{nf_path}/blast_binaries/linux/blastp"
                blast_db_path  =  f"{nf_path}/BLAST/na_all_db/na_train"
                blast_output_dir = os.path.join(wd, "blast_output_na")

                blast_scores = run_blast_and_get_scores(
                    fasta_input=fasta_input_path,
                    output_dir=blast_output_dir,
                    blast_bin_path=blast_bin_path,
                    blast_db_path=blast_db_path
                )

                # Step 6: Compute Hybrid Scores
                results = []
                for _, row in df_esm2.iterrows():
                    seq_id = row['SeqID']
                    sequence = row['Seq']
                    esm_prob = row[hybrid_col]
                    blast_score = blast_scores.get(seq_id, 0.0)
                    hybrid_score = esm_prob + blast_score
                    prediction = "Modulator" if hybrid_score > Threshold else "Non-Modulator"

                    results.append({
                        'SeqID': seq_id,
                        'Seq': sequence,
                        hybrid_col: round(hybrid_score, 4),
                        f"{hybrid_col}_Prediction": prediction
                    })

                df_final = pd.DataFrame(results)

                # Step 7: Display based on dplay
                if dplay == 1:
                    df_mod = df_final[df_final[f"{hybrid_col}_Prediction"] == "Modulator"]
                    print(df_mod)
                elif dplay == 2:
                    print(df_final)

                # Step 8: Save to file
                df_final.to_csv(f"{wd}/{result_filename}", index=False)

                # Step 9: Cleanup
                if os.path.exists(out_file2):
                    os.remove(out_file2)
                if os.path.exists(fasta_input_path):
                    os.remove(fasta_input_path)
                if os.path.exists(blast_output_dir):
                    shutil.rmtree(blast_output_dir)
                if os.path.exists(f"{wd}/muts.csv"):
                    os.remove(f"{wd}/muts.csv")
                if os.path.exists(f"{wd}/Sequence_1"):
                    os.remove(f"{wd}/Sequence_1")
                

    #=================================== K+ ================================== 
        elif Channel == 2:
            if Model == 1:
                print(f'\n======= You are using the Design Module of IonNTxPred. Your results will be stored in file: {wd}/{result_filename} =======\n')
                print('==== Predicting Modulating Activity using ESM2-t12 model: Processing sequences, please wait ...')

                # Generate mutants
                muts = all_mutants(seq, seqid_1)
                muts.to_csv(f'{wd}/muts.csv', index=None, header=None)  
                seq = muts['Seq'].tolist()
                seqid_1 = muts['Mutant_ID'].tolist()

                # Load tokenizer and model
                model_save_path = f"{nf_path}/Model/saved_model_t33_k"
                tokenizer = AutoTokenizer.from_pretrained(model_save_path)
                model = EsmForSequenceClassification.from_pretrained(model_save_path)
                model.eval()

                # Run model and generate output
                run_esm_model(seq, seqid_1, f"{wd}/{result_filename}", Threshold)
                df13 = pd.read_csv(f"{wd}/{result_filename}")
                df13.columns = ['MutantID', 'Sequence', 'ESM Score', "Prediction"]
                
                # Add original SeqIDs
                seqid = pd.DataFrame(muts["SeqID"])
                df14 = pd.concat([seqid, df13], axis=1)
                df14['SeqID'] = df14['SeqID'].str.replace('>', '')
                # Save final result with SeqID back to file
                df14.to_csv(f"{wd}/{result_filename}", index=False)
                # Display results
                if dplay == 1:
                    df15 = df14[df14['Prediction'] == "Modulator"]
                    print(df15)
                elif dplay == 2:
                    print(df14)

                # Clean up
                os.remove(f'{wd}/Sequence_1')
                os.remove(f'{wd}/muts.csv')

            #===================== Model 2: Hybrid (ESM + BLAST) =====================
            elif Model == 2:
                print(f'\n======= You are using the Design Module of IonNTxPred. Your results will be stored in file: {wd}/{result_filename} =======\n')
                print('==== Predicting Modulating Activity using Hybrid model: Processing sequences, please wait ...')

                # Step 1: Generate mutants 
                muts = all_mutants(seq, seqid_1)
                muts.to_csv(f'{wd}/muts.csv', index=None, header=None)
                seq = muts['Seq'].tolist()
                seqid_1 = muts['Mutant_ID'].tolist()

                # Step 2: Load model and tokenizer
                model_save_path = f"{nf_path}/Model/saved_model_t33_k"
                tokenizer = AutoTokenizer.from_pretrained(model_save_path)
                model = EsmForSequenceClassification.from_pretrained(model_save_path)
                model.eval()

                # Step 3: Run ESM model
                out_file2 = f"{wd}/model2_output_k.csv"
                run_esm_model(seq, seqid_1, out_file2, Threshold)
                df_esm2 = pd.read_csv(out_file2)

                hybrid_col = "K_hybrid"
                df_esm2.rename(columns={df_esm2.columns[2]: hybrid_col}, inplace=True)
                df_esm2['SeqID'] = df_esm2['SeqID'].astype(str).str.replace('>', '')

                # Step 4: Prepare FASTA for BLAST
                fasta_input_path = os.path.join(wd, "temp_k_blast_input.fasta")
                with open(fasta_input_path, "w") as f_fasta:
                    for s_id, s in zip(seqid_1, seq):
                        f_fasta.write(f">{s_id}\n{s}\n")

                # Step 5: Run BLAST
                blast_bin_path =  f"{nf_path}/blast_binaries/linux/blastp"
                blast_db_path  = f"{nf_path}/BLAST/k_all_db/k_train"
                blast_output_dir = os.path.join(wd, "blast_output_k")

                blast_scores = run_blast_and_get_scores(
                    fasta_input=fasta_input_path,
                    output_dir=blast_output_dir,
                    blast_bin_path=blast_bin_path,
                    blast_db_path=blast_db_path
                )

                # Step 6: Compute Hybrid Scores
                results = []
                for _, row in df_esm2.iterrows():
                    seq_id = row['SeqID']
                    sequence = row['Seq']
                    esm_prob = row[hybrid_col]
                    blast_score = blast_scores.get(seq_id, 0.0)
                    hybrid_score = esm_prob + blast_score
                    prediction = "Modulator" if hybrid_score > Threshold else "Non-Modulator"

                    results.append({
                        'SeqID': seq_id,
                        'Seq': sequence,
                        hybrid_col: round(hybrid_score, 4),
                        f"{hybrid_col}_Prediction": prediction
                    })

                df_final = pd.DataFrame(results)

                # Step 7: Display based on dplay
                if dplay == 1:
                    df_mod = df_final[df_final[f"{hybrid_col}_Prediction"] == "Modulator"]
                    print(df_mod)
                elif dplay == 2:
                    print(df_final)

                # Step 8: Save to file
                df_final.to_csv(f"{wd}/{result_filename}", index=False)

                # Step 9: Cleanup
                if os.path.exists(out_file2):
                    os.remove(out_file2)
                if os.path.exists(fasta_input_path):
                    os.remove(fasta_input_path)
                if os.path.exists(blast_output_dir):
                    shutil.rmtree(blast_output_dir)
                if os.path.exists(f"{wd}/muts.csv"):
                    os.remove(f"{wd}/muts.csv")
                if os.path.exists(f"{wd}/Sequence_1"):
                    os.remove(f"{wd}/Sequence_1")

    #=================================== Ca2+ ================================== 
        elif Channel == 3:
            if Model == 1:
                print(f'\n======= You are using the Design Module of IonNTxPred. Your results will be stored in file: {wd}/{result_filename} =======\n')
                print('==== Predicting Modulating Activity using ESM2-t12 model: Processing sequences, please wait ...')

                # Generate mutants
                muts = all_mutants(seq, seqid_1)
                muts.to_csv(f'{wd}/muts.csv', index=None, header=None)  
                seq = muts['Seq'].tolist()
                seqid_1 = muts['Mutant_ID'].tolist()

                # Load tokenizer and model
                model_save_path = f"{nf_path}/Model/saved_model_t33_ca"
                tokenizer = AutoTokenizer.from_pretrained(model_save_path)
                model = EsmForSequenceClassification.from_pretrained(model_save_path)
                model.eval()

                # Run model and generate output
                run_esm_model(seq, seqid_1, f"{wd}/{result_filename}", Threshold)
                df13 = pd.read_csv(f"{wd}/{result_filename}")
                df13.columns = ['MutantID', 'Sequence', 'ESM Score', "Prediction"]
        
                # Add original SeqIDs
                seqid = pd.DataFrame(muts["SeqID"])
                df14 = pd.concat([seqid, df13], axis=1)
                df14['SeqID'] = df14['SeqID'].str.replace('>', '')
                # Save final result with SeqID back to file
                df14.to_csv(f"{wd}/{result_filename}", index=False)
                # Display results
                if dplay == 1:
                    df15 = df14[df14['Prediction'] == "Modulator"]
                    print(df15)
                elif dplay == 2:
                    print(df14)

                # Clean up
                os.remove(f'{wd}/Sequence_1')
                os.remove(f'{wd}/muts.csv')

            #===================== Model 2: Hybrid (ESM + BLAST) =====================
            elif Model == 2:
                print(f'\n======= You are using the Design Module of IonNTxPred. Your results will be stored in file: {wd}/{result_filename} =======\n')
                print('==== Predicting Modulating Activity using Hybrid model: Processing sequences, please wait ...')

                # Step 1: Generate mutants 
                muts = all_mutants(seq, seqid_1)
                muts.to_csv(f'{wd}/muts.csv', index=None, header=None)
                seq = muts['Seq'].tolist()
                seqid_1 = muts['Mutant_ID'].tolist()

                # Step 2: Load model and tokenizer
                model_save_path = f"{nf_path}/Model/saved_model_t33_ca"
                tokenizer = AutoTokenizer.from_pretrained(model_save_path)
                model = EsmForSequenceClassification.from_pretrained(model_save_path)
                model.eval()

                # Step 3: Run ESM model
                out_file2 = f"{wd}/model2_output_ca.csv"
                run_esm_model(seq, seqid_1, out_file2, Threshold)
                df_esm2 = pd.read_csv(out_file2)

                hybrid_col = "Ca_hybrid"
                df_esm2.rename(columns={df_esm2.columns[2]: hybrid_col}, inplace=True)
                df_esm2['SeqID'] = df_esm2['SeqID'].astype(str).str.replace('>', '')

                # Step 4: Prepare FASTA for BLAST
                fasta_input_path = os.path.join(wd, "temp_ca_blast_input.fasta")
                with open(fasta_input_path, "w") as f_fasta:
                    for s_id, s in zip(seqid_1, seq):
                        f_fasta.write(f">{s_id}\n{s}\n")

                # Step 5: Run BLAST
                blast_bin_path =  f"{nf_path}/blast_binaries/linux/blastp"
                blast_db_path  = f"{nf_path}/BLAST/ca_all_db/ca_train"
                blast_output_dir = os.path.join(wd, "blast_output_ca")

                blast_scores = run_blast_and_get_scores(
                    fasta_input=fasta_input_path,
                    output_dir=blast_output_dir,
                    blast_bin_path=blast_bin_path,
                    blast_db_path=blast_db_path
                )

                # Step 6: Compute Hybrid Scores
                results = []
                for _, row in df_esm2.iterrows():
                    seq_id = row['SeqID']
                    sequence = row['Seq']
                    esm_prob = row[hybrid_col]
                    blast_score = blast_scores.get(seq_id, 0.0)
                    hybrid_score = esm_prob + blast_score
                    prediction = "Modulator" if hybrid_score > Threshold else "Non-Modulator"

                    results.append({
                        'SeqID': seq_id,
                        'Seq': sequence,
                        hybrid_col: round(hybrid_score, 4),
                        f"{hybrid_col}_Prediction": prediction
                    })

                df_final = pd.DataFrame(results)

                # Step 7: Display based on dplay
                if dplay == 1:
                    df_mod = df_final[df_final[f"{hybrid_col}_Prediction"] == "Modulator"]
                    print(df_mod)
                elif dplay == 2:
                    print(df_final)

                # Step 8: Save to file
                df_final.to_csv(f"{wd}/{result_filename}", index=False)

                # Step 9: Cleanup
                if os.path.exists(out_file2):
                    os.remove(out_file2)
                if os.path.exists(fasta_input_path):
                    os.remove(fasta_input_path)
                if os.path.exists(blast_output_dir):
                    shutil.rmtree(blast_output_dir)
                if os.path.exists(f"{wd}/muts.csv"):
                    os.remove(f"{wd}/muts.csv")
                if os.path.exists(f"{wd}/Sequence_1"):
                    os.remove(f"{wd}/Sequence_1")

    #=================================== Other ================================== 
        elif Channel == 4:
            if Model == 1:
                print(f'\n======= You are using the Design Module of IonNTxPred. Your results will be stored in file: {wd}/{result_filename} =======\n')
                print('==== Predicting Modulating Activity using ESM2-t12 model: Processing sequences, please wait ...')

                # Generate mutants
                muts = all_mutants(seq, seqid_1)
                muts.to_csv(f'{wd}/muts.csv', index=None, header=None)  
                seq = muts['Seq'].tolist()
                seqid_1 = muts['Mutant_ID'].tolist()

                # Load tokenizer and model
                model_save_path = f"{nf_path}/Model/saved_model_t33_other"
                tokenizer = AutoTokenizer.from_pretrained(model_save_path)
                model = EsmForSequenceClassification.from_pretrained(model_save_path)
                model.eval()

                # Run model and generate output
                run_esm_model(seq, seqid_1, f"{wd}/{result_filename}", Threshold)
                df13 = pd.read_csv(f"{wd}/{result_filename}")
                df13.columns = ['MutantID', 'Sequence', 'ESM Score', "Prediction"]

                # Add original SeqIDs
                seqid = pd.DataFrame(muts["SeqID"])
                df14 = pd.concat([seqid, df13], axis=1)
                df14['SeqID'] = df14['SeqID'].str.replace('>', '')
                # Save final result with SeqID back to file
                df14.to_csv(f"{wd}/{result_filename}", index=False)
                # Display results
                if dplay == 1:
                    df15 = df14[df14['Prediction'] == "Modulator"]
                    print(df15)
                elif dplay == 2:
                    print(df14)

                # Clean up
                os.remove(f'{wd}/Sequence_1')
                os.remove(f'{wd}/muts.csv')

            #===================== Model 2: Hybrid (ESM + BLAST) =====================
            elif Model == 2:
                print(f'\n======= You are using the Design Module of IonNTxPred. Your results will be stored in file: {wd}/{result_filename} =======\n')
                print('==== Predicting Modulating Activity using Hybrid model: Processing sequences, please wait ...')

                # Step 1: Generate mutants 
                muts = all_mutants(seq, seqid_1)
                muts.to_csv(f'{wd}/muts.csv', index=None, header=None)
                seq = muts['Seq'].tolist()
                seqid_1 = muts['Mutant_ID'].tolist()

                # Step 2: Load model and tokenizer
                model_save_path = f"{nf_path}/Model/saved_model_t33_other"
                tokenizer = AutoTokenizer.from_pretrained(model_save_path)
                model = EsmForSequenceClassification.from_pretrained(model_save_path)
                model.eval()

                # Step 3: Run ESM model
                out_file2 = f"{wd}/model2_output_other.csv"
                run_esm_model(seq, seqid_1, out_file2, Threshold)
                df_esm2 = pd.read_csv(out_file2)

                hybrid_col = "Other_hybrid"
                df_esm2.rename(columns={df_esm2.columns[2]: hybrid_col}, inplace=True)
                df_esm2['SeqID'] = df_esm2['SeqID'].astype(str).str.replace('>', '')

                # Step 4: Prepare FASTA for BLAST
                fasta_input_path = os.path.join(wd, "temp_other_blast_input.fasta")
                with open(fasta_input_path, "w") as f_fasta:
                    for s_id, s in zip(seqid_1, seq):
                        f_fasta.write(f">{s_id}\n{s}\n")

                # Step 5: Run BLAST
                blast_bin_path =  f"{nf_path}/blast_binaries/linux/blastp"
                blast_db_path  = f"{nf_path}/BLAST/other_all_db/other_train"
                blast_output_dir = os.path.join(wd, "blast_output_other")

                blast_scores = run_blast_and_get_scores(
                    fasta_input=fasta_input_path,
                    output_dir=blast_output_dir,
                    blast_bin_path=blast_bin_path,
                    blast_db_path=blast_db_path
                )


                # Step 6: Compute Hybrid Scores
                results = []
                for _, row in df_esm2.iterrows():
                    seq_id = row['SeqID']
                    sequence = row['Seq']
                    esm_prob = row[hybrid_col]
                    blast_score = blast_scores.get(seq_id, 0.0)
                    hybrid_score = esm_prob + blast_score
                    prediction = "Modulator" if hybrid_score > Threshold else "Non-Modulator"

                    results.append({
                        'SeqID': seq_id,
                        'Seq': sequence,
                        hybrid_col: round(hybrid_score, 4),
                        f"{hybrid_col}_Prediction": prediction
                    })

                df_final = pd.DataFrame(results)

                # Step 7: Display based on dplay
                if dplay == 1:
                    df_mod = df_final[df_final[f"{hybrid_col}_Prediction"] == "Modulator"]
                    print(df_mod)
                elif dplay == 2:
                    print(df_final)

                # Step 8: Save to file
                df_final.to_csv(f"{wd}/{result_filename}", index=False)

                # Step 9: Cleanup
                if os.path.exists(out_file2):
                    os.remove(out_file2)
                if os.path.exists(fasta_input_path):
                    os.remove(fasta_input_path)
                if os.path.exists(blast_output_dir):
                    shutil.rmtree(blast_output_dir)
                if os.path.exists(f"{wd}/muts.csv"):
                    os.remove(f"{wd}/muts.csv")
                if os.path.exists(f"{wd}/Sequence_1"):
                    os.remove(f"{wd}/Sequence_1")     
        
    #======================= Protein Scanning Module starts from here =====================      
    if Job == 3:
                    #=================================== Na+ ==================================        
        if Channel == 1:
            if Model == 1:
                print('\n======= You are using the Protein Scanning module of IonNTxPred. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
                print('==== Scanning through ESM2-t12 model: Processing sequences please wait ...')
                # Load the tokenizer and model
                model_save_path = f"{nf_path}/Model/saved_model_t33_na"
                tokenizer = AutoTokenizer.from_pretrained(model_save_path)
                model = EsmForSequenceClassification.from_pretrained(model_save_path)
                model.eval()

                # Wrap the list into a DataFrame if it's not already
                if isinstance(seq, list):
                    seq = pd.DataFrame(seq)
                seq = seq.iloc[:, 0].tolist()
                df_1 = seq_pattern(seq,seqid_1,Win_len)
                seq = df_1["Seq"].tolist()
                seqid_1=df_1["SeqID"].tolist()
                run_esm_model(seq, seqid_1, f"{wd}/{result_filename}", Threshold)
                df13 = pd.read_csv(f"{wd}/{result_filename}")
                df13.rename(columns={"ML Score": "ESM Score"}, inplace=True)
                df13 = df13.drop(columns = ['SeqID', 'Seq'])
                df13 = pd.concat([df_1, df13], axis=1)
                df13["SeqID"] = df13["SeqID"].str.lstrip(">")
                df13.columns = ['SeqID','Pattern ID', 'Start', 'End', 'Sequence', 'ESM Score', "Prediction"]
                df13 = round(df13, 4)
                df13.to_csv(f"{wd}/{result_filename}", index=None)

                if dplay == 1:
                    df13 = df13.loc[df13.Prediction == "Modulator"]
                    print(df13)
                elif dplay == 2:
                    df13=df13
                    print(df13)
                os.remove(f'{wd}/Sequence_1')                

            elif Model == 2: 
                print('\n======= You are using the Protein Scanning Module of IonNTxPred. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
                print('==== Scanning sequences and predicting Modulating Activity using Hybrid model: Processing sequences please wait ...')

                # Load ESM model and tokenizer
                model_save_path = f"{nf_path}/Model/saved_model_t33_na"
                tokenizer = AutoTokenizer.from_pretrained(model_save_path)
                model = EsmForSequenceClassification.from_pretrained(model_save_path)
                model.eval()

                # Ensure input is DataFrame
                if isinstance(seq, list):
                    seq = pd.DataFrame(seq)

                seq = seq.iloc[:, 0].tolist()
                df_1 = seq_pattern(seq, seqid_1, Win_len)
                seq = df_1["Seq"].tolist()

                # Generate pattern-style SeqIDs
                raw_seqid = df_1["SeqID"].str.lstrip(">").tolist()
                pattern_counter = defaultdict(int)
                seqid_1 = []
                for sid in raw_seqid:
                    pattern_counter[sid] += 1
                    seqid_1.append(f"{sid}_Pattern_{pattern_counter[sid]}")
                # IMPORTANT: update df_1 with new Pattern IDs
                df_1["SeqID"] = seqid_1
                # Step 1: Run ESM model
                out_file = f"{wd}/out22"
                run_esm_model(seq, seqid_1, out_file, Threshold)

                # Step 2: Prepare FASTA for BLAST
                fasta_path = f"{wd}/sequences.fasta"
                with open(fasta_path, "w") as fasta_file:
                    for s, sid in zip(seq, seqid_1):
                        fasta_file.write(f">{sid}\n{s}\n")

                # Step 3: Run BLAST
                blast_bin_path =  f"{nf_path}/blast_binaries/linux/blastp"
                blast_db_path  = f"{nf_path}/BLAST/na_all_db/na_train"
                blast_output_dir = os.path.join(wd, "blast_output_na")

                blast_scores = run_blast_and_get_scores(
                    fasta_input=fasta_path,
                    output_dir=blast_output_dir,
                    blast_bin_path=blast_bin_path,
                    blast_db_path=blast_db_path
                )

                # Step 4: Read ESM results and calculate hybrid score
                df_esm = pd.read_csv(out_file)
                hybrid_col = "Na_hybrid"
                df_esm.rename(columns={df_esm.columns[2]: hybrid_col}, inplace=True)
                df_esm["SeqID"] = df_esm["SeqID"].astype(str).str.replace('>', '')

                results = []
                for _, row in df_esm.iterrows():
                    seq_id = row["SeqID"]
                    sequence = row["Seq"]
                    esm_score = row[hybrid_col]
                    blast_score = blast_scores.get(seq_id, 0.0)
                    hybrid_score = esm_score + blast_score
                    prediction = "Modulator" if hybrid_score > Threshold else "Non-Modulator"

                    results.append({
                        "SeqID": seq_id,
                        "Sequence": sequence,
                        hybrid_col: round(hybrid_score, 4),
                        f"{hybrid_col}_Prediction": prediction
                    })

                df_final = pd.DataFrame(results)
                df_1.rename(columns={"Seq": "Sequence"}, inplace=True)
                # Step 5: Merge with pattern metadata
                df44 = df_1.merge(df_final, on=["SeqID", "Sequence"])

                # Step 6: Display results
                if dplay == 1:
                    df44 = df44[df44[f"{hybrid_col}_Prediction"] == "Modulator"]
                    print(df44)
                elif dplay == 2:
                    print(df44)

                # Step 7: Save results
                df44 = round(df44, 4)
                df44.to_csv(f"{wd}/{result_filename}", index=False)

                # Step 8: Cleanup
                temp_files = ['sequences.fasta', 'Sequence_1', 'out22']
                if os.path.exists(blast_output_dir):
                    shutil.rmtree(blast_output_dir)
                for file in temp_files:
                    path = os.path.join(wd, file)
                    if os.path.exists(path):
                        os.remove(path)

    #=================================== K+ ==================================        
        elif Channel == 2:
            if Model == 1:
                print('\n======= You are using the Protein Scanning module of IonNTxPred. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
                print('==== Scanning through ESM2-t12 model: Processing sequences please wait ...')
                # Load the tokenizer and model
                model_save_path = f"{nf_path}/Model/saved_model_t33_k"
                tokenizer = AutoTokenizer.from_pretrained(model_save_path)
                model = EsmForSequenceClassification.from_pretrained(model_save_path)
                model.eval()

                # Wrap the list into a DataFrame if it's not already
                if isinstance(seq, list):
                    seq = pd.DataFrame(seq)
                seq = seq.iloc[:, 0].tolist()
                df_1 = seq_pattern(seq,seqid_1,Win_len)
                seq = df_1["Seq"].tolist()
                seqid_1=df_1["SeqID"].tolist()
                run_esm_model(seq, seqid_1, f"{wd}/{result_filename}", Threshold)
                df13 = pd.read_csv(f"{wd}/{result_filename}")
                df13.rename(columns={"ML Score": "ESM Score"}, inplace=True)
                df13 = df13.drop(columns = ['SeqID', 'Seq'])
                df13 = pd.concat([df_1, df13], axis=1)
                df13["SeqID"] = df13["SeqID"].str.lstrip(">")
                df13.columns = ['SeqID','Pattern ID', 'Start', 'End', 'Sequence', 'ESM Score', "Prediction"]
                df13 = round(df13, 4)
                df13.to_csv(f"{wd}/{result_filename}", index=None)

                if dplay == 1:
                    df13 = df13.loc[df13.Prediction == "Modulator"]
                    print(df13)
                elif dplay == 2:
                    df13=df13
                    print(df13)
                os.remove(f'{wd}/Sequence_1')                

            elif Model == 2: 
                print('\n======= You are using the Protein Scanning Module of IonNTxPred. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
                print('==== Scanning sequences and predicting Modulating Activity using Hybrid model: Processing sequences please wait ...')

                # Load ESM model and tokenizer
                model_save_path = f"{nf_path}/Model/saved_model_t33_k"
                tokenizer = AutoTokenizer.from_pretrained(model_save_path)
                model = EsmForSequenceClassification.from_pretrained(model_save_path)
                model.eval()

                # Ensure input is DataFrame
                if isinstance(seq, list):
                    seq = pd.DataFrame(seq)

                seq = seq.iloc[:, 0].tolist()
                df_1 = seq_pattern(seq, seqid_1, Win_len)
                seq = df_1["Seq"].tolist()

                # Generate pattern-style SeqIDs
                raw_seqid = df_1["SeqID"].str.lstrip(">").tolist()
                pattern_counter = defaultdict(int)
                seqid_1 = []
                for sid in raw_seqid:
                    pattern_counter[sid] += 1
                    seqid_1.append(f"{sid}_Pattern_{pattern_counter[sid]}")
                # IMPORTANT: update df_1 with new Pattern IDs
                df_1["SeqID"] = seqid_1
                # Step 1: Run ESM model
                out_file = f"{wd}/out22"
                run_esm_model(seq, seqid_1, out_file, Threshold)

                # Step 2: Prepare FASTA for BLAST
                fasta_path = f"{wd}/sequences.fasta"
                with open(fasta_path, "w") as fasta_file:
                    for s, sid in zip(seq, seqid_1):
                        fasta_file.write(f">{sid}\n{s}\n")

                # Step 3: Run BLAST
                blast_bin_path =  f"{nf_path}/blast_binaries/linux/blastp"
                blast_db_path  = f"{nf_path}/BLAST/k_all_db/k_train"
                blast_output_dir = os.path.join(wd, "blast_output_k")

                blast_scores = run_blast_and_get_scores(
                    fasta_input=fasta_path,
                    output_dir=blast_output_dir,
                    blast_bin_path=blast_bin_path,
                    blast_db_path=blast_db_path
                )

                # Step 4: Read ESM results and calculate hybrid score
                df_esm = pd.read_csv(out_file)
                hybrid_col = "K_hybrid"
                df_esm.rename(columns={df_esm.columns[2]: hybrid_col}, inplace=True)
                df_esm["SeqID"] = df_esm["SeqID"].astype(str).str.replace('>', '')

                results = []
                for _, row in df_esm.iterrows():
                    seq_id = row["SeqID"]
                    sequence = row["Seq"]
                    esm_score = row[hybrid_col]
                    blast_score = blast_scores.get(seq_id, 0.0)
                    hybrid_score = esm_score + blast_score
                    prediction = "Modulator" if hybrid_score > Threshold else "Non-Modulator"

                    results.append({
                        "SeqID": seq_id,
                        "Sequence": sequence,
                        hybrid_col: round(hybrid_score, 4),
                        f"{hybrid_col}_Prediction": prediction
                    })

                df_final = pd.DataFrame(results)
                df_1.rename(columns={"Seq": "Sequence"}, inplace=True)
                # Step 5: Merge with pattern metadata
                df44 = df_1.merge(df_final, on=["SeqID", "Sequence"])

                # Step 6: Display results
                if dplay == 1:
                    df44 = df44[df44[f"{hybrid_col}_Prediction"] == "Modulator"]
                    print(df44)
                elif dplay == 2:
                    print(df44)

                # Step 7: Save results
                df44 = round(df44, 4)
                df44.to_csv(f"{wd}/{result_filename}", index=False)

                # Step 8: Cleanup
                temp_files = ['sequences.fasta', 'Sequence_1', 'out22']
                if os.path.exists(blast_output_dir):
                    shutil.rmtree(blast_output_dir)
                for file in temp_files:
                    path = os.path.join(wd, file)
                    if os.path.exists(path):
                        os.remove(path)
    

    #=================================== Ca2+ ==================================        
        elif Channel == 3:
            if Model == 1:
                print('\n======= You are using the Protein Scanning module of IonNTxPred. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
                print('==== Scanning through ESM2-t12 model: Processing sequences please wait ...')
                # Load the tokenizer and model
                model_save_path = f"{nf_path}/Model/saved_model_t33_ca"
                tokenizer = AutoTokenizer.from_pretrained(model_save_path)
                model = EsmForSequenceClassification.from_pretrained(model_save_path)
                model.eval()

                # Wrap the list into a DataFrame if it's not already
                if isinstance(seq, list):
                    seq = pd.DataFrame(seq)
                seq = seq.iloc[:, 0].tolist()
                df_1 = seq_pattern(seq,seqid_1,Win_len)
                seq = df_1["Seq"].tolist()
                seqid_1=df_1["SeqID"].tolist()
                run_esm_model(seq, seqid_1, f"{wd}/{result_filename}", Threshold)
                df13 = pd.read_csv(f"{wd}/{result_filename}")
                df13.rename(columns={"ML Score": "ESM Score"}, inplace=True)
                df13 = df13.drop(columns = ['SeqID', 'Seq'])
                df13 = pd.concat([df_1, df13], axis=1)
                df13["SeqID"] = df13["SeqID"].str.lstrip(">")
                df13.columns = ['SeqID','Pattern ID', 'Start', 'End', 'Sequence', 'ESM Score', "Prediction"]
                df13 = round(df13, 4)
                df13.to_csv(f"{wd}/{result_filename}", index=None)

                if dplay == 1:
                    df13 = df13.loc[df13.Prediction == "Modulator"]
                    print(df13)
                elif dplay == 2:
                    df13=df13
                    print(df13)
                os.remove(f'{wd}/Sequence_1')                

            elif Model == 2: 
                print('\n======= You are using the Protein Scanning Module of IonNTxPred. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
                print('==== Scanning sequences and predicting Modulating Activity using Hybrid model: Processing sequences please wait ...')

                # Load ESM model and tokenizer
                model_save_path = f"{nf_path}/Model/saved_model_t33_ca"
                tokenizer = AutoTokenizer.from_pretrained(model_save_path)
                model = EsmForSequenceClassification.from_pretrained(model_save_path)
                model.eval()

                # Ensure input is DataFrame
                if isinstance(seq, list):
                    seq = pd.DataFrame(seq)

                seq = seq.iloc[:, 0].tolist()
                df_1 = seq_pattern(seq, seqid_1, Win_len)
                seq = df_1["Seq"].tolist()

                # Generate pattern-style SeqIDs
                raw_seqid = df_1["SeqID"].str.lstrip(">").tolist()
                pattern_counter = defaultdict(int)
                seqid_1 = []
                for sid in raw_seqid:
                    pattern_counter[sid] += 1
                    seqid_1.append(f"{sid}_Pattern_{pattern_counter[sid]}")
                # 🔹 IMPORTANT: update df_1 with new Pattern IDs
                df_1["SeqID"] = seqid_1
                # Step 1: Run ESM model
                out_file = f"{wd}/out22"
                run_esm_model(seq, seqid_1, out_file, Threshold)

                # Step 2: Prepare FASTA for BLAST
                fasta_path = f"{wd}/sequences.fasta"
                with open(fasta_path, "w") as fasta_file:
                    for s, sid in zip(seq, seqid_1):
                        fasta_file.write(f">{sid}\n{s}\n")

                # Step 3: Run BLAST
                blast_bin_path =  f"{nf_path}/blast_binaries/linux/blastp"
                blast_db_path  = f"{nf_path}/BLAST/ca_all_db/ca_train"
                blast_output_dir = os.path.join(wd, "blast_output_ca")

                blast_scores = run_blast_and_get_scores(
                    fasta_input=fasta_path,
                    output_dir=blast_output_dir,
                    blast_bin_path=blast_bin_path,
                    blast_db_path=blast_db_path
                )

                # Step 4: Read ESM results and calculate hybrid score
                df_esm = pd.read_csv(out_file)
                hybrid_col = "Ca_hybrid"
                df_esm.rename(columns={df_esm.columns[2]: hybrid_col}, inplace=True)
                df_esm["SeqID"] = df_esm["SeqID"].astype(str).str.replace('>', '')

                results = []
                for _, row in df_esm.iterrows():
                    seq_id = row["SeqID"]
                    sequence = row["Seq"]
                    esm_score = row[hybrid_col]
                    blast_score = blast_scores.get(seq_id, 0.0)
                    hybrid_score = esm_score + blast_score
                    prediction = "Modulator" if hybrid_score > Threshold else "Non-Modulator"

                    results.append({
                        "SeqID": seq_id,
                        "Sequence": sequence,
                        hybrid_col: round(hybrid_score, 4),
                        f"{hybrid_col}_Prediction": prediction
                    })

                df_final = pd.DataFrame(results)
                df_1.rename(columns={"Seq": "Sequence"}, inplace=True)
                # Step 5: Merge with pattern metadata
                df44 = df_1.merge(df_final, on=["SeqID", "Sequence"])

                # Step 6: Display results
                if dplay == 1:
                    df44 = df44[df44[f"{hybrid_col}_Prediction"] == "Modulator"]
                    print(df44)
                elif dplay == 2:
                    print(df44)

                # Step 7: Save results
                df44 = round(df44, 4)
                df44.to_csv(f"{wd}/{result_filename}", index=False)

                # Step 8: Cleanup
                temp_files = ['sequences.fasta', 'Sequence_1', 'out22']
                if os.path.exists(blast_output_dir):
                    shutil.rmtree(blast_output_dir)
                for file in temp_files:
                    path = os.path.join(wd, file)
                    if os.path.exists(path):
                        os.remove(path)  

    #=================================== Other ==================================         
        elif Channel == 4:
            if Model == 1:
                print('\n======= You are using the Protein Scanning module of IonNTxPred. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
                print('==== Scanning through ESM2-t12 model: Processing sequences please wait ...')

                # Load the tokenizer and model
                model_save_path = f"{nf_path}/Model/saved_model_t33_other"
                tokenizer = AutoTokenizer.from_pretrained(model_save_path)
                model = EsmForSequenceClassification.from_pretrained(model_save_path)
                model.eval()

                # Wrap the list into a DataFrame if it's not already
                if isinstance(seq, list):
                    seq = pd.DataFrame(seq)

                seq = seq.iloc[:, 0].tolist()
                df_1 = seq_pattern(seq, seqid_1, Win_len)
                seq = df_1["Seq"].tolist()
                seqid_1 = df_1["SeqID"].tolist()
                run_esm_model(seq, seqid_1, f"{wd}/{result_filename}", Threshold)

                df13 = pd.read_csv(f"{wd}/{result_filename}")
                df13.rename(columns={"ML Score": "ESM Score"}, inplace=True)
                df13 = df13.drop(columns = ['SeqID', 'Seq'])
                df13 = pd.concat([df_1, df13], axis=1)
                df13["SeqID"] = df13["SeqID"].str.lstrip(">")
                df13.columns = ['SeqID','Pattern ID', 'Start', 'End', 'Sequence', 'ESM Score', "Prediction"]
                df13 = round(df13, 4)
                df13.to_csv(f"{wd}/{result_filename}", index=None)

                if dplay == 1:
                    df13 = df13.loc[df13.Prediction == "Modulator"]
                    print(df13)
                elif dplay == 2:
                    print(df13)

                os.remove(f'{wd}/Sequence_1')

            elif Model == 2: 
                print('\n======= You are using the Protein Scanning Module of IonNTxPred. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
                print('==== Scanning sequences and predicting Modulating Activity using Hybrid model: Processing sequences please wait ...')

                # Load ESM model and tokenizer
                model_save_path = f"{nf_path}/Model/saved_model_t33_other"
                tokenizer = AutoTokenizer.from_pretrained(model_save_path)
                model = EsmForSequenceClassification.from_pretrained(model_save_path)
                model.eval()

                # Ensure input is DataFrame
                if isinstance(seq, list):
                    seq = pd.DataFrame(seq)

                seq = seq.iloc[:, 0].tolist()
                df_1 = seq_pattern(seq, seqid_1, Win_len)
                seq = df_1["Seq"].tolist()

                # Generate pattern-style SeqIDs
                raw_seqid = df_1["SeqID"].str.lstrip(">").tolist()
                pattern_counter = defaultdict(int)
                seqid_1 = []
                for sid in raw_seqid:
                    pattern_counter[sid] += 1
                    seqid_1.append(f"{sid}_Pattern_{pattern_counter[sid]}")
                # IMPORTANT: update df_1 with new Pattern IDs
                df_1["SeqID"] = seqid_1
                # Step 1: Run ESM model
                out_file = f"{wd}/out22"
                run_esm_model(seq, seqid_1, out_file, Threshold)

                # Step 2: Prepare FASTA for BLAST
                fasta_path = f"{wd}/sequences.fasta"
                with open(fasta_path, "w") as fasta_file:
                    for s, sid in zip(seq, seqid_1):
                        fasta_file.write(f">{sid}\n{s}\n")

                # Step 3: Run BLAST
                blast_bin_path =  f"{nf_path}/blast_binaries/linux/blastp"
                blast_db_path  = f"{nf_path}/BLAST/other_all_db/other_train"
                blast_output_dir = os.path.join(wd, "blast_output_other")

                blast_scores = run_blast_and_get_scores(
                    fasta_input=fasta_path,
                    output_dir=blast_output_dir,
                    blast_bin_path=blast_bin_path,
                    blast_db_path=blast_db_path
                )

                # Step 4: Read ESM results and calculate hybrid score
                df_esm = pd.read_csv(out_file)
                hybrid_col = "Other_hybrid"
                df_esm.rename(columns={df_esm.columns[2]: hybrid_col}, inplace=True)
                df_esm["SeqID"] = df_esm["SeqID"].astype(str).str.replace('>', '')

                results = []
                for _, row in df_esm.iterrows():
                    seq_id = row["SeqID"]
                    sequence = row["Seq"]
                    esm_score = row[hybrid_col]
                    blast_score = blast_scores.get(seq_id, 0.0)
                    hybrid_score = esm_score + blast_score
                    prediction = "Modulator" if hybrid_score > Threshold else "Non-Modulator"

                    results.append({
                        "SeqID": seq_id,
                        "Sequence": sequence,
                        hybrid_col: round(hybrid_score, 4),
                        f"{hybrid_col}_Prediction": prediction
                    })

                df_final = pd.DataFrame(results)
                df_1.rename(columns={"Seq": "Sequence"}, inplace=True)
                # Step 5: Merge with pattern metadata
                df44 = df_1.merge(df_final, on=["SeqID", "Sequence"])

                # Step 6: Display results
                if dplay == 1:
                    df44 = df44[df44[f"{hybrid_col}_Prediction"] == "Modulator"]
                    print(df44)
                elif dplay == 2:
                    print(df44)

                # Step 7: Save results
                df44 = round(df44, 4)
                df44.to_csv(f"{wd}/{result_filename}", index=False)

                # Step 8: Cleanup
                temp_files = ['sequences.fasta', 'Sequence_1', 'out22']
                if os.path.exists(blast_output_dir):
                    shutil.rmtree(blast_output_dir)
                for file in temp_files:
                    path = os.path.join(wd, file)
                    if os.path.exists(path):
                        os.remove(path)          
                            
                #======================= Motif Scanning Module starts from here =====================
    if Job == 4:
                #=================================== Na+ ==================================        
            if Channel == 1:
                print('\n======= You are using the Motif Scanning module of IonNTxPred. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
                print('==== Scanning through Na ion channel: Processing sequences please wait ...')
                df_2, dfseq = readseq(Sequence)
                df1 = lenchk(dfseq)
                merci_script = f"{nf_path}/merci/MERCI_motif_locator.pl"
                pos_motif_file = f"{nf_path}/motif/Na/pos_motif.txt"
                neg_motif_file = f"{nf_path}/motif/Na/neg_motif.txt"

                os.system(f"perl {merci_script} -p {wd}/Sequence_1 -i {pos_motif_file} -o {wd}/merci_p.txt")
                os.system(f"perl {merci_script} -p {wd}/Sequence_1 -i {neg_motif_file} -o {wd}/merci_n.txt")

                # Process MERCI results
                MERCI_Processor_p(f"{wd}/merci_p.txt", f"{wd}/merci_output_p.csv", df_2)
                Merci_after_processing_p(f"{wd}/merci_output_p.csv", f"{wd}/merci_hybrid_p.csv")
                MERCI_Processor_n(f"{wd}/merci_n.txt", f"{wd}/merci_output_n.csv", df_2)
                Merci_after_processing_n(f"{wd}/merci_output_n.csv", f"{wd}/merci_hybrid_n.csv")
            
                # Read each CSV file into a separate DataFrame
                df_p = pd.read_csv(f"{wd}/merci_output_p.csv")
                df_n = pd.read_csv(f"{wd}/merci_output_n.csv")
                df_p = df_p.rename(columns={'Name': 'SeqID', 'Hits': 'PHits', 'Prediction': 'PPrediction'})
                df_n = df_n.rename(columns={'Name': 'SeqID', 'Hits': 'NHits', 'Prediction': 'NPrediction'})
                df_hp = pd.read_csv(f"{wd}/merci_hybrid_p.csv")
                df_hn = pd.read_csv(f"{wd}/merci_hybrid_n.csv")
                # Merge the DataFrames on 'SeqID'
                df_merged = df_p.merge(df_n, on='SeqID', how='outer')
                df4_selected = df_merged.iloc[:, [0, 1, 3]]

                # Define prediction function
                def determine_prediction(row):
                    if row['PHits'] == 0 and row['NHits'] == 0:
                        return 'Non-modulator'
                    elif row['PHits'] > row['NHits']:
                        return 'Modulator'
                    elif row['PHits'] < row['NHits']:
                        return 'Non-modulator'
                    elif row['PHits'] == row['NHits']:
                        return 'Modulator'
                    else:
                        return 'NA'

                # Predict and save results
                df4_selected_copy = df4_selected.copy()
                df4_selected_copy['Prediction'] = df4_selected_copy.apply(determine_prediction, axis=1)
                df4_selected_copy.columns = ["SeqID", 'Positive Hits', 'Negative Hits', "Prediction"]
                df4_selected_copy.to_csv(result_filename, index=None)

                if dplay == 1:
                    df4_selected_copy = df4_selected_copy.loc[df4_selected_copy.Prediction == "Modulator"]
                    print(df4_selected_copy)
                elif dplay == 2:
                    print(df4_selected_copy)

                # Clean up temporary files
                temp_files = [
                'final_output', 'merci_hybrid_p.csv', 'merci_hybrid_n.csv',
                'merci_output_p.csv', 'merci_output_n.csv',
                'merci_p.txt', 'merci_n.txt', 'out22', 'Sequence_1'
                ]
                for file in temp_files:
                    if os.path.exists(file):
                        os.remove(file)

                    #=================================== K+ ==================================
            elif Channel == 2:
                print('\n======= You are using the Motif Scanning module of IonNTxPred. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
                print('==== Scanning through K ion channel: Processing sequences please wait ...')
                df_2, dfseq = readseq(Sequence)
                df1 = lenchk(dfseq)
                merci_script = f"{nf_path}/merci/MERCI_motif_locator.pl"
                pos_motif_file = f"{nf_path}/motif/K/pos_motif.txt"
                neg_motif_file = f"{nf_path}/motif/K/neg_motif.txt"

                os.system(f"perl {merci_script} -p {wd}/Sequence_1 -i {pos_motif_file} -o {wd}/merci_p.txt")
                os.system(f"perl {merci_script} -p {wd}/Sequence_1 -i {neg_motif_file} -o {wd}/merci_n.txt")

                # Process MERCI results
                MERCI_Processor_p(f"{wd}/merci_p.txt", f"{wd}/merci_output_p.csv", df_2)
                Merci_after_processing_p(f"{wd}/merci_output_p.csv", f"{wd}/merci_hybrid_p.csv")
                MERCI_Processor_n(f"{wd}/merci_n.txt", f"{wd}/merci_output_n.csv", df_2)
                Merci_after_processing_n(f"{wd}/merci_output_n.csv", f"{wd}/merci_hybrid_n.csv")
            
                # Read each CSV file into a separate DataFrame
                df_p = pd.read_csv(f"{wd}/merci_output_p.csv")
                df_n = pd.read_csv(f"{wd}/merci_output_n.csv")
                df_p = df_p.rename(columns={'Name': 'SeqID', 'Hits': 'PHits', 'Prediction': 'PPrediction'})
                df_n = df_n.rename(columns={'Name': 'SeqID', 'Hits': 'NHits', 'Prediction': 'NPrediction'})
                df_hp = pd.read_csv(f"{wd}/merci_hybrid_p.csv")
                df_hn = pd.read_csv(f"{wd}/merci_hybrid_n.csv")
                # Merge the DataFrames on 'SeqID'
                df_merged = df_p.merge(df_n, on='SeqID', how='outer')
                df4_selected = df_merged.iloc[:, [0, 1, 3]]

                # Define prediction function
                def determine_prediction(row):
                    if row['PHits'] == 0 and row['NHits'] == 0:
                        return 'Non-modulator'
                    elif row['PHits'] > row['NHits']:
                        return 'Modulator'
                    elif row['PHits'] < row['NHits']:
                        return 'Non-modulator'
                    elif row['PHits'] == row['NHits']:
                        return 'Modulator'
                    else:
                        return 'NA'

                # Predict and save results
                df4_selected_copy = df4_selected.copy()
                df4_selected_copy['Prediction'] = df4_selected_copy.apply(determine_prediction, axis=1)
                df4_selected_copy.columns = ["SeqID", 'Positive Hits', 'Negative Hits', "Prediction"]
                df4_selected_copy.to_csv(f"{wd}/{result_filename}", index=None)

                if dplay == 1:
                    df4_selected_copy = df4_selected_copy.loc[df4_selected_copy.Prediction == "Modulator"]
                    print(df4_selected_copy)
                elif dplay == 2:
                    print(df4_selected_copy)

                # Clean up temporary files
                temp_files = [
                'final_output', 'merci_hybrid_p.csv', 'merci_hybrid_n.csv',
                'merci_output_p.csv', 'merci_output_n.csv',
                'merci_p.txt', 'merci_n.txt', 'out22', 'Sequence_1'
                ]
                for file in temp_files:
                    if os.path.exists(file):
                        os.remove(file)
            
                    #=================================== Ca2+ ==================================
            elif Channel == 3:
                print('\n======= You are using the Motif Scanning module of IonNTxPred. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
                print('==== Scanning through Ca ion channel: Processing sequences please wait ...')
                df_2, dfseq = readseq(Sequence)
                df1 = lenchk(dfseq)
                merci_script = f"{nf_path}/merci/MERCI_motif_locator.pl"
                pos_motif_file = f"{nf_path}/motif/Ca/pos_motif.txt"
                neg_motif_file = f"{nf_path}/motif/Ca/neg_motif.txt"

                os.system(f"perl {merci_script} -p {wd}/Sequence_1 -i {pos_motif_file} -o {wd}/merci_p.txt")
                os.system(f"perl {merci_script} -p {wd}/Sequence_1 -i {neg_motif_file} -o {wd}/merci_n.txt")

                # Process MERCI results
                MERCI_Processor_p(f"{wd}/merci_p.txt", f"{wd}/merci_output_p.csv", df_2)
                Merci_after_processing_p(f"{wd}/merci_output_p.csv", f"{wd}/merci_hybrid_p.csv")
                MERCI_Processor_n(f"{wd}/merci_n.txt", f"{wd}/merci_output_n.csv", df_2)
                Merci_after_processing_n(f"{wd}/merci_output_n.csv", f"{wd}/merci_hybrid_n.csv")
            
                # Read each CSV file into a separate DataFrame
                df_p = pd.read_csv(f"{wd}/merci_output_p.csv")
                df_n = pd.read_csv(f"{wd}/merci_output_n.csv")
                df_p = df_p.rename(columns={'Name': 'SeqID', 'Hits': 'PHits', 'Prediction': 'PPrediction'})
                df_n = df_n.rename(columns={'Name': 'SeqID', 'Hits': 'NHits', 'Prediction': 'NPrediction'})
                df_hp = pd.read_csv(f"{wd}/merci_hybrid_p.csv")
                df_hn = pd.read_csv(f"{wd}/merci_hybrid_n.csv")
                # Merge the DataFrames on 'SeqID'
                df_merged = df_p.merge(df_n, on='SeqID', how='outer')
                df4_selected = df_merged.iloc[:, [0, 1, 3]]

                # Define prediction function
                def determine_prediction(row):
                    if row['PHits'] == 0 and row['NHits'] == 0:
                        return 'Non-modulator'
                    elif row['PHits'] > row['NHits']:
                        return 'Modulator'
                    elif row['PHits'] < row['NHits']:
                        return 'Non-modulator'
                    elif row['PHits'] == row['NHits']:
                        return 'Modulator'
                    else:
                        return 'NA'

                # Predict and save results
                df4_selected_copy = df4_selected.copy()
                df4_selected_copy['Prediction'] = df4_selected_copy.apply(determine_prediction, axis=1)
                df4_selected_copy.columns = ["SeqID", 'Positive Hits', 'Negative Hits', "Prediction"]
                df4_selected_copy.to_csv(f"{wd}/{result_filename}", index=None)

                if dplay == 1:
                    df4_selected_copy = df4_selected_copy.loc[df4_selected_copy.Prediction == "Modulator"]
                    print(df4_selected_copy)
                elif dplay == 2:
                    print(df4_selected_copy)

                # Clean up temporary files
                temp_files = [
                'final_output', 'merci_hybrid_p.csv', 'merci_hybrid_n.csv',
                'merci_output_p.csv', 'merci_output_n.csv',
                'merci_p.txt', 'merci_n.txt', 'out22', 'Sequence_1'
                ]
                for file in temp_files:
                    if os.path.exists(file):
                        os.remove(file)       

                    #=================================== Other ==================================
            elif Channel == 4:
                print('\n======= You are using the Motif Scanning module of IonNTxPred. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
                print('==== Scanning through Other ion channel: Processing sequences please wait ...')
                df_2, dfseq = readseq(Sequence)
                df1 = lenchk(dfseq)
                merci_script = f"{nf_path}/merci/MERCI_motif_locator.pl"
                pos_motif_file = f"{nf_path}/motif/Other/pos_motif.txt"
                neg_motif_file = f"{nf_path}/motif/Other/neg_motif.txt"

                os.system(f"perl {merci_script} -p {wd}/Sequence_1 -i {pos_motif_file} -o {wd}/merci_p.txt")
                os.system(f"perl {merci_script} -p {wd}/Sequence_1 -i {neg_motif_file} -o {wd}/merci_n.txt")

                # Process MERCI results
                MERCI_Processor_p(f"{wd}/merci_p.txt", f"{wd}/merci_output_p.csv", df_2)
                Merci_after_processing_p(f"{wd}/merci_output_p.csv", f"{wd}/merci_hybrid_p.csv")
                MERCI_Processor_n(f"{wd}/merci_n.txt", f"{wd}/merci_output_n.csv", df_2)
                Merci_after_processing_n(f"{wd}/merci_output_n.csv", f"{wd}/merci_hybrid_n.csv")
            
                # Read each CSV file into a separate DataFrame
                df_p = pd.read_csv(f"{wd}/merci_output_p.csv")
                df_n = pd.read_csv(f"{wd}/merci_output_n.csv")
                df_p = df_p.rename(columns={'Name': 'SeqID', 'Hits': 'PHits', 'Prediction': 'PPrediction'})
                df_n = df_n.rename(columns={'Name': 'SeqID', 'Hits': 'NHits', 'Prediction': 'NPrediction'})
                df_hp = pd.read_csv(f"{wd}/merci_hybrid_p.csv")
                df_hn = pd.read_csv(f"{wd}/merci_hybrid_n.csv")
                # Merge the DataFrames on 'SeqID'
                df_merged = df_p.merge(df_n, on='SeqID', how='outer')
                df4_selected = df_merged.iloc[:, [0, 1, 3]]

                # Define prediction function
                def determine_prediction(row):
                    if row['PHits'] == 0 and row['NHits'] == 0:
                        return 'Non-modulator'
                    elif row['PHits'] > row['NHits']:
                        return 'Modulator'
                    elif row['PHits'] < row['NHits']:
                        return 'Non-modulator'
                    elif row['PHits'] == row['NHits']:
                        return 'Modulator'
                    else:
                        return 'NA'

                # Predict and save results
                df4_selected_copy = df4_selected.copy()
                df4_selected_copy['Prediction'] = df4_selected_copy.apply(determine_prediction, axis=1)
                df4_selected_copy.columns = ["SeqID", 'Positive Hits', 'Negative Hits', "Prediction"]
                df4_selected_copy.to_csv(f"{wd}/{result_filename}", index=None)

                if dplay == 1:
                    df4_selected_copy = df4_selected_copy.loc[df4_selected_copy.Prediction == "Modulator"]
                    print(df4_selected_copy)
                elif dplay == 2:
                    print(df4_selected_copy)

                # Clean up temporary files
                temp_files = [
                'final_output', 'merci_hybrid_p.csv', 'merci_hybrid_n.csv',
                'merci_output_p.csv', 'merci_output_n.csv',
                'merci_p.txt', 'merci_n.txt', 'out22', 'Sequence_1'
                ]
                for file in temp_files:
                    if os.path.exists(file):
                        os.remove(file)


    #======================= Blast Search Module starts from here =====================
    if Job == 5:
        #=================================== Na+ ==================================        
            if Channel == 1:
                print('\n======= Thanks for using BLAST scan module for Na Channel prediction. Your results will be stored in file :', result_filename, ' =====\n')

                # Read sequences
                df_2, dfseq = readseq(Sequence)
                df1 = lenchk(dfseq)
                df1.columns = ["Seq"]  # Ensure column name exists

                # Combine sequence IDs and sequences into one DataFrame
                df_fasta = df_2.copy()
                df_fasta["Seq"] = df1["Seq"]
                df_fasta.columns = ['SeqID', 'Seq']

                # Temp FASTA filename
                filename = os.path.join(wd, str(uuid.uuid4()) + ".fasta")
                # print("df_fasta columns:", df_fasta.columns.tolist())

                # Write valid FASTA file
                write_fasta(df_fasta, filename)

                # BLAST setup
                blast_bin_path = f"{nf_path}/blast_binaries/linux/blastp"
                blast_db_path  = f"{nf_path}/BLAST/na_all_db/na_train"

                # Output file path
                blast_output_file = os.path.join(wd, "RES_na_channel.out")

                # Run BLAST
                cmd = [
                    blast_bin_path,
                    "-query", filename,
                    "-db", blast_db_path,
                    "-out", blast_output_file,
                    "-evalue", "0.001",
                    "-outfmt", "6"
                ]

                print("[INFO] Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)

                # Parse BLAST output
                df44 = BLAST_search(blast_output_file, df_2)
                df44['Sequence'] = df1["Seq"]
                df44 = df44[['Seq ID', 'Sequence', 'BLAST Score', 'Prediction']]

                # Optional filter
                if dplay == 1:
                    df44 = df44.loc[df44.Prediction == "Modulator"]
                elif dplay == 2:
                    print(df44)

                df44 = round(df44, 3)
                df44.to_csv(f"{wd}/{result_filename}", index=None)

                # Clean temporary files
                os.remove(filename)         
                #os.remove("Sequence_1")     
                os.remove(blast_output_file) 

                    #=================================== K+ ==================================
            elif Channel == 2:
                print('\n======= Thanks for using BLAST scan module for K Channel prediction. Your results will be stored in file :', result_filename, ' =====\n')

                # Read sequences
                df_2, dfseq = readseq(Sequence)
                df1 = lenchk(dfseq)
                df1.columns = ["Seq"]  # Ensure column name exists

                # Combine sequence IDs and sequences into one DataFrame
                df_fasta = df_2.copy()
                df_fasta["Seq"] = df1["Seq"]
                df_fasta.columns = ['SeqID', 'Seq']

                # Temp FASTA filename
                filename = os.path.join(wd, str(uuid.uuid4()) + ".fasta")
                # print("df_fasta columns:", df_fasta.columns.tolist())

                # Write valid FASTA file
                write_fasta(df_fasta, filename)

                # BLAST setup
                blast_bin_path = f"{nf_path}/blast_binaries/linux/blastp"
                blast_db_path  = f"{nf_path}/BLAST/k_all_db/k_train"

                # Output file path
                blast_output_file = os.path.join(wd, "RES_k_channel.out")

                # Run BLAST
                cmd = [
                    blast_bin_path,
                    "-query", filename,
                    "-db", blast_db_path,
                    "-out", blast_output_file,
                    "-evalue", "0.001",
                    "-outfmt", "6"
                ]

                print("[INFO] Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)

                # Parse BLAST output
                df44 = BLAST_search(blast_output_file, df_2)
                df44['Sequence'] = df1["Seq"]
                df44 = df44[['Seq ID', 'Sequence', 'BLAST Score', 'Prediction']]

                # Optional filter
                if dplay == 1:
                    df44 = df44.loc[df44.Prediction == "Modulator"]
                elif dplay == 2:
                    print(df44)

                df44 = round(df44, 3)
                df44.to_csv(f"{wd}/{result_filename}", index=None)

                # Clean temporary files
                os.remove(filename)         
                #os.remove("Sequence_1")     
                os.remove(blast_output_file) 

                    #=================================== Ca2+ ==================================
            elif Channel == 3:
                print('\n======= Thanks for using BLAST scan module for Ca Channel prediction. Your results will be stored in file :', result_filename, ' =====\n')

                # Read sequences
                df_2, dfseq = readseq(Sequence)
                df1 = lenchk(dfseq)
                df1.columns = ["Seq"]  # Ensure column name exists

                # Combine sequence IDs and sequences into one DataFrame
                df_fasta = df_2.copy()
                df_fasta["Seq"] = df1["Seq"]
                df_fasta.columns = ['SeqID', 'Seq']

                # Temp FASTA filename
                filename = os.path.join(wd, str(uuid.uuid4()) + ".fasta")
                # print("df_fasta columns:", df_fasta.columns.tolist())

                # Write valid FASTA file
                write_fasta(df_fasta, filename)

                # BLAST setup
                blast_bin_path = f"{nf_path}/blast_binaries/linux/blastp"
                blast_db_path  = f"{nf_path}/BLAST/ca_all_db/ca_train"

                # Output file path
                blast_output_file = os.path.join(wd, "RES_ca_channel.out")

                # Run BLAST
                cmd = [
                    blast_bin_path,
                    "-query", filename,
                    "-db", blast_db_path,
                    "-out", blast_output_file,
                    "-evalue", "0.001",
                    "-outfmt", "6"
                ]

                print("[INFO] Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)

                # Parse BLAST output
                df44 = BLAST_search(blast_output_file, df_2)
                df44['Sequence'] = df1["Seq"]
                df44 = df44[['Seq ID', 'Sequence', 'BLAST Score', 'Prediction']]

                # Optional filter
                if dplay == 1:
                    df44 = df44.loc[df44.Prediction == "Modulator"]
                elif dplay == 2:
                    print(df44)

                df44 = round(df44, 3)
                df44.to_csv(f"{wd}/{result_filename}", index=None)

                # Clean temporary files
                os.remove(filename)         
                #os.remove("Sequence_1")     
                os.remove(blast_output_file)   

                    #=================================== Other ==================================
            elif Channel == 4:
                print('\n======= Thanks for using BLAST scan module for K Channel prediction. Your results will be stored in file :', result_filename, ' =====\n')

                # Read sequences
                df_2, dfseq = readseq(Sequence)
                df1 = lenchk(dfseq)
                df1.columns = ["Seq"]  # Ensure column name exists

                # Combine sequence IDs and sequences into one DataFrame
                df_fasta = df_2.copy()
                df_fasta["Seq"] = df1["Seq"]
                df_fasta.columns = ['SeqID', 'Seq']

                # Temp FASTA filename
                filename = os.path.join(wd, str(uuid.uuid4()) + ".fasta")
                # print("df_fasta columns:", df_fasta.columns.tolist())

                # Write valid FASTA file
                write_fasta(df_fasta, filename)

                # BLAST setup
                blast_bin_path = f"{nf_path}/blast_binaries/linux/blastp"
                blast_db_path  = f"{nf_path}/BLAST/other_all_db/other_train"

                # Output file path
                blast_output_file = os.path.join(wd, "RES_other_channel.out")

                # Run BLAST
                cmd = [
                    blast_bin_path,
                    "-query", filename,
                    "-db", blast_db_path,
                    "-out", blast_output_file,
                    "-evalue", "0.001",
                    "-outfmt", "6"
                ]

                print("[INFO] Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)

                # Parse BLAST output
                df44 = BLAST_search(blast_output_file, df_2)
                df44['Sequence'] = df1["Seq"]
                df44 = df44[['Seq ID', 'Sequence', 'BLAST Score', 'Prediction']]

                # Optional filter
                if dplay == 1:
                    df44 = df44.loc[df44.Prediction == "Modulator"]
                elif dplay == 2:
                    print(df44)

                df44 = round(df44, 3)
                df44.to_csv(f"{wd}/{result_filename}", index=None)

                # Clean temporary files
                os.remove(filename)         
                #os.remove("Sequence_1")     
                os.remove(blast_output_file) 


                                        
    print('\n\n🎉 ======= Thank You for Using IonNTxPred! ======= 🎉')
    print('🙏 We hope this tool contributed to your research on ion channel modulating proteins.')
    print('\n📖 If you found IonNTxPred useful, please cite us in your work:')
    print('    ➤ Rathore et al., *IonNTxPred: LLM-based Prediction and Designing of Ion Channel Impairing Proteins*, 2025.')
    print('\n🔗 Useful Links:')
    print('    🌐 Web Server : https://webs.iiitd.edu.in/raghava/ionntxpred/')
    print('    💻 GitHub     : https://github.com/raghavagps/IonNTxPred')
    print('    🤗 HuggingFace: https://huggingface.co/raghavagps-group/IonNTxPred\n')

if __name__ == "__main__":
    main()        


