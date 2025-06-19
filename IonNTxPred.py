#########################################################################
# IonNTxpred is developed for identifying modulator and non-modulator  #
# peptides that selectively modulate sodium, potassium, calcium, and    #
# other ion channels. It is developed by Prof G. P. S. Raghava's group. #                                                  
# Please cite : IonNTxpred                                             #
#########################################################################

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
from collections import defaultdict
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='Please provide following arguments. Please make the suitable changes in the envfile provided in the folder.') 

## Read Arguments from command
parser.add_argument("-i", "--input", type=str, required=True, help="Input: Peptide sequence in FASTA format or single sequence per line in single letter code")
parser.add_argument("-o", "--output",type=str, default="output.csv", help="Output: File for saving results by default output.csv")
parser.add_argument("-t","--threshold", type=float, default=0.5, help="Threshold: Value between 0 to 1 by default 0.5")
parser.add_argument("-j", "--job", type=int, choices=[1, 2, 3, 4], default=1, help="Job Type: 1: Prediction, 2: Design, 3: Protein Scanning, 4: Motif Scanning")
parser.add_argument("-c", "--channel", type=int, default=1, choices=[1, 2, 3, 4], help="Ion channel type: 1: Na+, 2: K+, 3: Ca+, 4: Other")
parser.add_argument("-m", "--model", type=int, default= 1, choices=[1, 2], help="Model: 1: ESM2-t12, 2: Hybrid (ESM2-t12 + MERCI)")
parser.add_argument("-w","--winleng", type=int, choices =range(8, 21), default=8, help="Window Length: 8 to 20 (scan mode only), by default 8")
parser.add_argument("-wd", "--working", type=str, default=os.getcwd(), help="Working directory for intermediate files (optional).")
parser.add_argument("-d","--display", type=int, choices = [1,2], default=2, help="Display: 1:Modulating ion, 2: All peptides, by default 2")


args = parser.parse_args()

nf_path = os.path.dirname(__file__)

################################### Model Calling ##########################################
import argparse
import os
import zipfile
import urllib.request
from tqdm.auto import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

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
        f.write("SeqID,Seq,ML Score,Prediction\n")
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


def hybrid(ML_output,name1, seq, merci_output_p, merci_output_n,threshold,final_output):
    if isinstance(seq, str):
        seq = [seq]  # convert to a list
    df6_2 = pd.DataFrame(seq)

    df6_3 = ML_output
    df6_2 = pd.DataFrame(seq)
    df6_1 = pd.DataFrame(name1)
    df5 = pd.read_csv(merci_output_p, dtype={'Subject': object, 'MERCI Score': np.float64})
    df4 = pd.read_csv(merci_output_n, dtype={'Subject': object, 'MERCI Score': np.float64})
    df6 = pd.concat([df6_1,df6_2, df6_3],axis=1)
    df6 = df6.iloc[:, :3]
    df6.columns = ['Subject','Sequence','ML Score']
    df6['Subject'] = df6['Subject'].str.replace('>','')
    df7 = pd.merge(df6,df5, how='outer',on='Subject')
    df8 = pd.merge(df7,df4, how='outer',on='Subject')
    df8.fillna(0, inplace=True)
    cols = ['ML Score', 'MERCI Score (+ve)', 'MERCI Score (-ve)']
    df8[cols] = df8[cols].apply(pd.to_numeric, errors='coerce')  # Convert to float, NaNs if invalid
    df8['Hybrid Score'] = df8[cols].sum(axis=1)
    df8 = df8.rename(columns={'Subject': 'SeqID', 'Sequence': 'Mutant Sequence', 'ML Score': 'ESM Score'})
    df8 = df8.round(4)
    ee = []
    for i in range(0,len(df8)):
        if df8['Hybrid Score'][i] > float(threshold):
            ee.append('Modulator')
        else:
            ee.append('Non-modulator')
    df8['Prediction'] = ee
    df8.to_csv(final_output, index=None)


print('\n')
print('##########################################################################################')
print('#                                                                                        #')
print('#  🧠 IonNTxPred: A Powerful Tool to Predict Ion Channel Modulators                      #')
print('#                                                                                        #')
print('#  This program is developed to identify **modulators** and **non-modulators** of        #')
print('#  ion channels—specifically targeting sodium (Na⁺), potassium (K⁺), calcium (Ca²⁺),     #')
print('#  and chloride (Cl⁻) channels.                                                          #')
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
    Threshold = 0.5
else:
    Threshold= float(args.threshold)

# Job Type
if args.job is None:
    Job = 1
else:
    Job = args.job

# Channel Type
if args.channel is None:
    Channel = 1
else:
    Channel = args.channel

# Model
if args.model is None:
    Model = 1
else:
    Model = int(args.model)

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
CM.to_csv(f"{wd}/Sequence_1",header=False,index=None,sep="\n")
f.close()

            
            #======================= Prediction Module starts from here =====================
if Job == 1:
    print(f'\n======= You are using the Prediction Module of IonNTxPred. Your results will be stored in file: {wd}/{result_filename}\n')
    print("\n==================== Running Prediction Module ====================")

    channel_configs = {
        "na": {
            "model_path": f"{nf_path}/Model/saved_model_t12_na_train_on_complete_data",
            "label": "Na"
        },
        "k": {
            "model_path": f"{nf_path}/Model/saved_model_t12_k_train_on_complete_data",
            "label": "K"
        },
        "ca": {
            "model_path": f"{nf_path}/Model/saved_model_t12_ca_train_on_complete_data",
            "label": "Ca"
        },
        "other": {
            "model_path": f"{nf_path}/Model/saved_model_t12_other_train_on_complete_data",
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
            df_model1['SeqID'] = df_model1['SeqID'].str.replace('>', '')
            df_final = df_model1

            # Clean up temporary file
            os.remove(out_file1)

        elif Model == 2:
            out_file2 = f"{wd}/model2_output_{channel}.csv"
            run_esm_model(seq, seqid_1, out_file2, Threshold)

            df_esm2 = pd.read_csv(out_file2)
            hybrid_col = f"{cfg['label']}_hybrid"
            df_esm2.rename(columns={df_esm2.columns[2]: hybrid_col}, inplace=True)
            seq_pred = df_esm2[hybrid_col]

            if isinstance(seq, pd.DataFrame):
                seq = seq.iloc[:, 0].tolist()

            merci_p_txt = f"{wd}/merci_p_{channel}.txt"
            merci_n_txt = f"{wd}/merci_n_{channel}.txt"
            merci_output_p = f"{wd}/merci_output_p_{channel}.csv"
            merci_output_n = f"{wd}/merci_output_n_{channel}.csv"
            merci_hybrid_p = f"{wd}/merci_hybrid_p_{channel}.csv"
            merci_hybrid_n = f"{wd}/merci_hybrid_n_{channel}.csv"
            final_output_file = f"{wd}/final_output_{channel}.csv"

            merci_script = f"{nf_path}/merci/MERCI_motif_locator.pl"
            pos_motif_file = f"{nf_path}/motif/{cfg['label']}/pos_motif.txt"
            neg_motif_file = f"{nf_path}/motif/{cfg['label']}/neg_motif.txt"

            os.system(f"perl {merci_script} -p {wd}/Sequence_1 -i {pos_motif_file} -o {merci_p_txt}")
            os.system(f"perl {merci_script} -p {wd}/Sequence_1 -i {neg_motif_file} -o {merci_n_txt}")

            MERCI_Processor_p(merci_p_txt, merci_output_p, seqid)
            Merci_after_processing_p(merci_output_p, merci_hybrid_p)
            MERCI_Processor_n(merci_n_txt, merci_output_n, seqid)
            Merci_after_processing_n(merci_output_n, merci_hybrid_n)

            hybrid(seq_pred, seqid_1, seq, merci_hybrid_p, merci_hybrid_n, Threshold, final_output_file)
            df_hybrid = pd.read_csv(final_output_file)
            df_hybrid[hybrid_col] = df_hybrid['Hybrid Score'].clip(0, 1)
            df_hybrid = df_hybrid[['SeqID', hybrid_col, 'Prediction']]
            df_hybrid['SeqID'] = df_hybrid['SeqID'].str.replace('>', '')
            df_hybrid.columns = ['SeqID', hybrid_col, f"{hybrid_col}_Prediction"]

            df_final = pd.DataFrame({'SeqID': [s.replace('>', '') for s in seqid_1], 'Seq': seq})
            df_final = pd.merge(df_final, df_hybrid, on='SeqID')

            # Clean up all temporary files
            os.remove(out_file2)
            os.remove(merci_p_txt)
            os.remove(merci_n_txt)
            os.remove(merci_output_p)
            os.remove(merci_output_n)
            os.remove(merci_hybrid_p)
            os.remove(merci_hybrid_n)
            os.remove(final_output_file)

        channel_results.append(df_final)

    # Merge channel results
    from functools import reduce
    final_df = reduce(lambda left, right: pd.merge(left, right, on=['SeqID', 'Seq']), channel_results)
    final_df.to_csv(result_filename, index=False)

    # Final display
    if dplay == 1:
        print(final_df[
            (final_df.get('Na_Prediction') == "Modulator") |
            (final_df.get('K_Prediction') == "Modulator") |
            (final_df.get('Ca_Prediction') == "Modulator") |
            (final_df.get('Other_Prediction') == "Modulator")
        ])
    elif dplay == 2:
        print(final_df)

    # Clean up shared files
    os.remove(f'{wd}/Sequence_1')

        #======================= Design Module starts from here =====================
if Job == 2:

            #=================================== Na+ ================================== 
    if Channel == 1:
        if Model == 1:
            print('\n======= You are using the Design Module of IonNTxPred. Your results will be stored in file: 'f"{wd}/{result_filename}"'\n')
            print('==== Predicting Modulating Activity using ESM2-t12 model: Processing sequences, please wait ...')
            muts = all_mutants(seq, seqid_1)
            muts.to_csv(f'{wd}/muts.csv', index=None, header=None)  
            seq=muts['Seq'].tolist()
            seqid_1=muts['Mutant_ID'].tolist()

            # Load the tokenizer and model
            model_save_path = f"{nf_path}/Model/saved_model_t12_na_train_on_complete_data"
            tokenizer = AutoTokenizer.from_pretrained(model_save_path)
            model = EsmForSequenceClassification.from_pretrained(model_save_path)
            model.eval()
            # Run the ESM model
            run_esm_model(seq, seqid_1, result_filename, Threshold)
            df13 = pd.read_csv(result_filename)
            df13.columns = ['MutantID', 'Sequence', 'ESM Score', "Prediction"]
            seqid = pd.DataFrame(muts["SeqID"])
            df14 = pd.concat([seqid,df13],axis=1)
            df14['SeqID'] = df14['SeqID'].str.replace('>','')


            # Display results based on 'dplay'
            if dplay == 1:
                df15 = df14.loc[df14['Mutant Prediction'] == "Modulator"]
                print(df15)
            elif dplay == 2:
                df15 = df14
                print(df15)

            # Clean up temporary files 
            os.remove(f'{wd}/Sequence_1')
            os.remove(f'{wd}/muts.csv')  
        
        elif Model == 2:
            print(f'\n======= You are using the Design Module of IonNTxPred. Your results will be stored in file: {wd}/{result_filename} =====\n')
            print('==== Predicting Modulating Activity using Hybrid model: Processing sequences please wait ...')

            # Generate mutants and save
            muts = all_mutants(seq, seqid_1)
            muts.columns = ['SeqID', 'MutantID', 'seq']
            muts.to_csv(f'{wd}/muts.csv', index=False)

            # Read back mutants
            muts_df = pd.read_csv(f'{wd}/muts.csv')
            seq = muts_df['seq'].tolist()
            seqid_1 = muts_df['MutantID'].tolist()

            # Prepare ESM input
            with open(f'{wd}/out_len_mut', 'w') as f:
                for s in seq:
                    f.write(s + '\n')
            print()
            # Load and run ESM model
            model_save_path = f"{nf_path}/Model/saved_model_t12_na_train_on_complete_data"
            tokenizer = AutoTokenizer.from_pretrained(model_save_path)
            model = EsmForSequenceClassification.from_pretrained(model_save_path)
            model.eval()

            run_esm_model(seq, seqid_1, result_filename, Threshold)
            ml_output_file = pd.read_csv(result_filename)
            seq_pred = ml_output_file.iloc[:, 2]

            # Prepare MERCI input
            muts_df[['MutantID', 'seq']].to_csv(f"{wd}/Sequence_1", index=False, header=False)

            # Modify Sequence_1 format to FASTA-like format
            with open(f"{wd}/Sequence_1", "r") as infile:
                lines = infile.readlines()

            with open(f"{wd}/Sequence_1", "w") as outfile:
                for line in lines:
                    parts = line.strip().split(",")
                    if len(parts) == 2:
                        outfile.write(f">{parts[0]}\n{parts[1]}\n")
 
            # Run MERCI
            merci_script = f"{nf_path}/merci/MERCI_motif_locator.pl"
            pos_motif_file = f"{nf_path}/motif/Na/pos_motif.txt"
            neg_motif_file = f"{nf_path}/motif/Na/neg_motif.txt"

            os.system(f"perl {merci_script} -p {wd}/Sequence_1 -i {pos_motif_file} -o {wd}/merci_p.txt")
            os.system(f"perl {merci_script} -p {wd}/Sequence_1 -i {neg_motif_file} -o {wd}/merci_n.txt")
            MERCI_Processor_p(f"{wd}/merci_p.txt", f"{wd}/merci_output_p.csv", seqid_1)
            Merci_after_processing_p(f"{wd}/merci_output_p.csv", f"{wd}/merci_hybrid_p.csv")
            MERCI_Processor_n(f"{wd}/merci_n.txt", f"{wd}/merci_output_n.csv", seqid_1)
            Merci_after_processing_n(f"{wd}/merci_output_n.csv", f"{wd}/merci_hybrid_n.csv")

            # Run Hybrid prediction
            hybrid(seq_pred, seqid_1, seq, f"{wd}/merci_hybrid_p.csv", f"{wd}/merci_hybrid_n.csv", Threshold, f"{wd}/final_output")
            df44 = pd.read_csv(f"{wd}/final_output")
            df44['Hybrid Score'] = df44['Hybrid Score'].clip(0, 1)
            
            # Rename column for clarity
            df44.rename(columns={'SeqID': 'MutantID', 'Mutant Sequence': 'Sequence'}, inplace=True)

            # Merge original SeqID for final output
            df44 = pd.merge(df44, muts_df[['MutantID', 'SeqID']], on='MutantID', how='left')
            # Reorder columns
            df44 = df44[['SeqID', 'MutantID', 'Sequence', 'ESM Score', 'MERCI Score (+ve)', 'MERCI Score (-ve)', 'Hybrid Score', 'Prediction']]
            df44['SeqID'] = df44['SeqID'].str.replace('>','')
            # Display
            if dplay == 1:
                print(df44[df44['Prediction'] == "Modulator"])
            elif dplay == 2:
                print(df44)

            # Save
            final_df = round(df44, 4)
            final_df.to_csv(result_filename, index=False)

            # Clean up
            temp_files = [
                'final_output', 'merci_hybrid_p.csv', 'merci_hybrid_n.csv',
                'merci_output_p.csv', 'merci_output_n.csv',
                'merci_p.txt', 'merci_n.txt', 'out_m',
                'Sequence_1', 'out_len_mut', 'out22', 'out33'
            ]
            for file in temp_files:
                path = f"{wd}/{file}"
                if os.path.exists(path):
                    os.remove(path)
            

#=================================== K+ ================================== 
    elif Channel == 2:
        if Model == 1:
            print('\n======= You are using the Design Module of IonNTxPred. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
            print('==== Predicting Modulating Activity using ESM2-t12 model: Processing sequences please wait ...')
            muts = all_mutants(seq, seqid_1)
            muts.to_csv(f'{wd}/muts.csv', index=None, header=None)  
            seq=muts['Seq'].tolist()
            seqid_1=muts['Mutant_ID'].tolist()

            # Load the tokenizer and model
            model_save_path = f"{nf_path}/Model/saved_model_t12_k_train_on_complete_data"
            tokenizer = AutoTokenizer.from_pretrained(model_save_path)
            model = EsmForSequenceClassification.from_pretrained(model_save_path)
            model.eval()
            # Run the ESM model
            run_esm_model(seq, seqid_1, result_filename, Threshold)
            df13 = pd.read_csv(result_filename)
            df13.columns = ['MutantID', 'Sequence', 'ESM Score', "Prediction"]
            seqid = pd.DataFrame(muts["SeqID"])
            df14 = pd.concat([seqid,df13],axis=1)
            df14['SeqID'] = df14['SeqID'].str.replace('>','')


            # Display results based on 'dplay'
            if dplay == 1:
                df15 = df14.loc[df14['Mutant Prediction'] == "Modulator"]
                print(df15)
            elif dplay == 2:
                df15 = df14
                print(df15)

            # Clean up temporary files 
            os.remove(f'{wd}/Sequence_1')
            os.remove(f'{wd}/muts.csv')
            
            
        elif Model == 2: 
            print('\n======= You are using the Design Module of IonNTxPred. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
            print('==== Predicting Modulating Activity using Hybrid model: Processing sequences please wait ...')

            # Generate mutants and save
            muts = all_mutants(seq, seqid_1)
            muts.columns = ['SeqID', 'MutantID', 'seq']
            muts.to_csv(f'{wd}/muts.csv', index=False)

            # Read back mutants
            muts_df = pd.read_csv(f'{wd}/muts.csv')
            seq = muts_df['seq'].tolist()
            seqid_1 = muts_df['MutantID'].tolist()

            # Prepare ESM input
            with open(f'{wd}/out_len_mut', 'w') as f:
                for s in seq:
                    f.write(s + '\n')
            print()
            # Load and run ESM model
            model_save_path = f"{nf_path}/Model/saved_model_t12_k_train_on_complete_data"
            tokenizer = AutoTokenizer.from_pretrained(model_save_path)
            model = EsmForSequenceClassification.from_pretrained(model_save_path)
            model.eval()

            run_esm_model(seq, seqid_1, result_filename, Threshold)
            ml_output_file = pd.read_csv(result_filename)
            seq_pred = ml_output_file.iloc[:, 2]

            # Prepare MERCI input
            muts_df[['MutantID', 'seq']].to_csv(f"{wd}/Sequence_1", index=False, header=False)

            # Modify Sequence_1 format to FASTA-like format
            with open(f"{wd}/Sequence_1", "r") as infile:
                lines = infile.readlines()

            with open(f"{wd}/Sequence_1", "w") as outfile:
                for line in lines:
                    parts = line.strip().split(",")
                    if len(parts) == 2:
                        outfile.write(f">{parts[0]}\n{parts[1]}\n")
 
            # Run MERCI
            merci_script = f"{nf_path}/merci/MERCI_motif_locator.pl"
            pos_motif_file = f"{nf_path}/motif/K/pos_motif.txt"
            neg_motif_file = f"{nf_path}/motif/K/neg_motif.txt"

            os.system(f"perl {merci_script} -p {wd}/Sequence_1 -i {pos_motif_file} -o {wd}/merci_p.txt")
            os.system(f"perl {merci_script} -p {wd}/Sequence_1 -i {neg_motif_file} -o {wd}/merci_n.txt")
            MERCI_Processor_p(f"{wd}/merci_p.txt", f"{wd}/merci_output_p.csv", seqid_1)
            Merci_after_processing_p(f"{wd}/merci_output_p.csv", f"{wd}/merci_hybrid_p.csv")
            MERCI_Processor_n(f"{wd}/merci_n.txt", f"{wd}/merci_output_n.csv", seqid_1)
            Merci_after_processing_n(f"{wd}/merci_output_n.csv", f"{wd}/merci_hybrid_n.csv")

            # Run Hybrid prediction
            hybrid(seq_pred, seqid_1, seq, f"{wd}/merci_hybrid_p.csv", f"{wd}/merci_hybrid_n.csv", Threshold, f"{wd}/final_output")
            df44 = pd.read_csv(f"{wd}/final_output")
            df44['Hybrid Score'] = df44['Hybrid Score'].clip(0, 1)
            
            # Rename column for clarity
            df44.rename(columns={'SeqID': 'MutantID', 'Mutant Sequence': 'Sequence'}, inplace=True)

            # Merge original SeqID for final output
            df44 = pd.merge(df44, muts_df[['MutantID', 'SeqID']], on='MutantID', how='left')
            # Reorder columns
            df44 = df44[['SeqID', 'MutantID', 'Sequence', 'ESM Score', 'MERCI Score (+ve)', 'MERCI Score (-ve)', 'Hybrid Score', 'Prediction']]
            df44['SeqID'] = df44['SeqID'].str.replace('>','')
            # Display
            if dplay == 1:
                print(df44[df44['Prediction'] == "Modulator"])
            elif dplay == 2:
                print(df44)

            # Save
            final_df = round(df44, 4)
            final_df.to_csv(result_filename, index=False)

            
            # # Clean up temporary files
            temp_files = [
                'final_output', 'merci_hybrid_p.csv', 'merci_hybrid_n.csv',
                'merci_output_p.csv', 'merci_output_n.csv',
                'merci_p.txt', 'merci_n.txt', 'out_m', 
                'Sequence_1', 'out_len_mut', 'out22', 'out33'
            ]
            for file in temp_files:
                if os.path.exists(file):
                    os.remove(file)

#=================================== Ca2+ ================================== 
    elif Channel == 3:
        if Model == 1:
            print('\n======= You are using the Design Module of IonNTxPred. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
            print('==== Predicting Modulating Activity using ESM2-t12 model: Processing sequences please wait ...')
            muts = all_mutants(seq, seqid_1)
            muts.to_csv(f'{wd}/muts.csv', index=None, header=None)  
            seq=muts['Seq'].tolist()
            seqid_1=muts['Mutant_ID'].tolist()

            # Load the tokenizer and model
            model_save_path = f"{nf_path}/Model/saved_model_t12_ca_train_on_complete_data"
            tokenizer = AutoTokenizer.from_pretrained(model_save_path)
            model = EsmForSequenceClassification.from_pretrained(model_save_path)
            model.eval()
            # Run the ESM model
            run_esm_model(seq, seqid_1, result_filename, Threshold)
            df13 = pd.read_csv(result_filename)
            df13.columns = ['MutantID', 'Sequence', 'ESM Score', "Prediction"]
            seqid = pd.DataFrame(muts["SeqID"])
            df14 = pd.concat([seqid,df13],axis=1)
            df14['SeqID'] = df14['SeqID'].str.replace('>','')


            # Display results based on 'dplay'
            if dplay == 1:
                df15 = df14.loc[df14['Mutant Prediction'] == "Modulator"]
                print(df15)
            elif dplay == 2:
                df15 = df14
                print(df15)

            # Clean up temporary files 
            os.remove(f'{wd}/Sequence_1')
            os.remove(f'{wd}/muts.csv')
            

        elif Model == 2: 
            print(f'\n======= You are using the Design Module of IonNTxPred. Your results will be stored in file: {wd}/{result_filename} =====\n')
            print('==== Predicting Modulating Activity using Hybrid model: Processing sequences please wait ...')
            # Generate mutants and save
            muts = all_mutants(seq, seqid_1)
            muts.columns = ['SeqID', 'MutantID', 'seq']
            muts.to_csv(f'{wd}/muts.csv', index=False)

            # Read back mutants
            muts_df = pd.read_csv(f'{wd}/muts.csv')
            seq = muts_df['seq'].tolist()
            seqid_1 = muts_df['MutantID'].tolist()

            # Prepare ESM input
            with open(f'{wd}/out_len_mut', 'w') as f:
                for s in seq:
                    f.write(s + '\n')
            print()
            # Load and run ESM model
            model_save_path = f"{nf_path}/Model/saved_model_t12_ca_train_on_complete_data"
            tokenizer = AutoTokenizer.from_pretrained(model_save_path)
            model = EsmForSequenceClassification.from_pretrained(model_save_path)
            model.eval()

            run_esm_model(seq, seqid_1, result_filename, Threshold)
            ml_output_file = pd.read_csv(result_filename)
            seq_pred = ml_output_file.iloc[:, 2]

            # Prepare MERCI input
            muts_df[['MutantID', 'seq']].to_csv(f"{wd}/Sequence_1", index=False, header=False)

            # Modify Sequence_1 format to FASTA-like format
            with open(f"{wd}/Sequence_1", "r") as infile:
                lines = infile.readlines()

            with open(f"{wd}/Sequence_1", "w") as outfile:
                for line in lines:
                    parts = line.strip().split(",")
                    if len(parts) == 2:
                        outfile.write(f">{parts[0]}\n{parts[1]}\n")
 
            # Run MERCI
            merci_script = f"{nf_path}/merci/MERCI_motif_locator.pl"
            pos_motif_file = f"{nf_path}/motif/Ca/pos_motif.txt"
            neg_motif_file = f"{nf_path}/motif/Ca/neg_motif.txt"

            os.system(f"perl {merci_script} -p {wd}/Sequence_1 -i {pos_motif_file} -o {wd}/merci_p.txt")
            os.system(f"perl {merci_script} -p {wd}/Sequence_1 -i {neg_motif_file} -o {wd}/merci_n.txt")
            MERCI_Processor_p(f"{wd}/merci_p.txt", f"{wd}/merci_output_p.csv", seqid_1)
            Merci_after_processing_p(f"{wd}/merci_output_p.csv", f"{wd}/merci_hybrid_p.csv")
            MERCI_Processor_n(f"{wd}/merci_n.txt", f"{wd}/merci_output_n.csv", seqid_1)
            Merci_after_processing_n(f"{wd}/merci_output_n.csv", f"{wd}/merci_hybrid_n.csv")

            # Run Hybrid prediction
            hybrid(seq_pred, seqid_1, seq, f"{wd}/merci_hybrid_p.csv", f"{wd}/merci_hybrid_n.csv", Threshold, f"{wd}/final_output")
            df44 = pd.read_csv(f"{wd}/final_output")
            df44['Hybrid Score'] = df44['Hybrid Score'].clip(0, 1)
            
            # Rename column for clarity
            df44.rename(columns={'SeqID': 'MutantID', 'Mutant Sequence': 'Sequence'}, inplace=True)

            # Merge original SeqID for final output
            df44 = pd.merge(df44, muts_df[['MutantID', 'SeqID']], on='MutantID', how='left')
            # Reorder columns
            df44 = df44[['SeqID', 'MutantID', 'Sequence', 'ESM Score', 'MERCI Score (+ve)', 'MERCI Score (-ve)', 'Hybrid Score', 'Prediction']]
            df44['SeqID'] = df44['SeqID'].str.replace('>','')
            # Display
            if dplay == 1:
                print(df44[df44['Prediction'] == "Modulator"])
            elif dplay == 2:
                print(df44)

            # Save
            final_df = round(df44, 4)
            final_df.to_csv(result_filename, index=False)
            
            # Save to csv file
            df44.to_csv(f'{wd}/muts.csv',index=None)
            df44.to_csv(result_filename, index=None)
            
            # Clean up temporary files
            temp_files = [
                'final_output', 'merci_hybrid_p.csv', 'merci_hybrid_n.csv',
                'merci_output_p.csv', 'merci_output_n.csv',
                'merci_p.txt', 'merci_n.txt', 'out_m', 
                'Sequence_1', 'out_len_mut', 'out22', 'out33'
            ]
            for file in temp_files:
                if os.path.exists(file):
                    os.remove(file)

#=================================== Other ================================== 
    elif Channel == 4:
        if Model == 1:
            print('\n======= You are using the Design Module of IonNTxPred. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
            print('==== Predicting Modulating Activity using ESM2-t12 model: Processing sequences please wait ...')
            muts = all_mutants(seq, seqid_1)
            muts.to_csv(f'{wd}/muts.csv', index=None, header=None)  
            seq=muts['Seq'].tolist()
            seqid_1=muts['Mutant_ID'].tolist()

            # Load the tokenizer and model
            model_save_path = f"{nf_path}/Model/saved_model_t12_other_train_on_complete_data"
            tokenizer = AutoTokenizer.from_pretrained(model_save_path)
            model = EsmForSequenceClassification.from_pretrained(model_save_path)
            model.eval()
            # Run the ESM model
            run_esm_model(seq, seqid_1, result_filename, Threshold)
            df13 = pd.read_csv(result_filename)
            df13.columns = ['MutantID', 'Sequence', 'ESM Score', "Prediction"]
            seqid = pd.DataFrame(muts["SeqID"])
            df14 = pd.concat([seqid,df13],axis=1)
            df14['SeqID'] = df14['SeqID'].str.replace('>','')


            # Display results based on 'dplay'
            if dplay == 1:
                df15 = df14.loc[df14['Mutant Prediction'] == "Modulator"]
                print(df15)
            elif dplay == 2:
                df15 = df14
                print(df15)

            # Clean up temporary files 
            os.remove(f'{wd}/Sequence_1')
            os.remove(f'{wd}/muts.csv')
            

        elif Model == 2: 
            print(f'\n======= You are using the Design Module of IonNTxPred. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
            print('==== Predicting Modulating Activity using Hybrid model: Processing sequences please wait ...')
            # Generate mutants and save
            muts = all_mutants(seq, seqid_1)
            muts.columns = ['SeqID', 'MutantID', 'seq']
            muts.to_csv(f'{wd}/muts.csv', index=False)

            # Read back mutants
            muts_df = pd.read_csv(f'{wd}/muts.csv')
            seq = muts_df['seq'].tolist()
            seqid_1 = muts_df['MutantID'].tolist()

            # Prepare ESM input
            with open(f'{wd}/out_len_mut', 'w') as f:
                for s in seq:
                    f.write(s + '\n')
            print()
            # Load and run ESM model
            model_save_path = f"{nf_path}/Model/saved_model_t12_other_train_on_complete_data"
            tokenizer = AutoTokenizer.from_pretrained(model_save_path)
            model = EsmForSequenceClassification.from_pretrained(model_save_path)
            model.eval()

            run_esm_model(seq, seqid_1, result_filename, Threshold)
            ml_output_file = pd.read_csv(result_filename)
            seq_pred = ml_output_file.iloc[:, 2]

            # Prepare MERCI input
            muts_df[['MutantID', 'seq']].to_csv(f"{wd}/Sequence_1", index=False, header=False)

            # Modify Sequence_1 format to FASTA-like format
            with open(f"{wd}/Sequence_1", "r") as infile:
                lines = infile.readlines()

            with open(f"{wd}/Sequence_1", "w") as outfile:
                for line in lines:
                    parts = line.strip().split(",")
                    if len(parts) == 2:
                        outfile.write(f">{parts[0]}\n{parts[1]}\n")
 
            # Run MERCI
            merci_script = f"{nf_path}/merci/MERCI_motif_locator.pl"
            pos_motif_file = f"{nf_path}/motif/Other/pos_motif.txt"
            neg_motif_file = f"{nf_path}/motif/Other/neg_motif.txt"

            os.system(f"perl {merci_script} -p {wd}/Sequence_1 -i {pos_motif_file} -o {wd}/merci_p.txt")
            os.system(f"perl {merci_script} -p {wd}/Sequence_1 -i {neg_motif_file} -o {wd}/merci_n.txt")
            MERCI_Processor_p(f"{wd}/merci_p.txt", f"{wd}/merci_output_p.csv", seqid_1)
            Merci_after_processing_p(f"{wd}/merci_output_p.csv", f"{wd}/merci_hybrid_p.csv")
            MERCI_Processor_n(f"{wd}/merci_n.txt", f"{wd}/merci_output_n.csv", seqid_1)
            Merci_after_processing_n(f"{wd}/merci_output_n.csv", f"{wd}/merci_hybrid_n.csv")

            # Run Hybrid prediction
            hybrid(seq_pred, seqid_1, seq, f"{wd}/merci_hybrid_p.csv", f"{wd}/merci_hybrid_n.csv", Threshold, f"{wd}/final_output")
            df44 = pd.read_csv(f"{wd}/final_output")
            df44['Hybrid Score'] = df44['Hybrid Score'].clip(0, 1)
            
            # Rename column for clarity
            df44.rename(columns={'SeqID': 'MutantID', 'Mutant Sequence': 'Sequence'}, inplace=True)

            # Merge original SeqID for final output
            df44 = pd.merge(df44, muts_df[['MutantID', 'SeqID']], on='MutantID', how='left')
            # Reorder columns
            df44 = df44[['SeqID', 'MutantID', 'Sequence', 'ESM Score', 'MERCI Score (+ve)', 'MERCI Score (-ve)', 'Hybrid Score', 'Prediction']]
            df44['SeqID'] = df44['SeqID'].str.replace('>','')
            # Display
            if dplay == 1:
                print(df44[df44['Prediction'] == "Modulator"])
            elif dplay == 2:
                print(df44)

            # Save
            final_df = round(df44, 4)
            final_df.to_csv(result_filename, index=False)
            
            # Clean up temporary files
            temp_files = [
                'final_output', 'merci_hybrid_p.csv', 'merci_hybrid_n.csv',
                'merci_output_p.csv', 'merci_output_n.csv',
                'merci_p.txt', 'merci_n.txt', 'out_m', 
                'Sequence_1', 'out_len_mut', 'out22', 'out33'
            ]
            for file in temp_files:
                if os.path.exists(file):
                    os.remove(file)     
    
#======================= Protein Scanning Module starts from here =====================      
if Job == 3:
                #=================================== Na+ ==================================        
    if Channel == 1:
        if Model == 1:
            print('\n======= You are using the Protein Scanning module of IonNTxPred. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
            print('==== Scanning through ESM2-t12 model: Processing sequences please wait ...')
            # Load the tokenizer and model
            model_save_path = f"{nf_path}/Model/saved_model_t12_na_train_on_complete_data"
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
            run_esm_model(seq, seqid_1, result_filename, Threshold)
            df13 = pd.read_csv(result_filename)
            df13.rename(columns={"ML Score": "ESM Score"}, inplace=True)
            df13 = df13.drop(columns = ['SeqID', 'Seq'])
            df13 = pd.concat([df_1, df13], axis=1)
            df13["SeqID"] = df13["SeqID"].str.lstrip(">")
            df13.columns = ['SeqID','Pattern ID', 'Start', 'End', 'Sequence', 'ESM Score', "Prediction"]
            df13 = round(df13, 4)
            df13.to_csv(result_filename, index=None)

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

            # Load the tokenizer and model
            model_save_path = f"{nf_path}/Model/saved_model_t12_na_train_on_complete_data"
            tokenizer = AutoTokenizer.from_pretrained(model_save_path)
            model = EsmForSequenceClassification.from_pretrained(model_save_path)
            model.eval()

            # Wrap the list into a DataFrame if it's not already
            if isinstance(seq, list):
                seq = pd.DataFrame(seq)

            seq = seq.iloc[:, 0].tolist()
            df_1 = seq_pattern(seq, seqid_1, Win_len)
            seq = df_1["Seq"].tolist()
            
            # Generate pattern-style SeqIDs like seq1_Pattern_1
            raw_seqid = df_1["SeqID"].str.lstrip(">").tolist()
            pattern_counter = defaultdict(int)
            seqid_1 = []

            for sid in raw_seqid:
                pattern_counter[sid] += 1
                seqid_1.append(f"{sid}_Pattern_{pattern_counter[sid]}")

            run_esm_model(seq, seqid_1, f"{wd}/out22", Threshold)
            
            pattern_counter = defaultdict(int)
            fasta_path = f"{wd}/sequences.fasta"

            with open(fasta_path, "w") as fasta_file:
                for s, sid in zip(seq, seqid_1):
                    pattern_counter[sid] += 1
                    pattern_id = f">{sid}"  
                    fasta_file.write(f"{pattern_id}\n{s}\n")

            # Read ESM results
            ml_output_file = pd.read_csv(f"{wd}/out22", index_col=None)
            ml_output_file.rename(columns={ml_output_file.columns[2]: 'ESM Score'}, inplace=True)
            seq_pred = ml_output_file['ESM Score']
            
            # MERCI motif search on sequence FASTA
            merci_script = f"{nf_path}/merci/MERCI_motif_locator.pl"
            pos_motif_file = f"{nf_path}/motif/Na/pos_motif.txt"
            neg_motif_file = f"{nf_path}/motif/Na/neg_motif.txt"

            os.system(f"perl {merci_script} -p {fasta_path} -i {pos_motif_file} -o {wd}/merci_p.txt")
            os.system(f"perl {merci_script} -p {fasta_path} -i {neg_motif_file} -o {wd}/merci_n.txt")

            # Process MERCI results
            MERCI_Processor_p(f"{wd}/merci_p.txt", f"{wd}/merci_output_p.csv", seqid_1)
            Merci_after_processing_p(f"{wd}/merci_output_p.csv", f"{wd}/merci_hybrid_p.csv")
            MERCI_Processor_n(f"{wd}/merci_n.txt", f"{wd}/merci_output_n.csv", seqid_1)
            Merci_after_processing_n(f"{wd}/merci_output_n.csv", f"{wd}/merci_hybrid_n.csv")

            # Hybrid scoring 
            hybrid(seq_pred, seqid_1, seq, f"{wd}/merci_hybrid_p.csv", f"{wd}/merci_hybrid_n.csv", Threshold, f"{wd}/final_output")
            hybrid_final = pd.read_csv(f"{wd}/final_output")
            hybrid_final['Hybrid Score'] = hybrid_final['Hybrid Score'].clip(0, 1)
            
            df44 = hybrid_final.drop(columns = ['SeqID', 'Mutant Sequence'])
            df44 = pd.concat([df_1, df44], axis=1)
            df44["SeqID"] = df44["SeqID"].str.lstrip(">")
            df44.rename(columns={"Seq": "Sequence"}, inplace=True)
                     
            df44.loc[df44['Hybrid Score'] > 1, 'Hybrid Score'] = 1
            df44.loc[df44['Hybrid Score'] < 0, 'Hybrid Score'] = 0
                        
            # Display based on user preference
            if dplay == 1:
                df44 = df44.loc[df44.Prediction == "Modulator"]
                print(df44)
            elif dplay == 2:
                print(df44)

            df44 = round(df44,4)
            df44.to_csv(result_filename, index=None)
            
            # Clean up temporary files
            temp_files = [
                'final_output', 'merci_hybrid_p.csv', 'merci_hybrid_n.csv',
                'merci_output_p.csv', 'merci_output_n.csv',
                'merci_p.txt', 'merci_n.txt', 'sequences.fasta',
                'Sequence_1', 'out22'
            ]
            for file in temp_files:
                if os.path.exists(file):
                    os.remove(file)   

#=================================== K+ ==================================        
    elif Channel == 2:
        if Model == 1:
            print('\n======= You are using the Protein Scanning module of IonNTxPred. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
            print('==== Scanning through ESM2-t12 model: Processing sequences please wait ...')
            # Load the tokenizer and model
            model_save_path = f"{nf_path}/Model/saved_model_t12_k_train_on_complete_data"
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
            run_esm_model(seq, seqid_1, result_filename, Threshold)
            df13 = pd.read_csv(result_filename)
            df13.rename(columns={"ML Score": "ESM Score"}, inplace=True)
            df13 = df13.drop(columns = ['SeqID', 'Seq'])
            df13 = pd.concat([df_1, df13], axis=1)
            df13["SeqID"] = df13["SeqID"].str.lstrip(">")
            df13.columns = ['SeqID','Pattern ID', 'Start', 'End', 'Sequence', 'ESM Score', "Prediction"]
            df13 = round(df13, 4)
            df13.to_csv(result_filename, index=None)

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

            # Load the tokenizer and model
            model_save_path = f"{nf_path}/Model/saved_model_t12_k_train_on_complete_data"
            tokenizer = AutoTokenizer.from_pretrained(model_save_path)
            model = EsmForSequenceClassification.from_pretrained(model_save_path)
            model.eval()

            # Wrap the list into a DataFrame if it's not already
            if isinstance(seq, list):
                seq = pd.DataFrame(seq)

            seq = seq.iloc[:, 0].tolist()
            df_1 = seq_pattern(seq, seqid_1, Win_len)
            seq = df_1["Seq"].tolist()
            
            # Generate pattern-style SeqIDs like seq1_Pattern_1
            raw_seqid = df_1["SeqID"].str.lstrip(">").tolist()
            pattern_counter = defaultdict(int)
            seqid_1 = []

            for sid in raw_seqid:
                pattern_counter[sid] += 1
                seqid_1.append(f"{sid}_Pattern_{pattern_counter[sid]}")

            run_esm_model(seq, seqid_1, f"{wd}/out22", Threshold)
            
            pattern_counter = defaultdict(int)
            fasta_path = f"{wd}/sequences.fasta"

            with open(fasta_path, "w") as fasta_file:
                for s, sid in zip(seq, seqid_1):
                    pattern_counter[sid] += 1
                    pattern_id = f">{sid}"  
                    fasta_file.write(f"{pattern_id}\n{s}\n")

            # Read ESM results
            ml_output_file = pd.read_csv(f"{wd}/out22", index_col=None)
            ml_output_file.rename(columns={ml_output_file.columns[2]: 'ESM Score'}, inplace=True)
            seq_pred = ml_output_file['ESM Score']
            
            # MERCI motif search on sequence FASTA
            merci_script = f"{nf_path}/merci/MERCI_motif_locator.pl"
            pos_motif_file = f"{nf_path}/motif/K/pos_motif.txt"
            neg_motif_file = f"{nf_path}/motif/K/neg_motif.txt"

            os.system(f"perl {merci_script} -p {fasta_path} -i {pos_motif_file} -o {wd}/merci_p.txt")
            os.system(f"perl {merci_script} -p {fasta_path} -i {neg_motif_file} -o {wd}/merci_n.txt")

            # Process MERCI results
            MERCI_Processor_p(f"{wd}/merci_p.txt", f"{wd}/merci_output_p.csv", seqid_1)
            Merci_after_processing_p(f"{wd}/merci_output_p.csv", f"{wd}/merci_hybrid_p.csv")
            MERCI_Processor_n(f"{wd}/merci_n.txt", f"{wd}/merci_output_n.csv", seqid_1)
            Merci_after_processing_n(f"{wd}/merci_output_n.csv", f"{wd}/merci_hybrid_n.csv")

            # Hybrid scoring 
            hybrid(seq_pred, seqid_1, seq, f"{wd}/merci_hybrid_p.csv", f"{wd}/merci_hybrid_n.csv", Threshold, f"{wd}/final_output")
            hybrid_final = pd.read_csv(f"{wd}/final_output")
            hybrid_final['Hybrid Score'] = hybrid_final['Hybrid Score'].clip(0, 1)
            
            df44 = hybrid_final.drop(columns = ['SeqID', 'Mutant Sequence'])
            df44 = pd.concat([df_1, df44], axis=1)
            df44["SeqID"] = df44["SeqID"].str.lstrip(">")
            df44.rename(columns={"Seq": "Sequence"}, inplace=True)
                     
            df44.loc[df44['Hybrid Score'] > 1, 'Hybrid Score'] = 1
            df44.loc[df44['Hybrid Score'] < 0, 'Hybrid Score'] = 0
                        
            # Display based on user preference
            if dplay == 1:
                df44 = df44.loc[df44.Prediction == "Modulator"]
                print(df44)
            elif dplay == 2:
                print(df44)

            df44 = round(df44,4)
            df44.to_csv(result_filename, index=None)
            
            # Clean up temporary files
            temp_files = [
                'final_output', 'merci_hybrid_p.csv', 'merci_hybrid_n.csv',
                'merci_output_p.csv', 'merci_output_n.csv',
                'merci_p.txt', 'merci_n.txt', 'sequences.fasta',
                'Sequence_1', 'out22'
            ]
            for file in temp_files:
                if os.path.exists(file):
                    os.remove(file)   

#=================================== Ca2+ ==================================        
    elif Channel == 3:
        if Model == 1:
            print('\n======= You are using the Protein Scanning module of IonNTxPred. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
            print('==== Scanning through ESM2-t12 model: Processing sequences please wait ...')
            # Load the tokenizer and model
            model_save_path = f"{nf_path}/Model/saved_model_t12_ca_train_on_complete_data"
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
            run_esm_model(seq, seqid_1, result_filename, Threshold)
            df13 = pd.read_csv(result_filename)
            df13.rename(columns={"ML Score": "ESM Score"}, inplace=True)
            df13 = df13.drop(columns = ['SeqID', 'Seq'])
            df13 = pd.concat([df_1, df13], axis=1)
            df13["SeqID"] = df13["SeqID"].str.lstrip(">")
            df13.columns = ['SeqID','Pattern ID', 'Start', 'End', 'Sequence', 'ESM Score', "Prediction"]
            df13 = round(df13, 4)
            df13.to_csv(result_filename, index=None)

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

            # Load the tokenizer and model
            model_save_path = f"{wd}/Model/saved_model_t12_ca_train_on_complete_data"
            tokenizer = AutoTokenizer.from_pretrained(model_save_path)
            model = EsmForSequenceClassification.from_pretrained(model_save_path)
            model.eval()

            # Wrap the list into a DataFrame if it's not already
            if isinstance(seq, list):
                seq = pd.DataFrame(seq)

            seq = seq.iloc[:, 0].tolist()
            df_1 = seq_pattern(seq, seqid_1, Win_len)
            seq = df_1["Seq"].tolist()
            
            # Generate pattern-style SeqIDs like seq1_Pattern_1
            raw_seqid = df_1["SeqID"].str.lstrip(">").tolist()
            pattern_counter = defaultdict(int)
            seqid_1 = []

            for sid in raw_seqid:
                pattern_counter[sid] += 1
                seqid_1.append(f"{sid}_Pattern_{pattern_counter[sid]}")

            run_esm_model(seq, seqid_1, f"{wd}/out22", Threshold)
            
            pattern_counter = defaultdict(int)
            fasta_path = f"{wd}/sequences.fasta"

            with open(fasta_path, "w") as fasta_file:
                for s, sid in zip(seq, seqid_1):
                    pattern_counter[sid] += 1
                    pattern_id = f">{sid}"  
                    fasta_file.write(f"{pattern_id}\n{s}\n")

            # Read ESM results
            ml_output_file = pd.read_csv(f"{wd}/out22", index_col=None)
            ml_output_file.rename(columns={ml_output_file.columns[2]: 'ESM Score'}, inplace=True)
            seq_pred = ml_output_file['ESM Score']
            
            # MERCI motif search on sequence FASTA
            merci_script = f"{nf_path}/merci/MERCI_motif_locator.pl"
            pos_motif_file = f"{nf_path}/motif/Ca/pos_motif.txt"
            neg_motif_file = f"{nf_path}/motif/Ca/neg_motif.txt"

            os.system(f"perl {merci_script} -p {fasta_path} -i {pos_motif_file} -o {wd}/merci_p.txt")
            os.system(f"perl {merci_script} -p {fasta_path} -i {neg_motif_file} -o {wd}/merci_n.txt")

            # Process MERCI results
            MERCI_Processor_p(f"{wd}/merci_p.txt", f"{wd}/merci_output_p.csv", seqid_1)
            Merci_after_processing_p(f"{wd}/merci_output_p.csv", f"{wd}/merci_hybrid_p.csv")
            MERCI_Processor_n(f"{wd}/merci_n.txt", f"{wd}/merci_output_n.csv", seqid_1)
            Merci_after_processing_n(f"{wd}/merci_output_n.csv", f"{wd}/merci_hybrid_n.csv")

            # Hybrid scoring 
            hybrid(seq_pred, seqid_1, seq, f"{wd}/merci_hybrid_p.csv", f"{wd}/merci_hybrid_n.csv", Threshold, f"{wd}/final_output")
            hybrid_final = pd.read_csv(f"{wd}/final_output")
            hybrid_final['Hybrid Score'] = hybrid_final['Hybrid Score'].clip(0, 1)
            
            df44 = hybrid_final.drop(columns = ['SeqID', 'Mutant Sequence'])
            df44 = pd.concat([df_1, df44], axis=1)
            df44["SeqID"] = df44["SeqID"].str.lstrip(">")
            df44.rename(columns={"Seq": "Sequence"}, inplace=True)
                     
            df44.loc[df44['Hybrid Score'] > 1, 'Hybrid Score'] = 1
            df44.loc[df44['Hybrid Score'] < 0, 'Hybrid Score'] = 0
                        
            # Display based on user preference
            if dplay == 1:
                df44 = df44.loc[df44.Prediction == "Modulator"]
                print(df44)
            elif dplay == 2:
                print(df44)

            df44 = round(df44,4)
            df44.to_csv(result_filename, index=None)
            
            # Clean up temporary files
            temp_files = [
                'final_output', 'merci_hybrid_p.csv', 'merci_hybrid_n.csv',
                'merci_output_p.csv', 'merci_output_n.csv',
                'merci_p.txt', 'merci_n.txt', 'sequences.fasta',
                'Sequence_1', 'out22'
            ]
            for file in temp_files:
                if os.path.exists(file):
                    os.remove(file)   

#=================================== Other ==================================         
    elif Channel == 4:
        if Model == 1:
            print('\n======= You are using the Protein Scanning module of IonNTxPred. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
            print('==== Scanning through ESM2-t12 model: Processing sequences please wait ...')

            # Load the tokenizer and model
            model_save_path = f"{nf_path}/Model/saved_model_t12_other_train_on_complete_data"
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

            df13 = pd.read_csv(result_filename)
            df13.rename(columns={"ML Score": "ESM Score"}, inplace=True)
            df13 = df13.drop(columns = ['SeqID', 'Seq'])
            df13 = pd.concat([df_1, df13], axis=1)
            df13["SeqID"] = df13["SeqID"].str.lstrip(">")
            df13.columns = ['SeqID','Pattern ID', 'Start', 'End', 'Sequence', 'ESM Score', "Prediction"]
            df13 = round(df13, 4)
            df13.to_csv(result_filename, index=None)

            if dplay == 1:
                df13 = df13.loc[df13.Prediction == "Modulator"]
                print(df13)
            elif dplay == 2:
                print(df13)

            os.remove(f'{wd}/Sequence_1')

        elif Model == 2:
            print(f'\n======= You are using the Protein Scanning Module of IonNTxPred. Your results will be stored in file: 'f"{wd}/{result_filename}"' =====\n')
            print('==== Scanning sequences and predicting Modulating Activity using Hybrid model: Processing sequences please wait ...')

            # Load the tokenizer and model
            model_save_path = f"{nf_path}/Model/saved_model_t12_other_train_on_complete_data"
            tokenizer = AutoTokenizer.from_pretrained(model_save_path)
            model = EsmForSequenceClassification.from_pretrained(model_save_path)
            model.eval()

            # Wrap the list into a DataFrame if it's not already
            if isinstance(seq, list):
                seq = pd.DataFrame(seq)

            seq = seq.iloc[:, 0].tolist()
            df_1 = seq_pattern(seq, seqid_1, Win_len)
            seq = df_1["Seq"].tolist()
            
            # Generate pattern-style SeqIDs like seq1_Pattern_1
            raw_seqid = df_1["SeqID"].str.lstrip(">").tolist()
            pattern_counter = defaultdict(int)
            seqid_1 = []

            for sid in raw_seqid:
                pattern_counter[sid] += 1
                seqid_1.append(f"{sid}_Pattern_{pattern_counter[sid]}")

            run_esm_model(seq, seqid_1, f"{wd}/out22", Threshold)
            
            pattern_counter = defaultdict(int)
            fasta_path = f"{wd}/sequences.fasta"

            with open(fasta_path, "w") as fasta_file:
                for s, sid in zip(seq, seqid_1):
                    pattern_counter[sid] += 1
                    pattern_id = f">{sid}"  
                    fasta_file.write(f"{pattern_id}\n{s}\n")

            # Read ESM results
            ml_output_file = pd.read_csv(f"{wd}/out22", index_col=None)
            ml_output_file.rename(columns={ml_output_file.columns[2]: 'ESM Score'}, inplace=True)
            seq_pred = ml_output_file['ESM Score']
            
            # MERCI motif search on sequence FASTA
            merci_script = f"{nf_path}/merci/MERCI_motif_locator.pl"
            pos_motif_file = f"{nf_path}/motif/Other/pos_motif.txt"
            neg_motif_file = f"{nf_path}/motif/Other/neg_motif.txt"

            os.system(f"perl {merci_script} -p {fasta_path} -i {pos_motif_file} -o {wd}/merci_p.txt")
            os.system(f"perl {merci_script} -p {fasta_path} -i {neg_motif_file} -o {wd}/merci_n.txt")

            # Process MERCI results
            MERCI_Processor_p(f"{wd}/merci_p.txt", f"{wd}/merci_output_p.csv", seqid_1)
            Merci_after_processing_p(f"{wd}/merci_output_p.csv", f"{wd}/merci_hybrid_p.csv")
            MERCI_Processor_n(f"{wd}/merci_n.txt", f"{wd}/merci_output_n.csv", seqid_1)
            Merci_after_processing_n(f"{wd}/merci_output_n.csv", f"{wd}/merci_hybrid_n.csv")

            # Hybrid scoring 
            hybrid(seq_pred, seqid_1, seq, f"{wd}/merci_hybrid_p.csv", f"{wd}/merci_hybrid_n.csv", Threshold, f"{wd}/final_output")
            hybrid_final = pd.read_csv(f"{wd}/final_output")
            hybrid_final['Hybrid Score'] = hybrid_final['Hybrid Score'].clip(0, 1)
            
            df44 = hybrid_final.drop(columns = ['SeqID', 'Mutant Sequence'])
            df44 = pd.concat([df_1, df44], axis=1)
            df44["SeqID"] = df44["SeqID"].str.lstrip(">")
            df44.rename(columns={"Seq": "Sequence"}, inplace=True)
                     
            df44.loc[df44['Hybrid Score'] > 1, 'Hybrid Score'] = 1
            df44.loc[df44['Hybrid Score'] < 0, 'Hybrid Score'] = 0
                        
            # Display based on user preference
            if dplay == 1:
                df44 = df44.loc[df44.Prediction == "Modulator"]
                print(df44)
            elif dplay == 2:
                print(df44)

            df44 = round(df44,4)
            df44.to_csv(result_filename, index=None)
            
            # Clean up temporary files
            temp_files = [
                'final_output', 'merci_hybrid_p.csv', 'merci_hybrid_n.csv',
                'merci_output_p.csv', 'merci_output_n.csv',
                'merci_p.txt', 'merci_n.txt', 'sequences.fasta',
                'Sequence_1', 'out22'
            ]
            for file in temp_files:
                if os.path.exists(file):
                    os.remove(file)               
                        
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
               	                    
print('\n\n🎉 ======= Thank You for Using IonNTxPred! ======= 🎉')
print('🙏 We hope this tool contributed to your research on ion channel modulating proteins.')
print('\n📖 If you found IonNTxPred useful, please cite us in your work:')
print('    ➤ Rathore et al., *IonNTxPred: LLM-based Prediction and Designing of Ion Channel Impairing Proteins*, 2024.')
print('\n🔗 Useful Links:')
print('    🌐 Web Server : https://webs.iiitd.edu.in/raghava/ionntxpred/')
print('    💻 GitHub     : https://github.com/raghavagps/IonNTxPred')
print('    🤗 HuggingFace: https://huggingface.co/raghavagps-group/IonNTxPred\n')
       