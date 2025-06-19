import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix, coo_matrix
from collections import defaultdict



def AMLSim_preprocessing(dimension):
    account_prop = {}
    max_steps = 200

    dataset_path = f"Datasets/AMLSim/{dimension} users"
    accounts_csv = f"{dataset_path}/accounts.csv"
    transactions_csv = f"{dataset_path}/transactions.csv"

    dict_dimension = {
        '100': 1000,
        '1K': 10000,
        '10K': 100000,
        '100K': 1000000,
        '1M': 10000000
    }

    accounts_df = pd.read_csv(accounts_csv)
    
    # Setup account mapping and initial state
    unique_accounts = sorted(accounts_df['ACCOUNT_ID'])
    
    initial_balances = {}
    fraud_accounts = set()
    
    for _, row in accounts_df.iterrows():
        acc_id = row['ACCOUNT_ID']
        initial_balances[acc_id] = float(row['INIT_BALANCE'])
        if str(row['IS_FRAUD']) == 'true':
            fraud_accounts.add(acc_id)
    
    # Initialize structures
    df = pd.DataFrame(index=range(max_steps), columns=unique_accounts, dtype=float)
    for acc_id in unique_accounts:
        df.loc[:, acc_id] = initial_balances[acc_id]
    
    current_balances = initial_balances.copy()
    # edge_weights = defaultdict(float)
        
    # Process transactions by timestamp chunks to maintain balance accuracy
    chunk_size = dict_dimension[dimension]
    
    # Read and process transactions chunk by chunk, sorting each chunk
    for chunk in pd.read_csv(transactions_csv, chunksize=chunk_size):
        chunk['IS_FRAUD'] = chunk['IS_FRAUD'].astype(str).str == 'true'
        chunk = chunk.sort_values('TIMESTAMP')
                
        for _, row in chunk.iterrows():
            timestamp = int(row['TIMESTAMP'])
                
            sender_id = row['SENDER_ACCOUNT_ID']
            receiver_id = row['RECEIVER_ACCOUNT_ID']
            amount = float(row['TX_AMOUNT'])
            is_fraud = row['IS_FRAUD']
            
            # Update balances and DataFrame directly
            if sender_id in current_balances:
                current_balances[sender_id] -= amount
                df.loc[timestamp:, sender_id] = current_balances[sender_id]
                
            if receiver_id in current_balances:
                current_balances[receiver_id] += amount
                df.loc[timestamp:, receiver_id] = current_balances[receiver_id]
            
            # Update transaction matrix
            # edge_weights[(sender_id, receiver_id)] += amount
            
            # Track fraud
            if is_fraud:
                fraud_accounts.add(sender_id)
                fraud_accounts.add(receiver_id)
        
        del chunk
    
    # Create transaction matrix and account properties (same as before)
    max_account_id = len(unique_accounts)
    
    # if edge_weights:
    #     rows, cols, data = zip(*[(i, j, amt) for (i, j), amt in edge_weights.items()])
    #     transaction_matrix = coo_matrix((data, (rows, cols)), 
    #                                   shape=(max_account_id, max_account_id)).tolil()
    # else:
    #     transaction_matrix = lil_matrix((max_account_id, max_account_id), dtype=float)
    
    account_prop = {}
    for i, acc_id in enumerate(unique_accounts):
        account_prop[i] = {
            "fraud": acc_id in fraud_accounts,
        }
    
    return df, account_prop


def PaySim_preprocessing(dimension):
    max_steps=365

    account_prop = {}

    dataset_path = f"Datasets/paysim/{dimension} users/rawLog.csv"

    print("First pass: Identifying users and fraud accounts...")
    unique_users = set()
    fraud_accounts = set()

    dict_dimension = {
        '100': 1000,
        '1K': 10000,
        '10K': 100000,
        '100K': 1000000,
        '1M': 10000000
    }

    for chunk in pd.read_csv(dataset_path, chunksize=dict_dimension[dimension]):
        # Get unique users
        orig_users = chunk['nameOrig'].dropna().unique()
        dest_users = chunk['nameDest'].dropna().unique()
        unique_users.update(orig_users)
        unique_users.update(dest_users)
        
        # Identify fraud accounts
        fraud_rows = chunk[chunk['isFraud'] == 1]
        fraud_accounts.update(fraud_rows['nameOrig'].dropna())
        fraud_accounts.update(fraud_rows['nameDest'].dropna())
        
        del chunk
    
    unique_users = sorted(list(unique_users))
    print(f"Found {len(unique_users)} unique users")
    print(f"Found {len(fraud_accounts)} fraud accounts")
    
    # Create user-to-id mapping
    user_to_id = {user: i for i, user in enumerate(unique_users)}
    max_account_id = len(unique_users)
    
    # Initialize DataFrame for balances
    df = pd.DataFrame(index=range(max_steps), columns=unique_users, dtype=float)
    
    # Initialize transaction tracking
    edge_weights = defaultdict(float)
    
    # Track which users we've seen for balance tracking
    user_first_seen = {}
    
    print("Second pass: Processing transactions and building matrices...")
    
    # Second pass: populate DataFrame and build transaction matrix
    for chunk in pd.read_csv(dataset_path, chunksize=dict_dimension[dimension]):
        for _, row in chunk.iterrows():
            step = int(row['step'])
            amount = float(row['amount'])
            
            # Get users
            orig_user = row['nameOrig']
            dest_user = row['nameDest']
            
            # Build transaction matrix (only if both users are valid)
            # if orig_user in user_to_id and dest_user in user_to_id:
            #     i = user_to_id[orig_user]
            #     j = user_to_id[dest_user]
            #     edge_weights[(i, j)] += amount
            
            # Process origin user for balance DataFrame
            old_balance = float(row['oldBalanceOrig'])
            new_balance = float(row['newBalanceOrig'])
                
            if orig_user not in user_first_seen:
                # First time seeing this user - backfill with old balance
                if step > 0:
                    df.loc[:step-1, orig_user] = old_balance
                user_first_seen[orig_user] = True
                
            # Update from current step onwards
            df.loc[step:, orig_user] = new_balance
            
            # Process destination user for balance DataFrame
            old_balance = float(row['oldBalanceDest'])
            new_balance = float(row['newBalanceDest'])
                
            if dest_user not in user_first_seen:
                # First time seeing this user - backfill with old balance
                if step > 0:
                    df.loc[:step-1, dest_user] = old_balance
                user_first_seen[dest_user] = True
                
            # Update from current step onwards
            df.loc[step:, dest_user] = new_balance
        
        del chunk
    
    # Convert edge_weights dict to sparse matrix
    # print("Converting transaction data to sparse matrix...")
    # rows, cols, data = zip(*[(i, j, amt) for (i, j), amt in edge_weights.items()])
    # transaction_matrix = coo_matrix((data, (rows, cols)), 
    #                                   shape=(max_account_id, max_account_id)).tolil()
   
    # Create account properties dictionary
    account_prop = {
        i: {
            "original_id": user,
            "fraud": user in fraud_accounts,
            "class": str(user)[0] if str(user)[0] != 'B' else 'C'  # Class is 'C' or 'M' but map 'B' to 'C' (since there is just 1 B)
        }
        for i, user in enumerate(unique_users)
    }

    return df, account_prop


def Libra_preprocessing():
    account_prop = {}

    dataset_path = 'Datasets/libra/realdata/libra_380K.csv'

    transactions = pd.read_csv(dataset_path)

    # Add timestamp column
    n_steps = 90
    transactions["step"] = np.random.randint(0, n_steps, size=len(transactions))

    transactions = transactions.sort_values("step").reset_index(drop=True)

    max_account_id = max(transactions.id_source.max(),
                        transactions.id_destination.max()) + 1

    # Initialize transaction matrix as sparse
    # trans_matrix = lil_matrix((max_account_id, max_account_id), dtype=int)


    balances = defaultdict(dict)   # {account: {step: balance}}
    fraud_accounts = set() # to track fraudolent accounts (nr_reports = 1)
    unique_accounts = set()

    for row in transactions.itertuples(index=False):
        source = int(row.id_source)
        destination = int(row.id_destination)
        step = int(row.step)
        amount = round(float(row.cum_amount), 3)  
        report = int(row.nr_reports)

        unique_accounts.update(source)
        unique_accounts.update(destination)

        # Update sparse trans_matrix
        # trans_matrix[source, destination] += amount

        # Initialize balances if user appears for the first time
        if source not in balances:
            balances[source] = {step: 0}  # First transaction, balance starts at 0
        if destination not in balances:
            balances[destination] = {step: 0}

        # Get the previous balance (default to 0 if first transaction)
        prev_balance_source = max(balances[source].values(), default=0)
        prev_balance_destination = max(balances[destination].values(), default=0)

        # Update balances for this step
        balances[source][step] = round(prev_balance_source - amount, 3)  # Source loses money
        balances[destination][step] = round(prev_balance_destination + amount, 3)  # Destination gains money

        if report > 0:
            fraud_accounts.update(source)
            fraud_accounts.update(destination)

    # Create DataFrame
    df = pd.DataFrame.from_dict(balances, orient='index').T

    # Identify the full range of steps (days)
    full_range = range(int(df.index.min()), int(df.index.max()) + 1)

    # Reindex to include all steps, then forward fill
    # df = df.reindex(full_range).fillna(method='ffill').fillna(method='bfill') -> will be depracated
    df = df.reindex(full_range).ffill().bfill()

    # Reorder columns by IDs
    columns_as_int = []
    for col in df.columns:
        columns_as_int.append(int(str(col).strip()))

    columns_as_int.sort()

    df = df[columns_as_int]

    account_prop = {}
    for acc_id in unique_accounts:
        account_prop[acc_id] = {
            "fraud": acc_id in fraud_accounts,
        }

    return df, account_prop
