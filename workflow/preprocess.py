import pandas as pd
from scipy.sparse import lil_matrix
from collections import defaultdict



def AMLSim_preprocessing(dimension):
    account_prop = {}

    dataset_path = f"datasets/AMLSim/{dimension} users"

    if dimension != '100':
        # dimension grater than 100 read by chunk
        if dimension == '1K':
            chunk_acc = pd.read_csv(f'{dataset_path}/accounts.csv', chunksize=100)
            chunk_trns = pd.read_csv(f'{dataset_path}/transactions.csv', chunksize=100)
        elif dimension == '10K':
            chunk_acc = pd.read_csv(f'{dataset_path}/accounts.csv', chunksize=1000)
            chunk_trns = pd.read_csv(f'{dataset_path}/transactions.csv', chunksize=1000)
        else:
            chunk_acc = pd.read_csv(f'{dataset_path}/accounts.csv', chunksize=10000)
            chunk_trns = pd.read_csv(f'{dataset_path}/transactions.csv', chunksize=10000)

        accounts = pd.concat(chunk_acc)
        transactions = pd.concat(chunk_trns)
    else:
        # dimension == 100 read in normal way
        accounts = pd.read_csv(f'{dataset_path}/accounts.csv')
        transactions = pd.read_csv(f'{dataset_path}/transactions.csv')

    # Create root for balances dict (initial balance and timestamp 0)
    balances = {
        acc_id: [{"date": 0, "balance": round(float(init_bal), 2)}] for acc_id, init_bal in zip(accounts["ACCOUNT_ID"], accounts["INIT_BALANCE"])
    }

    accounts["IS_FRAUD"] = accounts["IS_FRAUD"].astype(str).str.lower() == "true"

    # Create account_prop dictionary
    account_prop = {
        acc_id: is_fraud for acc_id, is_fraud in zip(accounts["ACCOUNT_ID"], accounts["IS_FRAUD"])
    }

    transactions.sort_values(by="TX_ID", inplace=True)

    max_account_id = max(transactions["SENDER_ACCOUNT_ID"].max(),
                        transactions["RECEIVER_ACCOUNT_ID"].max()) + 1

    matrix = lil_matrix((max_account_id, max_account_id), dtype=int)
    
    for row in transactions.itertuples(index=False): # If True, return the index as the first element of the tuple
        orig_acct = row.SENDER_ACCOUNT_ID
        bene_acct = row.RECEIVER_ACCOUNT_ID
        amount = float(row.TX_AMOUNT)
        tx_type = row.TX_TYPE
        date = row.TIMESTAMP

        # Update transaction matrix
        matrix[orig_acct, bene_acct] += float(amount)
        matrix[orig_acct, bene_acct] = round(matrix[orig_acct, bene_acct], 2)

        # Process sender
        if tx_type in ['TRANSFER', 'WITHDRAWAL'] and orig_acct in balances:
            last_balance = balances[orig_acct][-1]["balance"]
            new_balance = last_balance - amount
            balances[orig_acct].append({
                "date": date,
                "balance": round(new_balance, 2)
            })

        # Process receiver
        if tx_type in ['TRANSFER', 'DEPOSIT'] and bene_acct in balances:
            last_balance = balances[bene_acct][-1]["balance"]
            new_balance = last_balance + amount
            balances[bene_acct].append({
                "date": date,
                "balance": round(new_balance, 2)
            })
    
    # Step 1: Flatten the balances dict to a DataFrame
    records = []

    for user_id, history in balances.items():
        for entry in history:
            records.append({
                "ACCOUNT_ID": user_id,
                "TIMESTAMP": entry["date"],
                "BALANCE": entry["balance"]
            })

    df = pd.DataFrame(records)

    # Sort by timestamp and user
    df.sort_values(by=["ACCOUNT_ID", "TIMESTAMP"], inplace=True)
    # Remove duplicates keeping the latest
    df.drop_duplicates(subset=["ACCOUNT_ID", "TIMESTAMP"], keep="last", inplace=True)

    # Create a full grid of all timestamps and all users
    all_dates = sorted(transactions["TIMESTAMP"].unique())
    all_users = df["ACCOUNT_ID"].unique()
    grid = pd.MultiIndex.from_product([all_users, all_dates], names=["ACCOUNT_ID", "TIMESTAMP"])

    # Reindex the balance DataFrame to the full grid
    df = df.set_index(["ACCOUNT_ID", "TIMESTAMP"])
    df = df.reindex(grid)

    # Forward fill missing balances (per user)
    df["BALANCE"] = df["BALANCE"].groupby(level=0).ffill()

    # Reset index and pivot to get final table: one row per date, one column per user
    df = df.reset_index().pivot(index="TIMESTAMP", columns="ACCOUNT_ID", values="BALANCE")
    df.index.name = "date"
    df.columns = df.columns.astype(str)  # match your original str(user) keys

    return df, account_prop, matrix


def PaySim_preprocessing(dimension):

    account_prop = {}

    dataset_path = f"datasets/paysim/{dimension} users"

    if dimension != '100':
        if dimension == '1K':
            chunk = pd.read_csv(f'{dataset_path}/rawLog.csv', chunksize=100)
        elif dimension == '10K':
            chunk = pd.read_csv(f'{dataset_path}/rawLog.csv', chunksize=1000)
        elif dimension == '100K':
            chunk = pd.read_csv(f'{dataset_path}/rawLog.csv', chunksize=10000)
        
        transactions  = pd.concat(chunk)
    else:
        transactions = pd.read_csv(f'{dataset_path}/rawLog.csv')

    # Initialize
    balances = defaultdict(dict)   # {account: {step: balance}}
    fraud_account = set()
    users_set = set()

    for row in transactions.itertuples(index=False):
        step = row.step
        origin = row.nameOrig
        dest = row.nameDest
        new_orig_balance = row.newBalanceOrig
        new_dest_balance = row.newBalanceDest
        is_fraud = row.isFraud

        # Track users -> to transform then ID in Int
        users_set.add(origin)
        users_set.add(dest)

        # Track frauds
        if is_fraud == 1:
            fraud_account.add(origin)
            fraud_account.add(dest)

        balances[origin][step] = new_orig_balance
        balances[dest][step] = new_dest_balance

    # Finalize user list and properties
    users_list = list(users_set)

    account_prop = {
        i: {
            "original_id": user,
            "fraud": user in fraud_account,
            "class": str(user)[0]  # 'C', 'M', or 'B'
        } for i, user in enumerate(users_list)
    }

    # Create transactions matrix
    max_account_id = max(account_prop) + 1

    # Create sparse matrix
    matrix = lil_matrix((max_account_id, max_account_id), dtype=int)

    user_to_id = {user: i for i, user in enumerate(users_list)}

    for row in transactions.itertuples(index=False):
        origin = row.nameOrig
        dest = row.nameDest
        amount = float(row.amount)

        if origin in user_to_id and dest in user_to_id:
            i = user_to_id[origin]
            j = user_to_id[dest]
            matrix[i, j] += amount
            matrix[i, j] = round(matrix[i, j], 2)

    # Create DataFrame
    df = pd.DataFrame.from_dict(balances, orient='index').T

    # Identify the full range of steps (days)
    full_range = range(df.index.min(), df.index.max() + 1)

    # Reindex to include all steps, then forward fill
    df = df.reindex(full_range).ffill().bfill()

    return df, account_prop, matrix


def Libra_preprocessing():
    account_prop = {}

    dataset_path = 'datasets/libra/realdata/'

    transactions = pd.read_csv(dataset_path)

    # Add timestamp column
    n_steps = 90
    transactions["step"] = transactions.random.randint(0, n_steps, size=len(transactions))

    transactions = transactions.sort_values("step").reset_index(drop=True)

    max_account_id = max(transactions.id_source.max(),
                        transactions.id_destination.max()) + 1

    # Initialize transaction matrix as sparse
    matrix = lil_matrix((max_account_id, max_account_id), dtype=int)


    balances = defaultdict(dict)   # {account: {step: balance}}

    for row in transactions.itertuples(index=False):
        source = int(row.id_source)
        destination = int(row.id_destination)
        step = int(row.step)
        amount = round(float(row.cum_amount), 3)  

        # Update sparse matrix
        matrix[source, destination] += amount

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

    # Create DataFrame
    df = pd.DataFrame.from_dict(balances, orient='index').T

    # Identify the full range of steps (days)
    full_range = range(int(df.index.min()), int(df.index.max()) + 1)

    # Reindex to include all steps, then forward fill
    df = df.reindex(full_range).fillna(method='ffill').fillna(method='bfill')

    # Reorder columns by IDs
    columns_as_int = []
    for col in df.columns:
        columns_as_int.append(int(str(col).strip()))

    columns_as_int.sort()

    df = df[columns_as_int]

    return df, account_prop, matrix
