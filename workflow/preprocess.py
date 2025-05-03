import pandas as pd

def AMLSim_preprocessing(dimension):
    account_prop = {}

    dataset_path = f"datasets/AMLSim/{dimension}"

    accounts = pd.read_csv(f'{dataset_path}/accounts.csv')
    transactions = pd.read_csv(f'{dataset_path}/transactions.csv')

    # Create root
    balances = {}

    # Take users data from account.csv
    for _, row in accounts.iterrows():
        acc_id = row["ACCOUNT_ID"] # User id
        fraud = row['IS_FRAUD']

        # Initialize the nested dictionary for this account ID
        account_prop[acc_id] = {
            "original_id": acc_id,
            "fraud": False  # Default value
        }

        # Check if the account is fraudulent and update accordingly
        if fraud == 'true':
            account_prop[acc_id]['fraud'] = True

        open_date = 0
        initial_balance = round(float(row["INIT_BALANCE"]), 2)

        balances[acc_id] = [{"date": open_date,
                            "balance": initial_balance
                            }]
    
    # Apply transactions to users' balances
    transactions.sort_values(by="TX_ID")

    # Track the last balance for each account
    current_balances = {acct_id: balances[acct_id][0]["balance"] for acct_id in balances}

    for i, (_, row) in enumerate(transactions.iterrows()):
        orig_acct = row["SENDER_ACCOUNT_ID"]
        bene_acct = row["RECEIVER_ACCOUNT_ID"]
        amount = row["TX_AMOUNT"]
        tx_type = row['TX_TYPE']
        date = row["TIMESTAMP"]

        if orig_acct in balances: # check that sender has a deposit
            last_balance = balances[orig_acct][-1]["balance"] # read balance from balances
            if tx_type in ['TRANSFER', 'WITHDRAWAL']: # all TRANSFER in the csv
                new_balance = last_balance - amount

            balances[orig_acct].append({"date": date, "balance": round(new_balance, 2)})
                
        
        if bene_acct in balances: # check that receiver has a deposit
            last_balance = balances[bene_acct][-1]["balance"]
            if tx_type in ['TRANSFER', 'DEPOSIT']:
                new_balance = last_balance + amount
        
            balances[bene_acct].append({"date": date, "balance": round(new_balance, 2)})
    
    result = []

    for date in sorted(set(transactions["TIMESTAMP"])):
        daily_balances = {
            "date": date,
            "balances": {
                str(user): {
                    "balance": next((entry["balance"] for entry in reversed(b) if entry["date"] <= date), b[0]["balance"])
                } for user, b in balances.items()}
        }
        result.append(daily_balances)

    df = pd.DataFrame(result)

    # Unpack 'balances' into separate columns for each user
    df = df.join(pd.DataFrame(df['balances'].tolist()))

    df.drop('balances', axis=1, inplace=True)
    df.set_index('date', inplace=True)
    df.drop('_id', axis=1, inplace=True, errors='ignore')

    for col in df.columns:
        df[col] = df[col].apply(lambda x: x['balance'] if isinstance(x, dict) else x)

    return df, account_prop


def PaySim_preprocessing(dimension):

    account_prop = {}

    #dataset_path = f"datasets/paysim/{dimension}"

    #transactions = pd.read_csv(f'{dataset_path}/rawLog.csv')
    transactions = pd.read_csv('playground/squic_folder/paysim100.csv')

    balances = {}
    fraud_account = set()
    users_list = []

    for _, row in transactions.iterrows():
        origin = row['nameOrig']
        destination = row['nameDest']
        fraud = row['isFraud']

        if origin not in users_list:
            users_list.append(origin)
        if row['nameDest'] not in users_list:
            users_list.append(destination)

        if fraud == '1':
            fraud_account.add(origin)
            fraud_account.add(destination)

        # Track origin account
        if origin not in balances:
            balances[origin] = {}
        balances[origin][row['step']] = row['newBalanceOrig']

        # Track destination account
        if destination not in balances:
            balances[destination] = {}
        balances[destination][row['step']] = row['newBalanceDest']

    account_prop = {}

    for i, user in enumerate(users_list):
        account_prop[i] = {
            "original_id": user,  # Using assignment operator = not ==
            "fraud": user in fraud_account 
        }

    # Create DataFrame
    df = pd.DataFrame.from_dict(balances, orient='index').T

    # Identify the full range of steps (days)
    full_range = range(df.index.min(), df.index.max() + 1)

    # Reindex to include all steps, then forward fill
    df = df.reindex(full_range).ffill().bfill()

    return df, account_prop

