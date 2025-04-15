import pandas as pd
import json


if __name__ == '__main__':
    DATASET = "../../datasets/AMLSimData/3_100K"
    SAVE_IN = "../../datasets/AMLSimData/json"

    accounts = pd.read_csv(f'{DATASET}/accounts.csv')
    transactions = pd.read_csv(f'{DATASET}/transactions.csv')

    # Create root
    balances = {}

    # Take users data from account.csv
    for _, row in accounts.iterrows():
        acc_id = row["ACCOUNT_ID"] # User id
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

    with open(f'{SAVE_IN}/amlsim100K.json', 'w') as file_json:
        json.dump(result, file_json, indent=4)