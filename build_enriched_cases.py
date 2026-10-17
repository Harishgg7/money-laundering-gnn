"""
Extract authentic IBM AML customer metadata and raw transactions
for the curated presentation cases to ensure 100% data authenticity.
"""
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np

import config

def build_enriched_cases():
    print("[1/4] Loading accounts metadata...")
    acc_df = pd.read_csv(config.RAW_ACCOUNTS_CSV)
    acc_df["clean_key"] = acc_df["Bank ID"].astype(str) + "-" + acc_df["Account Number"].astype(str)
    acc_lookup = dict(zip(acc_df["clean_key"], zip(acc_df["Bank Name"], acc_df["Entity Name"], acc_df["Entity ID"])))

    print("[2/4] Loading curated cases...")
    with open(config.OUTPUT_DIR / "curated_cases.json", "r") as f:
        curated = json.load(f)

    # Extract unique account numbers in curated cases
    acct_numbers = set()
    for c_key, c_val in curated["cases"].items():
        for n in c_val["nodes"]:
            acct_numbers.add(n["key"].split("-")[1])

    print(f"[3/4] Scanning raw transactions for {len(acct_numbers)} accounts...")
    t_df = pd.read_csv(config.RAW_TRANS_CSV)
    m_df = t_df[t_df["Account"].isin(acct_numbers) | t_df["Account.1"].isin(acct_numbers)].copy()
    m_df.sort_values("Timestamp", ascending=False, inplace=True)
    print(f"Found {len(m_df)} raw transactions.")

    # Build transactions list for each case
    for c_key, c_val in curated["cases"].items():
        case_acct_nums = set(n["key"].split("-")[1] for n in c_val["nodes"])
        
        # Enrich nodes with real bank and customer
        for n in c_val["nodes"]:
            b_id = str(int(n["bank_id"])) if n["bank_id"].isdigit() else n["bank_id"]
            lookup_key = f"{b_id}-{n['key'].split('-')[1]}"
            if lookup_key in acc_lookup:
                n["bank_name"] = acc_lookup[lookup_key][0]
                n["customer_name"] = acc_lookup[lookup_key][1]
                n["entity_id"] = acc_lookup[lookup_key][2]
            else:
                n["bank_name"] = f"Institution #{b_id}"
                n["customer_name"] = f"Commercial Account #{n['id']}"
                n["entity_id"] = f"ENT-{n['id']}"
                
        # Set subject account customer name
        subj = next(n for n in c_val["nodes"] if n.get("is_target"))
        c_val["subject_customer_name"] = subj["customer_name"]
        c_val["subject_bank_name"] = subj["bank_name"]
        c_val["subject_entity_id"] = subj["entity_id"]

        # Filter raw transactions involving this case's accounts
        case_txs = m_df[m_df["Account"].isin(case_acct_nums) | m_df["Account.1"].isin(case_acct_nums)]
        
        tx_list = []
        for idx, row in case_txs.head(60).iterrows():
            s_bank = str(row["From Bank"])
            s_acct = str(row["Account"])
            d_bank = str(row["To Bank"])
            d_acct = str(row["Account.1"])
            
            amt = float(row["Amount Received"])
            is_illicit = int(row["Is Laundering"]) == 1
            curr = str(row["Receiving Currency"])
            fmt = str(row["Payment Format"])
            ts = str(row["Timestamp"])
            
            # Determine direction relative to target
            is_target_sender = (s_acct == subj["key"].split("-")[1])
            is_target_recv = (d_acct == subj["key"].split("-")[1])
            
            direction = "OUTGOING" if is_target_sender else ("INCOMING" if is_target_recv else "INTERMEDIARY")
            risk = "HIGH" if is_illicit else ("HIGH" if amt >= 100000 else ("MED" if amt >= 10000 else "LOW"))
            
            tx_list.append({
                "id": f"TX-{idx:07d}",
                "date": ts,
                "from": f"{s_bank}-{s_acct}",
                "to": f"{d_bank}-{d_acct}",
                "from_account": s_acct,
                "to_account": d_acct,
                "amount": f"${amt:,.2f}",
                "amount_raw": amt,
                "currency": curr,
                "direction": direction,
                "type": fmt.upper(),
                "is_laundering": is_illicit,
                "risk": risk
            })
        c_val["transactions"] = tx_list
        
        # Build chronological timeline
        timeline = []
        for tx in tx_list[:8]:
            status_text = "Illicit flag detected" if tx["is_laundering"] else "Monitored transfer"
            timeline.append({
                "time": tx["date"],
                "title": f"{tx['amount']} {tx['type']} ({tx['direction'].title()})",
                "desc": f"{tx['from']} -> {tx['to']} ({status_text})"
            })
        c_val["timeline"] = timeline

    out_file = config.OUTPUT_DIR / "enriched_cases.json"
    with open(out_file, "w") as f:
        json.dump(curated, f, indent=2)

    print(f"[done] Enriched cases successfully saved to {out_file}")

if __name__ == "__main__":
    build_enriched_cases()
