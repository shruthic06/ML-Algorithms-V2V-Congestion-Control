import argparse
import pandas as pd

# Default policy thresholds (tune to match your report appendix)
DEFAULTS = dict(
    Tpmax=20, Tpmin=5, Trmax=10, Trmin=5,
    Cac=0.05,   # CBR low threshold
    Cc=0.25,    # CBR high threshold
)

def tune_row(cbr, veh=None, pdr=None, Tp=None, Tr=None, cfg=DEFAULTS):
    Tp = cfg['Tpmax'] if Tp is None else Tp
    Tr = cfg['Trmin'] if Tr is None else Tr

    # Example control logic (placeholder — align with the exact rules in your appendix):
    # - If CBR is below acceptable low (Cac), increase Tx rate mildly.
    # - If CBR is above congestion (Cc), decrease Tx rate and increase period.
    # - Otherwise keep near operating point.
    if cbr < cfg['Cac']:
        Tr = min(cfg['Trmax'], Tr + 1)
        Tp = max(cfg['Tpmin'], Tp - 1)
    elif cbr > cfg['Cc']:
        Tr = max(cfg['Trmin'], Tr - 1)
        Tp = min(cfg['Tpmax'], Tp + 1)
    else:
        # within band: small nudges toward mid
        midTr = (cfg['Trmin'] + cfg['Trmax']) // 2
        if Tr < midTr: Tr += 1
        elif Tr > midTr: Tr -= 1

    return Tp, Tr

def run(csv_path, cbr_col='CBR', veh_col=None, pdr_col=None):
    df = pd.read_csv(csv_path)
    cbr_vals = df[cbr_col].tolist()
    veh = df[veh_col].tolist() if veh_col and veh_col in df.columns else [None]*len(cbr_vals)
    pdr = df[pdr_col].tolist() if pdr_col and pdr_col in df.columns else [None]*len(cbr_vals)

    Tp_list, Tr_list = [], []
    Tp, Tr = None, None
    for i, c in enumerate(cbr_vals):
        Tp, Tr = tune_row(c, veh[i], pdr[i], Tp, Tr)
        Tp_list.append(Tp)
        Tr_list.append(Tr)

    out = pd.DataFrame({'Tp_tuned': Tp_list, 'Tr_tuned': Tr_list})
    print(out.head(10).to_string(index=False))
    out.to_csv('pa_a_tuned.csv', index=False)
    print("\nSaved: pa_a_tuned.csv")

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='PA&A tuning harness (align rules to report appendix)')
    ap.add_argument('--csv', default='cbrdataset.csv', help='Input CSV containing CBR column')
    ap.add_argument('--cbr-col', default='CBR', help='Column name for CBR')
    ap.add_argument('--veh-col', default=None, help='Optional column for vehicle count')
    ap.add_argument('--pdr-col', default=None, help='Optional column for packet delivery rate')
    args = ap.parse_args()
    run(args.csv, args.cbr_col, args.veh_col, args.pdr_col)