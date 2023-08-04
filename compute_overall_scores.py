import argparse
import pandas as pd
from collections import Counter
from sklearn.metrics import f1_score

def compute_results(filename):
    df = pd.read_csv(filename, sep=",")
    df['session'] = df['ID'].apply(lambda x: x.split('_')[1])

    # This is "macro" F1 as it's computed based on the definition of macro F1
    # but it's "micro" F1 in the sense that the this computation accounts for 
    # individual samples (as opposed to obtaining F1 per session, then averaging)
    micro_f1 = f1_score(y_true=df.labels, y_pred=df.predictions, average='macro')
    
    new_df = []
    for session, df_sess in df.groupby('session'):
        sess_id = int(session)
        num_samples = len(df_sess)
        label = df_sess.labels.values[0]
        cpredictions = Counter(df_sess.predictions)
        voted_pred = cpredictions.most_common(1)[0][0]
        count_correct = (df_sess.predictions == df_sess.labels).sum()
        confidence = count_correct / num_samples
        new_df.append({"session": sess_id, 
                       "num_samples": num_samples,
                       "label": label,
                       "voted_pred": voted_pred,
                       "confidence": confidence,
                       "count_1": cpredictions[1],
                       "count_0": cpredictions[0]})
    
    new_df = pd.DataFrame(new_df)
    voted_f1 = f1_score(y_true=new_df.label, y_pred=new_df.voted_pred, average='macro')
    confidence_mean = new_df.confidence.mean()
    confidence_std = new_df.confidence.std()
    
    print(f"MicroF1: {micro_f1:.4f}")
    print(f"VotedF1: {voted_f1:.4f}")
    print(f"ConfidenceStats: {confidence_mean:.4f} (+/-{confidence_std:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default="../MI_data/predictions.csv")
    args = parser.parse_args()

    compute_results(args.results)

