import os
import pandas as pd
import json
import numpy as np
import argparse
import sys
import seaborn as sns
import matplotlib.pyplot as plt

module_path = os.path.abspath("..")
if module_path not in sys.path:
    sys.path.append(module_path)

from exps import split_student_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="../dataset/CodeWorkout")
    parser.add_argument("--result_dir", type=str, default="../results/data")

    args = parser.parse_args()

    dataset_root_dir = args.dataset_dir

    submission_path = os.path.join(dataset_root_dir, "MainTable.csv")
    submission_df = pd.read_csv(submission_path)
    submission_df = submission_df[submission_df["EventType"] == "Run.Program"]
    submission_df = submission_df.astype({"AssignmentID":'int32'})
    print(f"submission_df: {submission_df.shape}")

    semester_splited_student_ids, overall_student_ids = split_student_ids.load_ids(args.result_dir)

    def get_seq_len_stat(df, student_ids, histo_fig_fname):
        problem_list = sorted(list(df["ProblemID"].unique()))

        seq_len_per_student = df.groupby('SubjectID').size().rename('seq_len')

        mean_plus_2std = seq_len_per_student.mean() + (2 * seq_len_per_student.std())
        seq_len_per_student_stats = {
            "mean_plus_2std": float(mean_plus_2std)
        }

        percentiles = seq_len_per_student.quantile([0.5, 0.95, 0.97, 0.99]).to_dict()
        seq_len_per_student_stats["percentiles"] = {float(k):int(v) for k, v in percentiles.items()}

        seq_len_stat = {
            "n_problem": len(problem_list),
            "seq_len_per_student": seq_len_per_student_stats
        }

        fig, ax1 = plt.subplots(figsize=(10, 6))

        sns.histplot(seq_len_per_student, bins=30, kde=False, ax=ax1, color='skyblue')

        ax1.set_title('Student Submission Distribution', fontsize=14)
        ax1.set_xlabel('Submission Count (seq_len)')
        ax1.set_ylabel('Number of Students (Count)')

        ax2 = ax1.twinx()

        total = len(seq_len_per_student)
        ax2.set_ylim(0, ax1.get_ylim()[1] / total * 100)
        ax2.set_ylabel('Percentage (%)')

        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        import matplotlib as mpl
        mpl.rcParams['pdf.fonttype'] = 42
        for ext in ["pdf", "png"]:
            fig.savefig(f"{histo_fig_fname}.{ext}", 
                        dpi=300,
                        bbox_inches='tight',
                        pad_inches=0)

        return seq_len_stat

    total_seq_len_stat = {}

    histo_fig_fname = os.path.join(args.result_dir, "seq_len_histo_overall")
    overall_seq_len_stat = get_seq_len_stat(submission_df, overall_student_ids, histo_fig_fname)
    total_seq_len_stat["overall"] = overall_seq_len_stat

    for assignment, splited_student_ids in semester_splited_student_ids.items():
        assignment_df = submission_df[submission_df['AssignmentID'] == assignment]

        histo_fig_fname = os.path.join(args.result_dir, f"seq_len_histo_{assignment}")

        problem_list = sorted(list(assignment_df["ProblemID"].unique()))

        seq_len_stat = get_seq_len_stat(assignment_df, splited_student_ids, histo_fig_fname)
        total_seq_len_stat[assignment] = seq_len_stat

    with open(os.path.join(args.result_dir, "seq_len_stat.json"), "w") as fout:
        json.dump(total_seq_len_stat, fout, ensure_ascii=False, indent=3)
