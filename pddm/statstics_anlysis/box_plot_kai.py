import numpy as  np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import argparse
import os
from statannot import add_stat_annotation
import time

parser = argparse.ArgumentParser()
parser.add_argument(
        '--job_path1', type=str,
        default='../output/cheetah')  #address this WRT working directory
parser.add_argument(
        '--job_path2', type=str,
        default='../output/cheetah')
parser.add_argument(
        '--job_path3', type=str,
        default='../output/cheetah')
parser.add_argument(
        '--save_dir', type=str,
        default='../output/cheetah')
parser.add_argument(
        '--data_type', type=str,
        default='rewards')
parser.add_argument(
        '--save_num', type=str,
        default='1')
args = parser.parse_args()

if not os.path.isdir(args.save_dir):
    os.makedirs(args.save_dir)
sns.set()
sns.set_style('whitegrid')
sns.set_palette('Set3')




mbrlf_data=[]
mbrl_data=[]
pid_data =[]

if args.data_type=="eei":
    for i in range(10):
        mbrlf_data.append(np.load(args.job_path1 + "/eval_eeis_{}.npy".format(i)))
        mbrl_data.append(np.load(args.job_path2 + "/eval_eeis_{}.npy".format(i)))
        pid_data.append(np.load(args.job_path3 + "/eval_eeis_{}.npy".format(i)))
elif args.data_type=="rewards":
    for i in range(10):
        mbrlf_data.append(np.load(args.job_path1 + "/eval_rewards_{}.npy".format(i)))
        mbrl_data.append(np.load(args.job_path2 + "/eval_rewards_{}.npy".format(i)))
        pid_data.append(np.load(args.job_path3 + "/eval_rewards_{}.npy".format(i)))


mbrlf_data=np.array(mbrlf_data).flatten()
mbrl_data=np.array(mbrl_data).flatten()
pid_data=np.array(pid_data).flatten()
#print("mbrlf_data max {}".format(max(mbrlf_data)))
#print("mbrl_data max {}".format(max(mbrl_data)))
#print("pid max {}".format(max(pid_data)))
df = pd.DataFrame({
    'mbrl_f': mbrlf_data,
    'mbrl_nf': mbrl_data,
    'pid': pid_data
})
#df_melt=df
df_melt = pd.melt(df)
#print(df_melt.head())
print(df_melt)
##   variable      value
## 0     leaf   9.446465
## 1     leaf  11.163702
## 2     leaf  14.296799
## 3     leaf   7.441026
## 4     leaf  11.004554

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
my_pal = {"pid": "g", "mbrl_f": "b", "mbrl_nf":"r"}
sns.stripplot(x='variable', y='value', data=df_melt, jitter=True, color="black", ax=ax)
#sns.boxplot(x='variable', y='value', data=df_melt, showfliers=True, ax=ax, palette=my_pal )
sns.boxplot(x='variable', y='value', data=df_melt, showfliers=True, ax=ax,showmeans=True, palette=my_pal ,meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue"})
#sns.violinplot(x='variable', y='value', data=df_melt, showfliers=True, ax=ax,showmeans=True, palette=my_pal ,meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue"})
ax.set_xlabel('controller')
if args.data_type=="eei":
    ax.set_ylabel('EEI')
elif args.data_type=="rewards":
    ax.set_ylabel('Rewards')
order = ['mbrl_f', 'mbrl_nf', 'pid']
test_results = add_stat_annotation(ax, data=df_melt, x='variable', y='value', order=order,
                                   box_pairs=[('mbrl_f', 'mbrl_nf'), ('mbrl_nf', 'pid'), ('pid', 'mbrl_f')],
                                   test='t-test_welch', text_format='simple',
                                   loc='outside', verbose=2)
#test_results
#handler, label = ax.get_legend_handles_labels()
#ax.legend(handler,["mbrl_f","mbrl_nf","pid"],loc='upper left',bbox_to_anchor=(1.05,1))
#ind = 0
#plt.ion()
#plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
plt.savefig(args.save_dir + "/{}_{}".format(args.data_type,args.save_num),bbox_inches='tight')

def main():

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--jobs', action='append', nargs='+', help='job/experiment')
    parser.add_argument('-l', '--labels', action='append', nargs='?', help='label for plotting that experiment')
    parser.add_argument('-plot_rew', '--plot_rew', action='store_true')
    parser.add_argument(
        '--save_dir', type=str,
        default='../output/cheetah')
    parser.add_argument(
        '--data_type', type=str,
        default='rewards')
    parser.add_argument(
        '--save_num', type=str,
        default='1')

    args = parser.parse_args()
    jobs = args.jobs[0]

    # scan labels
    if args.labels is not None:
        assert (len(jobs)==len(args.labels)), "The number of labels has to be same as the number of jobs"
    else:
        args.labels = ['']*len(jobs)

    # Scan jobs and plot
    colors=['b','r', 'g',  'k', 'c', 'm', 'pink', 'purple']
    all_data=[]
    for i in range(len(jobs)):
        if args.plot_rew:
            print("LOOKING AT REW")
            data=[]
            for j in range(10):
                data.append(np.load(jobs[i] + "/eval_eeis_{}.npy".format(j)))
            data=np.array(data).flatten()
            all_data.append(data)
        else:
            data = []
            for j in range(10):
                data.append(np.load(jobs[i] + "/eval_ewards_{}.npy".format(j)))
            data = np.array(data).flatten()
            all_data.append(data)
    data_dic={"data{}".format(i):all_data[i] for i in range(len())}
    plt.savefig(args.save_dir + "/{}_{}".format(args.data_type, args.save_num), bbox_inches='tight')
    #plt.show()

if __name__ == '__main__':
    main()