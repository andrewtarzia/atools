import os
#import matplotlib.pyplot as plt
import sys


def main():
    file = sys.argv[1]
    out_file = file.replace('.log', '.grep')

    grep_cmd = f"grep 'RB3LYP)' {file} > {out_file}"
    os.system(grep_cmd)

    with open(out_file, 'r') as f:
        lines = f.readlines()

    vals = [float(i.split('  ')[2]) for i in lines]
    print(f'current energy = {vals[-1]}')
    print(f'minimum energy of {min(vals)} at {vals.index(min(vals))+1} of {len(vals)}')

 #   fig, ax = plt.subplots()
  #  ax.plot([i-min(vals) for i in vals], c='k')
   # ax.tick_params(axis='both', which='major', labelsize=16)
    #ax.set_xlabel('step', fontsize=16)
    #ax.set_ylabel('energy', fontsize=16)
    #fig.tight_layout()
    #fig.savefig("energy_plot.pdf", dpi=720, bbox_inches='tight')
    #plt.close()


if __name__ == '__main__':
    main()

