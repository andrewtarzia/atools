import os
import sys


def main():
    file = sys.argv[1]
    out_file = file.replace('.log', '.grep')

    grep_cmd = f"grep 'RB3LYP)' {file} | grep 'A.U.' > {out_file}"
    os.system(grep_cmd)

    with open(out_file, 'r') as f:
        lines = f.readlines()

    vals = [float(i.split('  ')[2]) for i in lines]
    print(f'current energy = {vals[-1]}')
    print(
        f'minimum energy of {min(vals)} at '
        f'{vals.index(min(vals))+1} of {len(vals)}'
    )


if __name__ == '__main__':
    main()
