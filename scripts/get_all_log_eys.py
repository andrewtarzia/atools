import os
import glob


def main():
    files = sorted(glob.glob('*.log'))

    min_min_energy = (None, 1E24, None)
    for file in files:
        out_file = file.replace('.log', '.grep')
        grep_cmd = (
            f"grep 'RPBE1PBE)' {file}  | grep 'A.U.' > {out_file}"
        )
        os.system(grep_cmd)

        with open(out_file, 'r') as f:
            lines = f.readlines()

        vals = [float(i.split('  ')[2]) for i in lines]
        for i, val in enumerate(vals):
            if val < min_min_energy[1]:
                min_min_energy = (file, val, i)
        print(
            f'{file}: energy = {vals[-1]}; min energy {min(vals)} at '
            f'{vals.index(min(vals))+1} of {len(vals)}'
        )

    print('min of min (file, energy, step):', min_min_energy)


if __name__ == '__main__':
    main()
