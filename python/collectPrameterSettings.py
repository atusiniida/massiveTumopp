import os
import glob
import argparse

parser = argparse.ArgumentParser()
help_ = "data directory"
parser.add_argument("-d", "--data", help=help_, default='data')
help_ = "result directory"
parser.add_argument("-r", "--res", help=help_,  default='result')
args = parser.parse_args()

indir = os.path.abspath(args.data)
outdir = os.path.abspath(args.res)
infiles = glob.glob(indir + "/*.prm")

fout = open(outdir + '/prm.tab', "w")
first = True
for f in infiles:
	fin = open(f)
	l = fin.readline()
	p = []
	v = []
	while l:
		tmp = l.rstrip().split("\t")
		p.append(tmp[0])
		v.append(tmp[1])
		l = fin.readline()
	fin.close()
	if first:
		fout.write("\t" + "\t".join(p) + "\n")
		first = False
	fout.write(os.path.splitext(os.path.basename(f))[0] + "\t" + "\t".join(v) + "\n")
fout.close()
