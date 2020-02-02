import os
import sys
import time
import re

argv = sys.argv
del argv[0]

argvStr = " ".join(argv)

homedir = os.path.abspath(os.path.dirname(__file__))
homedir = re.sub("/python", "", homedir)
tumoppR = homedir + "/R/tumoppMrs.R"

if argvStr.find("-h") != -1:
	print("python tumoppMrs.py -rand (with random parameter settings)"\
			" -out outFilePrefix -itr numberOfIteration"\
			" -prm parameterFile tumoppArgs")
	exit(0)

outprefix = "out"
itr = 1
rand = False
parmfile = ""


pattern = r"-out\s+(\S+)"
match = re.search(pattern, argvStr)
if match:
	outprefix = os.path.abspath(match.group(1))
	argvStr = re.sub(pattern, "", argvStr)


pattern = r"-itr\s+(\S+)"
match = re.search(pattern, argvStr)
if match:
	itr = int(match.group(1))
	argvStr = re.sub(pattern, "", argvStr)

pattern = r"-rand"
match = re.search(pattern, argvStr)
if match:
	rand=True

pattern = r"-prm\s+(\S+)"
match = re.search(pattern, argvStr)
if match:
	prmfile = os.path.abspath(match.group(1))
	argvStr = re.sub(pattern, "", argvStr)

if os.path.exists(prmfile):
	f = open(prmfile)
	l = f.readline()
	tmp = []
	while l:
		tmp2 = l.rstrip().split("\t")
		tmp.append("-" + tmp2[0] + " " + tmp2[1])
		l = f.readline()
	argvStr = " ".join(tmp)

def runR(script, args):
	out = []
	for k, v in args.items():
		if type(v) is str:
			out.append(k + " <- as.character('" + v + "')")
		elif type(v) is int:
			out.append(k + " <- as.integer('" + str(v) + "')")
		elif type(v) is float:
			out.append(k + " <- as.numeric('" + str(v) + "')")
		elif type(v) is bool:
			out.append(k + " <- as.logical('" + str(v) + "')")
	out.append("source('" +  script + "')")
	tmpR = "tmp" + str(os.getpid()) + ".R"
	with open(tmpR, "w") as file:
		file.write("\n".join(out) + "\n")
	status = os.system("R --slave --args  < " + tmpR)
	os.system("rm " + tmpR)
	return status

args = {}
if argvStr == "" and rand == False:
	args["tumoppArgs"] = "-N 40000 -D 2 -C hex -k 100 -L const"
else:
	args["tumoppArgs"] =  argvStr
args["outprefix"] = outprefix
args["itr"] = itr
args["rand"] = rand
if runR(tumoppR, args):
	sys.exit(1)
