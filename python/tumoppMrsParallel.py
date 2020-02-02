import os
import sys
import time
import re
import subprocess

outdir = "out"
N = 100 # Number of jobs
n = 10  # Number of simulations per job
argv = sys.argv
argvStr = " ".join(argv[1:])


if argvStr.find("-h") != -1:
	print( 'Usage: # python %s  [-o outdir -N #ofJobs -n #ofSimulationsPerJob]' % argv[0])
	exit(0)

pattern = r"-o\s+(\S+)"
match = re.search(pattern, argvStr)
if match:
	outdir = os.path.abspath(match.group(1))
	argvStr = re.sub(pattern, "", argvStr)

pattern = r"-N\s+(\S+)"
match = re.search(pattern, argvStr)
if match:
    N = int(match.group(1))
    argvStr = re.sub(pattern, "", argvStr)

pattern = r"-n\s+(\S+)"
match = re.search(pattern, argvStr)
if match:
    n = int(match.group(1))
    argvStr = re.sub(pattern, "", argvStr)


if not os.path.exists(outdir) :
	os.system("mkdir " + outdir)

homedir = os.path.abspath(os.path.dirname(__file__))
homedir = re.sub("/python", "", homedir)
tumopp = homedir + "/python/tumoppMrs.py"

pid = os.getpid() #get process id
prefix = "tmp"  + str(pid)

# define qsub function

def printUGEscript(command, scriptfile):
	env = ["PATH", "PERL5LIB", "R_LIBS", "CLASSPATH", "LD_LIBRARY_PATH"]

	out = []
	out.append('#! /usr//bin/perl')
	out.append('#$ -S /usr/bin/perl')

	for x in env:
		if x in os.environ:
			out.append('$ENV{' + x + '}="' + os.environ[x] + '";')

	out.extend([ "warn \"command : " + command + "\\n\";",
				 "warn \"started @ \".scalar(localtime).\"\\n\";",
				 "if(system (\"" + command + "\" )){",
				 "die \"failed @ \".scalar(localtime).\"\\n\";",
				 "}else{",
				"warn \"ended @ \".scalar(localtime).\"\\n\";",
				"}", ])
	with open(scriptfile, "w") as file:
		file.write("\n".join(out) + "\n")

# wait until # of jobs is less than cutoff

def waitForUGEjobFinishing(target, cutoff=1):
    target = target[0:9]
    while True:
        qstat = ""
        try:
            qstat = subprocess.check_output("qstat").decode('utf-8')
        except:
            continue
        qstat = qstat.rstrip().split("\n")
        greped = list(filter( lambda x: re.search(target,x), qstat))
        if len(greped) < cutoff:
            return
        else:
            time.sleep(10)
sys.stderr.write("submit jobs...\n")
mem = 8
user = "niiyan"
maxjob = 3000

for i in range(N):
    prefix2  =  outdir + "/" + prefix + "." + str(i+1)
    prefix3 =  outdir + "/" + str(i+1)
    command = "python " + tumopp +  " -rand -out " + prefix3  + "  -itr " + str(n)
    scriptfile = prefix2 + ".pl"
    outfile = prefix2 + ".o"
    errfile = prefix2 + ".e"
    printUGEscript(command, scriptfile) # write script
    qsub =  " ".join(["qsub", "-cwd",  "-N",  prefix + "." + str(i+1), "-l",  "s_vmem=" + str(mem) + "G,mem_req=" + str(mem) + "G,os7", "-o",  outfile, "-e", errfile, scriptfile, "> /dev/null"])
    waitForUGEjobFinishing(user, maxjob)
    sys.stderr.write(qsub + "\n")
    while  os.system(qsub)>0:
        time.sleep(10)

waitForUGEjobFinishing(prefix)
