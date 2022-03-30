#/bin/bash

python -c '''import sys,os, subprocess, time, random

NJOBS =10 
jobid = [file for file in os.listdir("moab_jobs/")]

number_of_job_to_submit = len(jobid)
random.shuffle(jobid)

job_gen = (i for i in jobid)

running_jobs = int(subprocess.check_output("qstat | grep -e R -e Q\ | grep icu | wc -l", shell=True))
while True:
    if running_jobs < NJOBS:
        current_job  = next(job_gen)
        for moab_job in os.listdir("moab_jobs/"):
            if current_job in moab_job:
                number_of_job_to_submit -=1
                os.system("qsub moab_jobs/"+current_job)
                time.sleep(2)
        running_jobs = int(subprocess.check_output("qstat | grep -e R -e Q\ | grep icu | wc -l", shell=True))
    else:
        running_jobs = int(subprocess.check_output("qstat | grep -e R -e Q\ | grep icu | wc -l", shell=True))
        print ("more then {} jobs are running".format(running_jobs))
        time.sleep(60)'''
