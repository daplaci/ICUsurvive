import sys, os, subprocess, time

print ("start watching.. ")

max_num_rerun = 2
 
while True:
    if not os.listdir("moab_jobs/") or not max_num_rerun:
        print ('Finished re-running') 
        break
    job_running = int(subprocess.check_output("qstat | grep -e R -e Q\ | grep icu | wc -l", shell=True)) > 0 
    if not job_running:
        if  os.listdir("moab_jobs/"):
            print ("Trying to rerun again job left..")
            subprocess.call("./run_models.sh", shell=True)
            # there is a problem with this approach:
            # the script that failed may already have saved some results to the output files
            max_num_rerun-=1
        else:
            break
    time.sleep(60)

#after this loop all the jobs can run are launched. now the second round of results can start
     
