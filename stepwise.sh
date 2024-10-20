#!/bin/bash

start_time=$SECONDS

etr_values=("1,2,3,4,5,6,7,8" "2,3,5,7")
max_jobs=16  # Maximum number of parallel jobs

for etr in "${etr_values[@]}"; do
    echo "Running experiments with -etr: $etr"
    
    for i in {1..10}; do
        exp_name="run${i}"
        echo "Running experiment with: $exp_name and -etr: $etr"
        python generalist_hannah.py -k -m uncorrelated -ms 265 -t -exp "$exp_name" -etr "$etr" -ps 100 -tg 100 -hi -mp 0.6984240873524129 -s 0.3315383180076688 &
        
        if (( $(jobs -r | wc -l) >= max_jobs )); then
            wait -n  # Wait for the first job to finish before starting a new one
        fi
    done
    
    # Wait for all background jobs to finish before continuing to the next etr
    wait
done

elapsed_time=$(( SECONDS - start_time ))

echo "Total time taken: $elapsed_time seconds"

# chmod +x stepwise.sh
# ./stepwise.sh