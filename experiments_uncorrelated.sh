#!/bin/bash

start_time=$SECONDS

etr_values=(1 4 7)
max_jobs=16  # Maximum number of parallel jobs

for etr in "${etr_values[@]}"; do
    echo "Running experiments with -etr: $etr"
    
    for i in {1..10}; do
        exp_name="run${i}"
        echo "Running experiment with: $exp_name and -etr: $etr"
        
        python specialist_silvia.py -k -m uncorrelated -t -exp "$exp_name" -etr "$etr" -ms 265 -tg 200&
        
        if (( $(jobs -r | wc -l) >= max_jobs )); then
            wait -n  
        fi
    done
    
    # Wait for all background jobs to finish before continuing to the next etr
    wait
done

elapsed_time=$(( SECONDS - start_time ))

echo "Total time taken: $elapsed_time seconds"

# chmod +x experiments_uncorrelated.sh
# ./experiments_uncorrelated.sh