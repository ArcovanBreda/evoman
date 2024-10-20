#!/bin/bash

start_time=$SECONDS

etr_values=("1,2,3,4,5,6,7,8" "2,3,5,7")
max_jobs=16  # Maximum number of parallel jobs

for etr in "${etr_values[@]}"; do
    echo "Running experiments with -etr: $etr"
    
    for i in {1..10}; do
        exp_name="run${i}"
        echo "Running experiment with: $exp_name and -etr: $etr"
        # python generalist_hannah.py -k -m uncorrelated -ff gradual -ms 265 -t -exp "$exp_name" -etr "$etr" -ps 100 -tg 100 -hi -mp 0.3130 -s 0.08550 -g 29,40,64,70 -mt 0.6970&
        python CMA.py -exp "$exp_name" -ps 100 -tg 1000 -k -s 0.2 -etr "$etr" -t -ff steps -se "$i"&
        if (( $(jobs -r | wc -l) >= max_jobs )); then
            wait -n
        fi
    done
    
    # Wait for all background jobs to finish before continuing to the next etr
    wait
done

elapsed_time=$(( SECONDS - start_time ))

echo "Total time taken: $elapsed_time seconds"

# chmod +x CMA_step.sh
# ./CMA_step.sh