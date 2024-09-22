#!/bin/bash

start_time=$SECONDS

etr_values=(1 3 5)

for etr in "${etr_values[@]}"; do
    echo "Running experiments with -etr: $etr"
    
    for i in {1..10}; do
        exp_name="run${i}"
        echo "Running experiment with: $exp_name and -etr: $etr"
        
        python specialist_silvia.py -k -m correlated -t -exp "$exp_name" -etr "$etr"
        
        sleep 1
    done
done

elapsed_time=$(( SECONDS - start_time ))

echo "Total time taken: $elapsed_time seconds"

# chmod +x experiments_correlated.sh
# ./experiments_correlated.sh