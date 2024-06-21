#!/bin/bash

# Loop over directories 0 through 40
for dir in {0..40}
do
    # Check if the parent directory exists before proceeding
    if [ -d "$dir/sam_features_b" ]; then
        # Navigate into the parent directory
        cd "$dir/sam_features_b"

        # Loop through all seq* directories
        for seqdir in seq*
        do
            if [ -d "$seqdir" ]; then
                # Navigate into the sequence directory
                cd "$seqdir"

                # Loop through all npy related files
                for file in *.npy
                do
                    # Check if the file actually exists and if it's incorrectly named
                    if [[ -e "$file" && ! "$file" =~ ^[0-9]+\.npy$ ]]; then
                        # Correct the file name by constructing a new name
                        # Remove erroneous sequences like 'npy' or extra dots, and ensure the format is numeric followed by '.npy'
                        newname=$(echo "$file" | sed -E 's/([0-9]+).*\.npy$/\1.npy/')
                        mv "$file" "$newname"
                        echo "Renamed $file to $newname"
                    fi
                done

                # Return to the parent directory
                cd - > /dev/null
            fi
        done

        # Return to the root data directory
        cd - > /dev/null
    else
        echo "Directory $dir/sam_features_b does not exist."
    fi
done
