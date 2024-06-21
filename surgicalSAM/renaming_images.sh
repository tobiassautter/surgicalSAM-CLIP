#!/bin/bash

# Navigate to the images directory
cd ./images

# Loop through all seq* directories
for seqdir in seq*
do
    if [ -d "$seqdir" ]; then
        # Navigate into the sequence directory
        cd "$seqdir"

        # Loop through all .png files that start with 'frame'
        for file in frame*.png
        do
            # Check if the file actually exists
            if [ -e "$file" ]; then
                # Construct the new filename by stripping 'frame' and padding the number
                newname=$(printf "%05d.png" "${file#frame}")
                mv "$file" "$newname"
                echo "Renamed $file to $newname"
            fi
        done

        # Return to the parent images directory
        cd - > /dev/null
    fi
done

# Return to the original directory
cd - > /dev/null
