#!/bin/bash

# Absolute path to the images directory
cd "$(dirname "$0")/images" || exit

# Loop through all seq* directories
for seqdir in seq*/; do
    if [ -d "$seqdir" ]; then
        # Store the current directory
        pushd "$seqdir" > /dev/null || continue

        # Loop through all .png files that start with 'frame'
        for file in frame*.png; do
            # Check if the file is already in the desired format
            if [[ ! $file =~ ^[0-9]{5}\.png$ ]]; then
                # Extract the numeric part, assuming the format 'frame000.png'
                num_part="${file#frame}"
                num_part="${num_part%.png}"

                # Correctly interpret as decimal (avoiding leading zeros causing octal interpretation)
                # Strip leading zeros for safety before converting to decimal
                num_part=$(echo $num_part | sed 's/^0*//')

                # Use printf to format the number as a five-digit decimal
                newname=$(printf "%05d.png" "$num_part")
                if mv "$file" "$newname"; then
                    echo "Renamed $file to $newname"
                else
                    echo "Failed to rename $file"
                fi
            else
                echo "File $file is already in the correct format."
            fi
        done

        # Return to the parent images directory
        popd > /dev/null
    fi
done

