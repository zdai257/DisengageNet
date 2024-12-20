#!/bin/bash

# Check if Docker is installed
docker --version >/dev/null 2>&1 || { echo "Docker is not installed. Please install Docker first."; exit 1; }

# Define input and output directories
INPUT_DIR="/home/CAMPUS/daiz1/Documents/DAiSEE"
OUTPUT_DIR="./output"

# Ensure the input directory exists
if [ ! -d "$INPUT_DIR" ]; then
  echo "Input directory $INPUT_DIR does not exist. Please provide a valid directory."
  exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Pull the algebr/openface image (if not already pulled)
echo "Pulling the algebr/openface Docker image..."
docker pull algebr/openface

# Loop through all files in the input directory and its subdirectories
find "$INPUT_DIR" -type f | while read -r file; do
  # Determine file extension and process accordingly: for WALI-HRI -- '*master1.mkv'; for DAiSEE -- '.avi'
  #if [[ $file == *master1.mkv ]]; then
  if [[ $file == *master1.mkv || $file == *.mp4 || $file == *.avi || $file == *.mov || $file == *.mkv || $file == *.jpg || $file == *.jpeg || $file == *.png ]]; then
    echo "Processing $file..."

    # Extract relative path and filename
    rel_path="${file#$INPUT_DIR/}"
    filename=$(basename -- "$file")
    filename_noext="${filename%.*}"
    output_subdir="$OUTPUT_DIR/$(dirname "$rel_path")"

    outdir_base=$(basename -- "$OUTPUT_DIR")

    # Create the corresponding output subdirectory if it doesn't exist
    mkdir -p "$output_subdir"

    # Run the Docker container to process the file
    docker run --rm -v "$INPUT_DIR":/input -v "$PWD/$outdir_base":/output \
    --entrypoint /home/openface-build/build/bin/FeatureExtraction algebr/openface -aus -gaze -pose \
    -f "/input/$rel_path" -out_dir "/output/$(dirname "$rel_path")"
  else
    echo "Skipping unsupported file $file"
  fi
done

echo "Processing complete. Results saved in $OUTPUT_DIR."
