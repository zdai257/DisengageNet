import subprocess
import os


def run_openface(input_path, output_path="data", docker_image="algebr/openface"):
    """
    Runs the OpenFace feature extractor inside a Docker container.

    Args:
        input_path (str): Path to the input video or image.
        output_path (str): Mount output directory
        docker_image (str): Name of the Docker image for OpenFace.

    Returns:
        None
    """
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    print(os.path.dirname(input_path), os.path.basename(input_path))
    print(os.path.abspath(input_path))
    print(os.path.abspath(output_path))
    print()

    # Adjust the mount paths to align with the container's requirements
    command = [
        "docker", "run", "--rm",
        "-p", "8002:8002",
        "--entrypoint", "/home/openface-build/build/bin/FeatureExtraction",  # FeatureExtraction, FaceLandmarkImg
        "-v", f"{os.path.abspath(input_path)}:/data",  # Mount input as
        #"-v", f"{os.path.abspath(output_path)}:/home/openface-build/processed",  # Mount output as
        docker_image,
        #"/home/openface-build/build/bin/FeatureExtraction",
        "-f", f"/data/{os.path.basename(input_path)}", "-out_dir", "/data/processed"
    ]

    # Run the command and capture output
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Check for errors
    if result.returncode != 0:
        raise RuntimeError(f"OpenFace failed: {result.stderr}")

    print(result.stdout)
    return result


def call_openface(input_path, outdir="processed", docker_image="algebr/openface", entry="/home/openface-build/build/bin/FeatureExtraction"):

    with open("/tmp/output.log", "a") as output:
        result = subprocess.call(f"docker run --rm -p 8001:8001 --entrypoint {entry} -v {os.path.abspath(input_path)}:/data {docker_image} -f /data/{os.path.basename(input_path)} -out_dir /data/{outdir}",
                                 shell=True, stdout=output, stderr=output)

    return result

input_file = "data/joye.jpg"
output_path = "output"
results = run_openface(input_file)

#res = call_openface(input_file)
