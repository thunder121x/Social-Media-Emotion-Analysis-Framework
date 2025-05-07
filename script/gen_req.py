import subprocess
import os

def create_requirements_txt():
    """
    Generates a requirements.txt file by freezing the currently installed packages
    in the active Python environment.  It handles potential errors during the
    pip freeze process and checks if the file already exists.
    """
    try:
        # Use subprocess.run for better control and error handling
        result = subprocess.run(['pip', 'freeze'], capture_output=True, text=True, check=True)
        packages = result.stdout
        if not packages:
            print("No packages found in the current environment.")
            return

        # Specify the file path
        file_path = "requirements.txt"

        # Check if the file exists before writing
        if os.path.exists(file_path):
            print(f"Warning: {file_path} already exists and will be overwritten.")

        # Write the packages to the requirements.txt file
        with open(file_path, 'w') as f:
            f.write(packages)
        print(f"Successfully created {file_path}")

    except subprocess.CalledProcessError as e:
        print(f"Error generating requirements.txt: {e}")
        print(f"  Return code: {e.returncode}")
        print(f"  Standard error:\n{e.stderr}")
        print("  Please ensure that pip is installed and accessible in your environment.")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")

if __name__ == "__main__":
    create_requirements_txt()
