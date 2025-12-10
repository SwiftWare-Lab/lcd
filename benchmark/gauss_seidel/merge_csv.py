import glob
import os

def merge_csv_files():
    # 1. Find all matching files
    pattern = "gauss-seidel-performance-*.csv"
    files = glob.glob(pattern)
    files.sort() # Sort to ensure deterministic order (optional)

    output_filename = "gauss-seidel-performance-raw.csv"

    print(f"Found {len(files)} files. Merging into {output_filename}...")

    with open(output_filename, 'w') as outfile:
        # 2. Loop through all files
        for i, filename in enumerate(files):
            with open(filename, 'r') as infile:
                header = infile.readline()
                data = infile.readline()

                # 3. Write header only for the very first file
                if i == 0:
                    outfile.write(header)
                
                # 4. Write data line for every file
                # Ensure it ends with a newline if missing
                if data:
                    if not data.endswith('\n'):
                        data += '\n'
                    outfile.write(data)

    print("Done.")

if __name__ == "__main__":
    merge_csv_files()
