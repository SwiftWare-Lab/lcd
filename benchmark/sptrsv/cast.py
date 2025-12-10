input_file = "matrix_list.txt"
output_file = "matrix_list_1.txt"

old_prefix = "/home/mad29/projects/testdata/matrices/"
new_prefix = "/scratch/dama/testdata/matrices/"

with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for line in fin:
        # Strip quotes, commas, whitespace
        cleaned = line.strip().strip('",')

        # Replace the prefix
        cleaned = cleaned.replace(old_prefix, new_prefix)

        # Write new path
        fout.write(cleaned + "\n")
