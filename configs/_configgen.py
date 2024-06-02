import os

# Base filenames and corresponding mus values
base_files = [ "configCbird1mu1", "configCbird2mu1", "configCbird3mu1", "configCbird4mu1", "configCbird5mu1", "configCbird6mu1", "configCbird7mu1"]
mus_values = [.20]

# Function to replace mus line in the content
def replace_mus_line(content, new_mu):
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'mus:' in line:
            lines[i] = f"  mus: [{new_mu}]"
            break
    return '\n'.join(lines)

# go to the directory where the script is located
os.chdir("/home/gauffrea/Projects/SMABalancingGroups/configs")

# Read, modify, and write the config files
for base_file in base_files:
    with open(f"{base_file}.yaml", 'r') as file:
        content = file.read()

    for mu in mus_values:
        new_content = replace_mus_line(content, mu)
        new_filename = f"{base_file[:-1]}{int(mu*100)}.yaml"
        
        with open(new_filename, 'w') as file:
            file.write(new_content)

print("Config files generated successfully.")
