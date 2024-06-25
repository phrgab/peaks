import toml

# Define the path to your pyproject.toml file
pyproject_path = '../pyproject.toml'

# Load the pyproject.toml file
with open(pyproject_path, 'r') as f:
    pyproject = toml.load(f)

# Extract the dependencies and dev dependencies
dependencies = pyproject['project']['dependencies']
dev_dependencies = pyproject['project']['optional-dependencies']['dev']

# Function to format dependencies
def format_dependencies(deps):
    formatted_deps = []
    for dep in deps:
        print(dep)
        # Split the dependency into name and version
        try:
            name, version = dep.split(' ')
            # Remove the parentheses from the version
            version = version.strip('()')
            # Replace the comma with nothing
            version = version.replace(',', '')
            # Add the formatted dependency to the list
            formatted_deps.append(f'{name}{version}')
        except ValueError:
            name = dep
            formatted_deps.append(f'{name}')

    return formatted_deps

# Format the dependencies and dev dependencies
dependencies = format_dependencies(dependencies)
dev_dependencies = format_dependencies(dev_dependencies)

# Write the dependencies and dev dependencies to the requirements.txt file
with open('requirements.txt', 'w') as f:
    for dep in dependencies + dev_dependencies:
        f.write(f'{dep}\n')