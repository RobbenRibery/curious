import toml

def extract_dependencies(pyproject_file_path):
    with open(pyproject_file_path, "r", encoding="utf-8") as file:
        pyproject = toml.load(file)

    # Access the Poetry section
    poetry_section = pyproject.get("tool", {}).get("poetry", {})
    dependencies = poetry_section.get("dependencies", {})

    return dependencies

if __name__ == "__main__":
    pyproject_path = "pyproject.toml"  # Adjust if needed
    deps = extract_dependencies(pyproject_path)
    print("Project dependencies:")
    for name, version in deps.items():
        print(f"• {name}: {version}")