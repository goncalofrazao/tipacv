def parse_config_file(file_path):
    config_dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            # Ignore comments
            if line.startswith('#'):
                continue

            # Split the line into tokens
            tokens = line.split()

            if len(tokens) == 0:
                continue
            
            # Extract parameter names and values
            param_name = tokens[0]
            param_values = [tokens[1:]]

            # Check if the token already exists in the dictionary
            if param_name in config_dict:
                # Add new values to the existing token
                config_dict[param_name].extend(param_values)
            else:
                # Create a new entry in the dictionary
                config_dict[param_name] = param_values

    return config_dict
