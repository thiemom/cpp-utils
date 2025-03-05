# ConfigLoader

A modern C++ configuration loader with type safety and validation support. This library provides a flexible and easy-to-use interface for loading and managing configuration values from text files.

## Features

- **Type-Safe Access**: Strongly typed configuration values with compile-time type checking
- **Automatic Type Detection**: Automatically detects and converts values to appropriate types
- **Validation Support**: Built-in validation rules for numbers and strings
- **Flexible Value Types**: Supports:
  - Integers
  - Floating-point numbers
  - Strings
  - Arrays (comma-separated values)
- **Error Handling**: Comprehensive error reporting with line numbers
- **Comments Support**: Ignores comment lines starting with '#'

## Usage

### Basic Example

```cpp
#include "config.h"
#include <iostream>

int main() {
    try {
        ConfigLoader config;
        config.load("server.conf");

        // Access values with type safety
        int port = config.get<int>("port");
        std::string host = config.getOr<std::string>("host", "localhost");
        auto allowed_ips = config.get<std::vector<std::string>>("allowed_ips");

        std::cout << "Server configuration loaded:\n"
                  << "Port: " << port << "\n"
                  << "Host: " << host << "\n"
                  << "Allowed IPs: " << allowed_ips.size() << " entries\n";
    } catch (const ConfigError& e) {
        std::cerr << "Configuration error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
```

### Configuration File Format

```ini
# Server Configuration
port 8080
host localhost
allowed_ips 192.168.1.1,192.168.1.2,192.168.1.3
max_connections 100
timeout 30.5
```

### Adding Validation Rules

```cpp
ConfigLoader config;

// Validate port number range
config.addRangeRule<int>("port", 1024, 65535);

// Validate email format
config.addPatternRule("email", "[\\w.-]+@[\\w.-]+\\.[a-zA-Z]{2,}");

// Custom validation rule
config.addValidationRule(
    "max_connections",
    [](const ConfigLoader::Value& v) {
        try {
            int val = std::get<int>(v);
            return val > 0 && val <= 1000;
        } catch (const std::bad_variant_access&) {
            return false;
        }
    },
    "max_connections must be between 1 and 1000"
);
```

## API Reference

### Core Methods

- `bool load(const std::string& filename)`: Load configuration from a file
- `T get<T>(const std::string& key)`: Get a value with type checking
- `T getOr<T>(const std::string& key, const T& defaultValue)`: Get a value with default
- `bool hasKey(const std::string& key)`: Check if a key exists

### Validation Methods

- `void addRangeRule<T>(const std::string& key, T min, T max)`: Add numeric range validation
- `void addPatternRule(const std::string& key, const std::string& pattern)`: Add regex pattern validation
- `void addValidationRule(const std::string& key, ValidationCallback callback, const std::string& errorMessage)`: Add custom validation

## Error Handling

The library uses the `ConfigError` exception class which provides:
- Detailed error messages
- Line numbers for parsing errors
- Type mismatch information
- Validation failure details

## Building

This is a header-only library. Simply include `config.h` in your project.

## Requirements

- C++17 or later
- Standard Library support for:
  - `std::variant`
  - `std::optional`
  - Regular expressions

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
