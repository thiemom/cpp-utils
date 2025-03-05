#ifndef CONFIG_LOADER_HPP
#define CONFIG_LOADER_HPP

#include <string>
#include <unordered_map>
#include <vector>
#include <variant>
#include <optional>
#include <functional>
#include <stdexcept>

/**
 * @brief Custom exception class for configuration-related errors
 * Includes line number information for better error reporting
 */
class ConfigError : public std::runtime_error {
public:
    explicit ConfigError(const std::string& msg, size_t line = 0)
        : std::runtime_error(line ? msg + " at line " + std::to_string(line) : msg)
        , lineNumber(line) {}
    size_t getLine() const { return lineNumber; }
private:
    size_t lineNumber;
};

/**
 * @brief Configuration loader class that supports type-safe access and validation
 * Supports various data types including integers, floats, strings, and vectors
 *
 * Example usage:
 * @code
 * ConfigLoader config;
 * 
 * // Add validation rules
 * config.addRangeRule<int>("port", 1024, 65535);
 * config.addPatternRule("email", "[\w.-]+@[\w.-]+\.[a-zA-Z]{2,}");
 * 
 * try {
 *     config.load("server.conf");
 *     
 *     // Access values with type safety
 *     int port = config.get<int>("port");
 *     std::string host = config.getOr<std::string>("host", "localhost");
 *     auto ips = config.get<std::vector<std::string>>("allowed_ips");
 * } catch (const ConfigError& e) {
 *     std::cerr << "Config error: " << e.what() << std::endl;
 * }
 * @endcode
 */
class ConfigLoader {
public:
    using Value = std::variant<int, float, std::string, std::vector<int>, std::vector<float>, std::vector<std::string>>;
    using ValidationCallback = std::function<bool(const Value&)>;

    struct ValidationRule {
        ValidationCallback callback;
        std::string errorMessage;
    };

    /**
     * @brief Type-safe getter that throws on type mismatch
     * @param key The configuration key to look up
     * @return The value cast to the requested type
     * @throws ConfigError if key not found or type mismatch
     *
     * Example:
     * @code
     * // Integer access
     * int port = config.get<int>("port");
     * 
     * // String access
     * std::string host = config.get<std::string>("host");
     * 
     * // Vector access
     * auto ips = config.get<std::vector<std::string>>("allowed_ips");
     * @endcode
     */
    template<typename T>
    T get(const std::string& key) const {
        try {
            return std::get<T>(get(key));
        } catch (const std::bad_variant_access&) {
            throw ConfigError("Type mismatch for key: " + key);
        }
    }

    /**
     * @brief Type-safe getter with default value
     * @param key The configuration key to look up
     * @param defaultValue Value to return if key not found
     * @return The configuration value or defaultValue if not found
     */
    template<typename T>
    T getOr(const std::string& key, const T& defaultValue) const {
        return hasKey(key) ? get<T>(key) : defaultValue;
    }

    /**
     * @brief Loads configuration from a file
     * @param filename Path to the configuration file
     * @return true if loading successful
     * @throws ConfigError on file access or parsing errors
     *
     * Example config file format:
     * @code
     * # Server configuration
     * port 8080
     * host localhost
     * allowed_ips 192.168.1.1,192.168.1.2,192.168.1.3
     * max_connections 100
     * timeout 30.5
     * @endcode
     */
    bool load(const std::string& filename);

    /**
     * @brief Gets raw variant value for a key
     * @param key The configuration key to look up
     * @return The variant value
     * @throws ConfigError if key not found
     */
    Value get(const std::string& key) const;

    /**
     * @brief Checks if a key exists in the configuration
     * @param key The key to check
     * @return true if key exists
     */
    bool hasKey(const std::string& key) const;

    /**
     * @brief Adds a custom validation rule for a key
     * @param key The key to validate
     * @param callback Function that returns true if value is valid
     * @param errorMessage Message to show if validation fails
     */
    void addValidationRule(const std::string& key, ValidationCallback callback, const std::string& errorMessage);
    
    /**
     * @brief Adds a range validation rule for numeric types
     * @param key The key to validate
     * @param min Minimum allowed value
     * @param max Maximum allowed value
     *
     * Example:
     * @code
     * // Validate port number range
     * config.addRangeRule<int>("port", 1024, 65535);
     * 
     * // Validate temperature range
     * config.addRangeRule<float>("temperature", -50.0, 100.0);
     * @endcode
     */
    template<typename T>
    void addRangeRule(const std::string& key, T min, T max) {
        addValidationRule(key,
            [min, max](const Value& v) {
                try {
                    const T& val = std::get<T>(v);
                    return val >= min && val <= max;
                } catch (const std::bad_variant_access&) {
                    return false;
                }
            },
            "Value must be between " + std::to_string(min) + " and " + std::to_string(max)
        );
    }

    /**
     * @brief Adds a regex pattern validation rule for string values
     * @param key The key to validate
     * @param pattern Regular expression pattern to match against
     *
     * Example:
     * @code
     * // Validate email format
     * config.addPatternRule("email", "[\w.-]+@[\w.-]+\.[a-zA-Z]{2,}");
     * 
     * // Validate IPv4 address format
     * config.addPatternRule("ip", "\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}");
     * @endcode
     */
    void addPatternRule(const std::string& key, const std::string& pattern);

private:
    std::unordered_map<std::string, Value> configData;
    std::unordered_map<std::string, ValidationRule> validationRules;

    /**
     * @brief Parses a string into appropriate variant type
     * @param value String value to parse
     * @return Variant containing parsed value
     * @throws ConfigError on parsing errors
     */
    static Value parseValue(const std::string& value);

    /**
     * @brief Validates a value against its rules if any exist
     * @param key The key being validated
     * @param value The value to validate
     * @return true if valid or no rules exist
     * @throws ConfigError if validation fails
     */
    bool validateValue(const std::string& key, const Value& value) const;
};

#endif // CONFIG_LOADER_HPP