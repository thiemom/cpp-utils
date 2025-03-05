/*
 * Copyright (c) 2025 Thiemo M.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "config.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <regex>

/**
 * @brief Loads and parses a configuration file
 * @param filename Path to the configuration file
 * @return true if loading successful
 * @throws ConfigError on file access or parsing errors
 * Format: key value
 * Comments start with #, empty lines are ignored
 * Values can be: numbers, strings, or comma-separated arrays
 */
bool ConfigLoader::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        throw ConfigError("Could not open config file: " + filename);
    }

    std::string line;
    size_t lineNum = 0;
    while (std::getline(file, line)) {
        lineNum++;
        if (line.empty() || line[0] == '#') continue; // Skip empty lines and comments

        std::istringstream iss(line);
        std::string key, value;
        if (!(iss >> key)) {
            throw ConfigError("Invalid line format", lineNum);
        }

        std::getline(iss >> std::ws, value); // Read the rest of the line after the key
        if (value.empty()) {
            throw ConfigError("Missing value for key: " + key, lineNum);
        }

        auto parsedValue = parseValue(value);
        if (!validateValue(key, parsedValue)) {
            // validateValue will throw with specific error
            throw ConfigError("Validation failed for key: " + key, lineNum);
        }
        
        configData[key] = std::move(parsedValue);
    }
    return true;
}

/**
 * @brief Retrieves a value by key from the configuration
 * @param key The key to look up
 * @return The variant value associated with the key
 * @throws ConfigError if key not found
 */
ConfigLoader::Value ConfigLoader::get(const std::string& key) const {
    auto it = configData.find(key);
    if (it != configData.end()) {
        return it->second;
    }
    throw ConfigError("Key not found: " + key);
}

/**
 * @brief Checks if a key exists in the configuration
 * @param key The key to check
 * @return true if the key exists
 */
bool ConfigLoader::hasKey(const std::string& key) const {
    return configData.find(key) != configData.end();
}

/**
 * @brief Adds a validation rule for a configuration key
 * @param key The key to validate
 * @param callback Function that returns true if value is valid
 * @param errorMessage Message to show if validation fails
 */
void ConfigLoader::addValidationRule(const std::string& key, ValidationCallback callback, const std::string& errorMessage) {
    validationRules[key] = ValidationRule{callback, errorMessage};
}

/**
 * @brief Adds a regex pattern validation rule for string values
 * @param key The key to validate
 * @param pattern Regular expression pattern to match against
 */
void ConfigLoader::addPatternRule(const std::string& key, const std::string& pattern) {
    addValidationRule(key,
        [pattern](const Value& v) {
            try {
                const auto& str = std::get<std::string>(v);
                return std::regex_match(str, std::regex(pattern));
            } catch (const std::bad_variant_access&) {
                return false;
            } catch (const std::regex_error&) {
                return false;
            }
        },
        "Value must match pattern: " + pattern
    );
}

/**
 * @brief Validates a value against its rules if any exist
 * @param key The key being validated
 * @param value The value to validate
 * @return true if valid or no rules exist
 * @throws ConfigError if validation fails
 */
bool ConfigLoader::validateValue(const std::string& key, const Value& value) const {
    auto it = validationRules.find(key);
    if (it != validationRules.end()) {
        if (!it->second.callback(value)) {
            throw ConfigError(it->second.errorMessage);
        }
    }
    return true;
}

/**
 * @brief Parses a string into the appropriate variant type
 * @param value String value to parse
 * @return Variant containing the parsed value
 * @throws ConfigError on parsing errors
 * 
 * Handles:
 * - Integer values (with optional sign)
 * - Float values (with optional sign and decimal)
 * - String values (default)
 * - Arrays (comma-separated values)
 *
 * Example inputs:
 * @code
 * // Integers
 * "42"      -> int(42)
 * "-17"     -> int(-17)
 * 
 * // Floats
 * "3.14"    -> float(3.14)
 * "-0.5"    -> float(-0.5)
 * 
 * // Strings
 * "hello"   -> string("hello")
 * "127.0.0.1" -> string("127.0.0.1")
 * 
 * // Arrays
 * "1,2,3"   -> vector<int>{1,2,3}
 * "1.0,2.0" -> vector<float>{1.0,2.0}
 * "a,b,c"   -> vector<string>{"a","b","c"}
 * @endcode
 */
ConfigLoader::Value ConfigLoader::parseValue(const std::string& value) {
    if (value.find(',') != std::string::npos) { // Handle vector types
        std::istringstream ss(value);
        std::vector<std::string> tokens;
        std::string token;
        
        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }

        // Special handling for IP addresses
        bool looksLikeIPs = false;
        for (const auto& token : tokens) {
            // Count dots in the token
            int dotCount = 0;
            for (char c : token) {
                if (c == '.') dotCount++;
            }
            
            // IP addresses typically have 3 dots (e.g., 192.168.1.1)
            if (dotCount == 3 && token.find_first_not_of("0123456789. ") == std::string::npos) {
                looksLikeIPs = true;
            }
        }
        
        // If it looks like IP addresses, treat as string vector
        if (looksLikeIPs) {
            std::vector<std::string> stringVec;
            for (const auto& t : tokens) {
                stringVec.push_back(t);
            }
            return stringVec;
        }
        
        // Check if tokens contain non-numeric characters
        bool containsNonNumeric = false;
        for (const auto& token : tokens) {
            if (token.find_first_not_of("0123456789.-+ ") != std::string::npos) {
                containsNonNumeric = true;
                break;
            }
        }
        
        // If tokens contain non-numeric chars, treat as string vector
        if (containsNonNumeric) {
            std::vector<std::string> stringVec;
            for (const auto& t : tokens) {
                stringVec.push_back(t);
            }
            return stringVec;
        }
        
        // First try to parse as vector of integers
        try {
            if (std::all_of(tokens.begin(), tokens.end(), [](const std::string& s) {
                return s.find_first_not_of(" 0123456789-") == std::string::npos;
            })) {
                std::vector<int> vec;
                for (const auto& t : tokens) vec.push_back(std::stoi(t));
                return vec;
            }
        } catch (const std::invalid_argument&) {
            // Not integers, continue to next type
        } catch (const std::out_of_range&) {
            throw ConfigError("Number out of range in array: " + value);
        }
        
        // Then try to parse as vector of floats
        try {
            if (std::all_of(tokens.begin(), tokens.end(), [](const std::string& s) {
                return s.find_first_not_of(" 0123456789.-") == std::string::npos;
            })) {
                std::vector<float> vec;
                for (const auto& t : tokens) vec.push_back(std::stof(t));
                return vec;
            }
        } catch (const std::invalid_argument&) {
            // Not floats, continue to next type
        } catch (const std::out_of_range&) {
            throw ConfigError("Number out of range in array: " + value);
        }
        
        // If not numbers, return as vector of strings
        std::vector<std::string> stringVec;
        for (const auto& t : tokens) {
            stringVec.push_back(t);
        }
        return stringVec;
    }

    // Handle single values
    try {
        if (value.find_first_not_of(" 0123456789-") == std::string::npos)
            return std::stoi(value);
        if (value.find_first_not_of(" 0123456789.-") == std::string::npos)
            return std::stof(value);
    } catch (const std::invalid_argument&) {
        throw ConfigError("Invalid number format: " + value);
    } catch (const std::out_of_range&) {
        throw ConfigError("Number out of range: " + value);
    }
    return value; // Assume string
}