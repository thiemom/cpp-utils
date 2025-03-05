#include <gtest/gtest.h>
#include "../config.h"
#include <fstream>
#include <cstdio>

class ConfigLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a test config file
        std::ofstream testFile("test_config.txt");
        testFile << "# Test configuration\n";
        testFile << "port 8080\n";
        testFile << "host localhost\n";
        testFile << "allowed_ips 192.168.1.1,192.168.1.2\n";
        testFile << "max_connections 100\n";
        testFile << "timeout 30.5\n";
        testFile.close();
    }

    void TearDown() override {
        // Remove test file
        std::remove("test_config.txt");
    }
};

TEST_F(ConfigLoaderTest, LoadConfig) {
    ConfigLoader config;
    ASSERT_TRUE(config.load("test_config.txt"));
}

TEST_F(ConfigLoaderTest, GetIntValue) {
    ConfigLoader config;
    ASSERT_TRUE(config.load("test_config.txt"));
    EXPECT_EQ(config.get<int>("port"), 8080);
    EXPECT_EQ(config.get<int>("max_connections"), 100);
}

TEST_F(ConfigLoaderTest, GetStringValue) {
    ConfigLoader config;
    ASSERT_TRUE(config.load("test_config.txt"));
    EXPECT_EQ(config.get<std::string>("host"), "localhost");
}

TEST_F(ConfigLoaderTest, GetFloatValue) {
    ConfigLoader config;
    ASSERT_TRUE(config.load("test_config.txt"));
    EXPECT_FLOAT_EQ(config.get<float>("timeout"), 30.5f);
}

TEST_F(ConfigLoaderTest, GetVectorValue) {
    ConfigLoader config;
    ASSERT_TRUE(config.load("test_config.txt"));
    auto ips = config.get<std::vector<std::string>>("allowed_ips");
    ASSERT_EQ(ips.size(), 2);
    EXPECT_EQ(ips[0], "192.168.1.1");
    EXPECT_EQ(ips[1], "192.168.1.2");
}

TEST_F(ConfigLoaderTest, GetDefaultValue) {
    ConfigLoader config;
    ASSERT_TRUE(config.load("test_config.txt"));
    EXPECT_EQ(config.getOr<int>("missing_key", 42), 42);
}

TEST_F(ConfigLoaderTest, ValidationRules) {
    ConfigLoader config;
    config.addRangeRule<int>("port", 1024, 65535);
    ASSERT_TRUE(config.load("test_config.txt"));
    EXPECT_EQ(config.get<int>("port"), 8080);
    
    // Create a file with invalid port
    std::ofstream testFile("invalid_config.txt");
    testFile << "port 80\n"; // Below allowed range
    testFile.close();
    
    EXPECT_THROW(config.load("invalid_config.txt"), ConfigError);
    std::remove("invalid_config.txt");
}
