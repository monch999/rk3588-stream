#pragma once
// 极简 JSON 解析: 仅支持扁平 {"key": value} 结构
// value 可以是 number / string / bool, 不支持嵌套对象/数组
// 适用于本项目所有配置文件(字段固定且扁平)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

class JsonFlat {
public:
  // 加载并解析文件; 失败返回 false
  bool LoadFile(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
      fprintf(stderr, "[JSON ] open failed: %s\n", path.c_str());
      return false;
    }
    std::stringstream ss;
    ss << ifs.rdbuf();
    return Parse(ss.str());
  }

  bool Parse(const std::string& text) {
    kv_.clear();
    const char* p = text.c_str();
    const char* end = p + text.size();
    SkipWs(p, end);
    if (p >= end || *p != '{') return false;
    p++;
    while (p < end) {
      SkipWs(p, end);
      if (p < end && *p == '}') return true;
      // key
      std::string key;
      if (!ParseString(p, end, key)) return false;
      SkipWs(p, end);
      if (p >= end || *p != ':') return false;
      p++;
      SkipWs(p, end);
      // value (作为字符串保存, 取值时再转换)
      std::string val;
      if (!ParseValue(p, end, val)) return false;
      kv_[key] = val;
      SkipWs(p, end);
      if (p < end && *p == ',') { p++; continue; }
      if (p < end && *p == '}') return true;
      return false;
    }
    return false;
  }

  bool Has(const std::string& key) const { return kv_.count(key) > 0; }

  // 带默认值的取值; 找不到/转换失败返回默认值
  double GetDouble(const std::string& key, double def = 0.0) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) return def;
    return std::strtod(it->second.c_str(), nullptr);
  }

  int GetInt(const std::string& key, int def = 0) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) return def;
    return static_cast<int>(std::strtol(it->second.c_str(), nullptr, 10));
  }

  std::string GetString(const std::string& key, const std::string& def = "") const {
    auto it = kv_.find(key);
    return (it == kv_.end()) ? def : it->second;
  }

  bool GetBool(const std::string& key, bool def = false) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) return def;
    return it->second == "true" || it->second == "1";
  }

private:
  std::unordered_map<std::string, std::string> kv_;

  static void SkipWs(const char*& p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    // 行注释 #...  或  // ...  (非标准 JSON, 但本项目允许, 方便人类编辑)
    while (p < end && (*p == '#' || (p + 1 < end && *p == '/' && *(p + 1) == '/'))) {
      while (p < end && *p != '\n') p++;
      while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    }
  }

  static bool ParseString(const char*& p, const char* end, std::string& out) {
    if (p >= end || *p != '"') return false;
    p++;
    out.clear();
    while (p < end && *p != '"') {
      if (*p == '\\' && p + 1 < end) { out += *(p + 1); p += 2; }
      else { out += *p; p++; }
    }
    if (p >= end) return false;
    p++;
    return true;
  }

  static bool ParseValue(const char*& p, const char* end, std::string& out) {
    if (p >= end) return false;
    if (*p == '"') return ParseString(p, end, out);
    // number / true / false / null
    out.clear();
    while (p < end && *p != ',' && *p != '}' && *p != ' ' && *p != '\n'
           && *p != '\r' && *p != '\t') {
      out += *p;
      p++;
    }
    return !out.empty();
  }
};
