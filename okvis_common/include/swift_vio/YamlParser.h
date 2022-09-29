/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   YamlParser.h
 * @brief  Base class for YAML parsers.
 * @author Antoni Rosinol
 */

#ifndef YAML_PARSER_H
#define YAML_PARSER_H

#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <glog/logging.h>

#include <opencv2/core/core.hpp>

#include "okvis/class_macros.hpp"

namespace swift_vio {
class YamlParser {
 public:
  POINTER_TYPEDEFS(YamlParser);

  YamlParser(const std::string& filepath) : fs_(), filepath_(filepath) {
    openFile(filepath, &fs_);
  }
  ~YamlParser() { closeFile(&fs_); }

  template <class T>
  void getYamlParam(const std::string& id, T* output) const {
    if (id.empty()) {
      return;
    }
    const cv::FileNode& file_handle = fs_[id];
    if (file_handle.type() == cv::FileNode::NONE)
      LOG(WARNING) << "Missing parameter: " << id.c_str()
                   << " in file: " << filepath_.c_str();
    else
      file_handle >> *output;
  }

  template <class T>
  void getNestedYamlParam(const std::string& id, const std::string& id_2,
                          T* output) const {
    if (id.empty() || id_2.empty()) {
      return;
    }
    const cv::FileNode& file_handle = fs_[id];
    if (file_handle.type() == cv::FileNode::NONE) {
      LOG(WARNING) << "Missing parameter: " << id.c_str()
                   << " in file: " << filepath_.c_str();
      return;
    }
    const cv::FileNode& file_handle_2 = file_handle[id_2];
    if (file_handle_2.type() == cv::FileNode::NONE) {
      LOG(WARNING) << "Missing nested parameter: " << id_2.c_str() << " inside "
                   << id.c_str() << '\n'
                   << " in file: " << filepath_.c_str();
      return;
    }
    if (!file_handle.isMap()) {
      LOG(WARNING) << "I think that if the parent node is not a map, we can't "
                      "use >> for the child node";
      return;
    }
    file_handle_2 >> *output;
  }

 private:
  void openFile(const std::string& filepath, cv::FileStorage* fs) const {
    CHECK(!filepath.empty()) << "Empty filepath!";
    try {
      fs->open(filepath, cv::FileStorage::READ);
    } catch (cv::Exception& e) {
      LOG(FATAL) << "Cannot open file: " << filepath << '\n'
                 << "OpenCV error code: " << e.msg;
    }
    LOG_IF(FATAL, !fs->isOpened())
        << "Cannot open file in parseYAML: " << filepath
        << " (remember that the first line should be: %YAML:1.0)";
  }

  inline void closeFile(cv::FileStorage* fs) const { fs->release(); }

 private:
  cv::FileStorage fs_;
  std::string filepath_;
};
}  // namespace swift_vio
#endif