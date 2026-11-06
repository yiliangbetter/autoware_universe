// Copyright 2026 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The ImplicitGemm plugin's ONNX-facing contract, checked without an engine build: which input
// counts, types and attributes the creator and the plugin accept, and what they refuse (naming the
// reason). The INT8 form is the one autoware-ml exports and t4sparse reads (RDDR-1406):
//   inputs 0 INT8 features, 1 INT8 filters, 2-4 INT32 rulebook, 5 FP32 weight_scale [C_out],
//   6 FP32 bias [C_out] or [0]; attributes input_scale > 0 and output_type "f16"; FP16 output.
// Needs a CUDA device: the plugin reads the compute capability when it is created.

#include "autoware/tensorrt_plugins/implicit_gemm_plugin.hpp"
#include "autoware/tensorrt_plugins/implicit_gemm_plugin_creator.hpp"

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace
{

class RecordingLogger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, nvinfer1::AsciiChar const * msg) noexcept override
  {
    if (severity <= Severity::kERROR) {
      errors.emplace_back(msg);
    }
  }
  std::vector<std::string> errors;
};

/// The plugin logs through TensorRT's global getLogger(), which is the logger of the runtime (or
/// builder) created last: one runtime for the whole test binary, kept alive.
struct Logging
{
  RecordingLogger logger;
  std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
};

Logging & logging()
{
  static Logging instance;
  return instance;
}

std::string last_error()
{
  auto const & errors = logging().logger.errors;
  return errors.empty() ? std::string() : errors.back();
}

void clear_errors()
{
  logging().logger.errors.clear();
}

/// The attributes of one ONNX node, as the parser hands them to the creator.
class NodeAttributes
{
public:
  NodeAttributes & int32(char const * name, std::int32_t value)
  {
    ints_.push_back(std::make_unique<std::int32_t>(value));
    fields_.emplace_back(name, ints_.back().get(), nvinfer1::PluginFieldType::kINT32, 1);
    return *this;
  }
  NodeAttributes & float32(char const * name, float value)
  {
    floats_.push_back(std::make_unique<float>(value));
    fields_.emplace_back(name, floats_.back().get(), nvinfer1::PluginFieldType::kFLOAT32, 1);
    return *this;
  }
  NodeAttributes & string(char const * name, std::string const & value)
  {
    strings_.push_back(std::make_unique<std::string>(value));
    // The ONNX parser counts the terminating NUL in the length.
    fields_.emplace_back(
      name, strings_.back()->c_str(), nvinfer1::PluginFieldType::kCHAR,
      static_cast<std::int32_t>(value.size() + 1));
    return *this;
  }
  nvinfer1::PluginFieldCollection const * collection()
  {
    fc_.nbFields = static_cast<std::int32_t>(fields_.size());
    fc_.fields = fields_.data();
    return &fc_;
  }

private:
  std::vector<std::unique_ptr<std::int32_t>> ints_;
  std::vector<std::unique_ptr<float>> floats_;
  std::vector<std::unique_ptr<std::string>> strings_;
  std::vector<nvinfer1::PluginField> fields_;
  nvinfer1::PluginFieldCollection fc_{};
};

/// The attributes every exported node carries.
NodeAttributes base_attributes()
{
  NodeAttributes attrs;
  attrs.float32("act_alpha", 0.0F)
    .float32("act_beta", 0.0F)
    .int32("is_subm", 1)
    .int32("is_train", 0)
    .float32("output_add_scale", 0.0F)
    .float32("output_scale", 1.0F)
    .int32("act_type", 0);
  return attrs;
}

NodeAttributes int8_attributes(float input_scale = 0.05F, std::string const & output_type = "f16")
{
  NodeAttributes attrs = base_attributes();
  attrs.float32("input_scale", input_scale).string("output_type", output_type);
  return attrs;
}

struct Plugin
{
  explicit Plugin(nvinfer1::IPluginV3 * p) : raw(p)
  {
    if (p != nullptr) {
      build = static_cast<nvinfer1::IPluginV3OneBuild *>(
        p->getCapabilityInterface(nvinfer1::PluginCapabilityType::kBUILD));
    }
  }
  // IPluginV3 has no destroy(): the plugin is owned by whoever created it.
  ~Plugin() { delete raw; }
  Plugin(Plugin const &) = delete;
  Plugin & operator=(Plugin const &) = delete;
  nvinfer1::IPluginV3 * raw{nullptr};
  nvinfer1::IPluginV3OneBuild * build{nullptr};
};

Plugin create(NodeAttributes & attrs, char const * name = "conv/ImplicitGemm")
{
  nvinfer1::plugin::ImplicitGemmPluginCreator creator;
  return Plugin(creator.createPlugin(name, attrs.collection(), nvinfer1::TensorRTPhase::kBUILD));
}

nvinfer1::Dims dims(std::vector<std::int64_t> const & d)
{
  nvinfer1::Dims out{};
  out.nbDims = static_cast<std::int32_t>(d.size());
  for (std::size_t i = 0; i < d.size(); ++i) {
    out.d[i] = d[i];
  }
  return out;
}

/// The [inputs..., output] descriptors of one node: N voxels, C_in -> C_out, 3x3x3, KV = 27.
std::vector<nvinfer1::DynamicPluginTensorDesc> descriptors(
  std::vector<nvinfer1::DataType> const & types, std::int64_t c_in, std::int64_t c_out,
  std::int64_t bias_length = 0)
{
  constexpr std::int64_t n = 1000;
  constexpr std::int64_t kv = 27;
  std::vector<nvinfer1::Dims> shapes{
    dims({n, c_in}), dims({c_out, 3, 3, 3, c_in}), dims({kv, n}), dims({n, 1}), dims({n})};
  std::size_t const num_inputs = types.size() - 1;
  if (num_inputs == 6) {  // bias in the features' type
    shapes.push_back(dims({c_out}));
  } else if (num_inputs == 7) {  // weight_scale, bias
    shapes.push_back(dims({c_out}));
    shapes.push_back(dims({bias_length}));
  }
  shapes.push_back(dims({n, c_out}));
  EXPECT_EQ(shapes.size(), types.size());
  std::vector<nvinfer1::DynamicPluginTensorDesc> out(types.size());
  for (std::size_t i = 0; i < types.size(); ++i) {
    out[i].desc.dims = shapes[i];
    out[i].desc.type = types[i];
    out[i].desc.format = nvinfer1::TensorFormat::kLINEAR;
    out[i].min = shapes[i];
    out[i].opt = shapes[i];
    out[i].max = shapes[i];
  }
  return out;
}

bool all_formats_supported(Plugin & plugin, std::vector<nvinfer1::DynamicPluginTensorDesc> & in_out)
{
  auto const num_inputs = static_cast<std::int32_t>(in_out.size()) - 1;
  for (std::int32_t pos = 0; pos <= num_inputs; ++pos) {
    if (!plugin.build->supportsFormatCombination(pos, in_out.data(), num_inputs, 1)) {
      return false;
    }
  }
  return true;
}

using nvinfer1::DataType;
std::vector<DataType> const kInt8Types{DataType::kINT8,  DataType::kINT8,  DataType::kINT32,
                                       DataType::kINT32, DataType::kINT32, DataType::kFLOAT,
                                       DataType::kFLOAT, DataType::kHALF};

}  // namespace

TEST(ImplicitGemmContract, Int8SharedFormIsAcceptedAndWritesFp16)
{
  clear_errors();
  NodeAttributes attrs = int8_attributes();
  Plugin plugin = create(attrs);
  ASSERT_NE(plugin.raw, nullptr) << last_error();

  std::vector<DataType> types(kInt8Types.begin(), kInt8Types.end() - 1);
  DataType output{};
  ASSERT_EQ(plugin.build->getOutputDataTypes(&output, 1, types.data(), 7), 0) << last_error();
  EXPECT_EQ(output, DataType::kHALF);

  for (std::int64_t bias_length : {0, 64}) {
    auto in_out = descriptors(kInt8Types, 64, 64, bias_length);
    EXPECT_TRUE(all_formats_supported(plugin, in_out)) << "bias [" << bias_length << "]";
    EXPECT_EQ(plugin.build->configurePlugin(in_out.data(), 7, &in_out.back(), 1), 0)
      << last_error();
  }
  EXPECT_TRUE(logging().logger.errors.empty()) << last_error();
}

TEST(ImplicitGemmContract, Int8RefusesChannelsNotMultipleOf32)
{
  for (auto const & [c_in, c_out] :
       std::vector<std::pair<std::int64_t, std::int64_t>>{{48, 64}, {64, 48}, {16, 16}}) {
    clear_errors();
    NodeAttributes attrs = int8_attributes();
    Plugin plugin = create(attrs);
    ASSERT_NE(plugin.raw, nullptr);
    auto in_out = descriptors(kInt8Types, c_in, c_out);
    EXPECT_FALSE(plugin.build->supportsFormatCombination(0, in_out.data(), 7, 1));
    EXPECT_NE(last_error().find("multiples of 32"), std::string::npos) << last_error();
    EXPECT_NE(last_error().find("conv/ImplicitGemm"), std::string::npos) << last_error();
    EXPECT_NE(plugin.build->configurePlugin(in_out.data(), 7, &in_out.back(), 1), 0);
  }
  // Multiples of 32 that are not powers of two are fine.
  clear_errors();
  NodeAttributes attrs = int8_attributes();
  Plugin plugin = create(attrs);
  auto in_out = descriptors(kInt8Types, 96, 160);
  EXPECT_TRUE(all_formats_supported(plugin, in_out)) << last_error();
}

TEST(ImplicitGemmContract, Int8NeedsSevenInputsAndInt8Filters)
{
  NodeAttributes attrs = int8_attributes();
  Plugin plugin = create(attrs);
  ASSERT_NE(plugin.raw, nullptr);
  DataType output{};

  clear_errors();
  std::vector<DataType> five{
    DataType::kINT8, DataType::kINT8, DataType::kINT32, DataType::kINT32, DataType::kINT32};
  EXPECT_NE(plugin.build->getOutputDataTypes(&output, 1, five.data(), 5), 0);
  EXPECT_NE(last_error().find("7 inputs"), std::string::npos) << last_error();

  clear_errors();
  std::vector<DataType> half_filters{DataType::kINT8,  DataType::kHALF,  DataType::kINT32,
                                     DataType::kINT32, DataType::kINT32, DataType::kFLOAT,
                                     DataType::kFLOAT};
  EXPECT_NE(plugin.build->getOutputDataTypes(&output, 1, half_filters.data(), 7), 0);
  EXPECT_NE(last_error().find("INT8 filters"), std::string::npos) << last_error();

  // Offered FP16 features (a weakly typed build handing over the QuantizeLinear's input), a
  // quantized node declines.
  auto in_out = descriptors(kInt8Types, 64, 64);
  in_out[0].desc.type = DataType::kHALF;
  EXPECT_FALSE(plugin.build->supportsFormatCombination(0, in_out.data(), 7, 1));
}

TEST(ImplicitGemmContract, Int8NeedsInputScaleAndOutputType)
{
  DataType output{};
  std::vector<DataType> types(kInt8Types.begin(), kInt8Types.end() - 1);

  clear_errors();
  NodeAttributes no_scale = base_attributes();
  no_scale.string("output_type", "f16");
  Plugin plugin = create(no_scale);
  ASSERT_NE(plugin.raw, nullptr);
  EXPECT_NE(plugin.build->getOutputDataTypes(&output, 1, types.data(), 7), 0);
  EXPECT_NE(last_error().find("input_scale"), std::string::npos) << last_error();

  clear_errors();
  NodeAttributes no_type = base_attributes();
  no_type.float32("input_scale", 0.05F);
  Plugin plugin2 = create(no_type);
  ASSERT_NE(plugin2.raw, nullptr);
  EXPECT_NE(plugin2.build->getOutputDataTypes(&output, 1, types.data(), 7), 0);
  EXPECT_NE(last_error().find("output_type"), std::string::npos) << last_error();
}

TEST(ImplicitGemmContract, CreatorRefusesOtherOutputTypesAndBadScales)
{
  for (std::string const & output_type : std::vector<std::string>{"bf16", "s8", "e4m3", "fp16"}) {
    clear_errors();
    NodeAttributes attrs = int8_attributes(0.05F, output_type);
    Plugin plugin = create(attrs);
    EXPECT_EQ(plugin.raw, nullptr) << output_type;
    EXPECT_NE(last_error().find("output_type"), std::string::npos) << last_error();
  }
  for (float input_scale : {0.0F, -1.0F}) {
    clear_errors();
    NodeAttributes attrs = int8_attributes(input_scale);
    Plugin plugin = create(attrs);
    EXPECT_EQ(plugin.raw, nullptr) << input_scale;
    EXPECT_NE(last_error().find("input_scale"), std::string::npos) << last_error();
  }
  clear_errors();
  NodeAttributes attrs = base_attributes();
  attrs.float32("output_scale", 2.0F);
  Plugin plugin = create(attrs);
  EXPECT_EQ(plugin.raw, nullptr);
  EXPECT_NE(last_error().find("output_scale"), std::string::npos) << last_error();
}

TEST(ImplicitGemmContract, FloatingPointFormsAreUnchanged)
{
  clear_errors();
  NodeAttributes attrs = base_attributes();
  Plugin plugin = create(attrs);
  ASSERT_NE(plugin.raw, nullptr) << last_error();
  DataType output{};

  // Five inputs, FP16: FP16 out; FP32: FP32 out. Any channel width.
  for (DataType t : {DataType::kHALF, DataType::kFLOAT}) {
    std::vector<DataType> types{t, t, DataType::kINT32, DataType::kINT32, DataType::kINT32};
    ASSERT_EQ(plugin.build->getOutputDataTypes(&output, 1, types.data(), 5), 0) << last_error();
    EXPECT_EQ(output, t);
    types.push_back(t);
    auto in_out = descriptors(types, 16, 48);
    EXPECT_TRUE(all_formats_supported(plugin, in_out));
  }
  // Six inputs: the bias in the features' type.
  {
    std::vector<DataType> types{DataType::kHALF,  DataType::kHALF,  DataType::kINT32,
                                DataType::kINT32, DataType::kINT32, DataType::kHALF,
                                DataType::kHALF};
    auto in_out = descriptors(types, 32, 32);
    EXPECT_TRUE(all_formats_supported(plugin, in_out));
    in_out[5].desc.type = DataType::kFLOAT;
    EXPECT_FALSE(plugin.build->supportsFormatCombination(5, in_out.data(), 6, 1));
  }
  // An FP16 node must not carry input_scale: that is the quantized form.
  clear_errors();
  NodeAttributes quantized = int8_attributes();
  Plugin plugin2 = create(quantized);
  std::vector<DataType> types{
    DataType::kHALF, DataType::kHALF, DataType::kINT32, DataType::kINT32, DataType::kINT32};
  EXPECT_NE(plugin2.build->getOutputDataTypes(&output, 1, types.data(), 5), 0);
  EXPECT_NE(last_error().find("input_scale"), std::string::npos) << last_error();
  // FP8 is refused by name.
  clear_errors();
  std::vector<DataType> fp8{
    DataType::kFP8, DataType::kFP8, DataType::kINT32, DataType::kINT32, DataType::kINT32};
  EXPECT_NE(plugin.build->getOutputDataTypes(&output, 1, fp8.data(), 5), 0);
  EXPECT_NE(last_error().find("FP8"), std::string::npos) << last_error();
}
