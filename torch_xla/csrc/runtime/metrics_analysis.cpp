#include "torch_xla/csrc/runtime/metrics_analysis.h"

#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/types/variant.h"
#include "torch_xla/csrc/runtime/metrics.h"
#include "torch_xla/csrc/runtime/tf_logging.h"
#include "torch_xla/csrc/runtime/types.h"

namespace torch_xla {
namespace runtime {
namespace metrics {

namespace {

static const char* kAnalysisPrefix = "pt-xla-profiler";

float DivToFloat(std::ldiv_t val, std::int64_t denominator) {
  return val.quot + (float)val.rem / denominator;
}

class MetricFrequency : public Analyzer {
 public:
  MetricFrequency(std::string metric_name, float frequency_threshold,
                  long warmup_steps = 0)
      : metric_name_(metric_name),
        frequency_threshold_(frequency_threshold),
        warmup_steps_(warmup_steps) {}

  Analysis Run() override {
    CounterData* step = GetCounter("MarkStep");
    MetricData* metric = GetMetric(metric_name_);
    if (!step || !metric) {
      return {Analysis::Symptom::kNormal};
    }
    size_t metric_count = metric->TotalSamples();
    int64_t step_count = step->Value();
    if (step_count <= warmup_steps_) {
      return {Analysis::Symptom::kNormal};
    }
    auto res = std::div(metric_count, step_count);
    if (DivToFloat(res, step_count) > frequency_threshold_) {
      return {
          Analysis::Symptom::kMetricTooFrequent,
          absl::StrFormat("%s: %s too frequent: %zu counts during %zu steps",
                          kAnalysisPrefix, metric_name_, metric_count,
                          step_count),
      };
    }
    return {Analysis::Symptom::kNormal};
  }

 private:
  std::string metric_name_;
  float frequency_threshold_;
  long warmup_steps_;
};

class MetricTime : public Analyzer {
 public:
  MetricTime(std::string metric_name, long threshdold_nsec)
      : metric_name_(metric_name), threshold_nsec_(threshdold_nsec) {}

  Analysis Run() override {
    double max_metric_time = 0;
    MetricData* metric = GetMetric(metric_name_);
    if (!metric) {
      return {Analysis::Symptom::kNormal};
    }
    // No need for accumulator and we want all recent samples.
    for (const Sample& sample : metric->Samples(nullptr, nullptr)) {
      max_metric_time = std::max(sample.value, max_metric_time);
    }
    if (max_metric_time > threshold_nsec_) {
      return {
          Analysis::Symptom::kMetricTooSlow,
          absl::StrFormat(
              "%s: %s too slow: longest instance took %s. "
              "Please open a GitHub issue with the graph dump for "
              "our team to optimize.",
              kAnalysisPrefix, metric_name_,
              torch_xla::runtime::metrics::MetricFnTime(max_metric_time)),
      };
    }
    return {Analysis::Symptom::kNormal};
  }

 private:
  std::string metric_name_;
  long threshold_nsec_;
};

class XrtMetricFrequency : public Analyzer {
 public:
  XrtMetricFrequency(std::map<std::string, float> metric_name_thresholds,
                     int run_every_n)
      : metric_name_thresholds_(std::move(metric_name_thresholds)),
        run_every_n_(run_every_n),
        counter_(0) {}

  Analysis Run() override {
    LOG(FATAL) << "For XrtMetricFrequency, use the metrics overload";
  }

  Analysis Run(const std::map<std::string, torch_xla::runtime::Metric>&
                   xrt_metrics) override {
    // XRT GetMetrics call is relatively expensive.
    if (counter_++ != run_every_n_) {
      return {Analysis::Symptom::kNormal};
    }
    counter_ = 0;
    CounterData* step = GetCounter("MarkStep");
    if (!step) {
      return {Analysis::Symptom::kNormal};
    }

    std::stringstream ss;
    int64_t step_count = step->Value();
    for (const auto& kv : metric_name_thresholds_) {
      auto it = xrt_metrics.find(kv.first);
      if (it == xrt_metrics.end()) {
        continue;
      }
      torch_xla::runtime::Metric xrt_metric = it->second;
      std::ldiv_t res;
      if (xrt_metric.int64_value) {
        int64_t metric_count = *xrt_metric.int64_value;
        res = std::div(metric_count, step_count);
      } else if (xrt_metric.percentile) {
        size_t metric_count = (*xrt_metric.percentile).total_samples;
        res = std::div(metric_count, step_count);
      } else {
        continue;
      }
      if (DivToFloat(res, step_count) > kv.second) {
        ss << kv.first << " (" << DivToFloat(res, step_count) * step_count
           << " counts), ";
      }
    }

    std::string repr = ss.str();
    if (!repr.empty()) {
      return {
          Analysis::Symptom::kMetricTooFrequent,
          absl::StrFormat(
              "%s: Following metrics too frequent: %sduring %zu steps. "
              "Note: XRT metrics follow the lifecycle of the TPU "
              "so you may need "
              "to restart the TPU for fresh metrics.",
              kAnalysisPrefix, repr, step_count),
      };
    }
    return {Analysis::Symptom::kNormal};
  }

 private:
  std::map<std::string, float> metric_name_thresholds_;
  bool is_xrt_metric_;
  int run_every_n_;
  int counter_;
};

class UnloweredOp : public Analyzer {
 public:
  Analysis Run() override {
    std::stringstream ss;
    MetricsArena* arena = MetricsArena::Get();
    arena->ForEachCounter([&ss](const std::string& name, CounterData* data) {
      if (absl::StrContains(name, "aten::") &&
          name != "aten::_local_scalar_dense") {
        ss << name << ", ";
      }
    });

    std::string repr = ss.str();
    if (!repr.empty()) {
      return {Analysis::Symptom::kUnloweredOp,
              absl::StrCat(kAnalysisPrefix, ": Op(s) not lowered: ", repr,
                           " Please open a GitHub issue with the above op "
                           "lowering requests.")};
    }
    return {Analysis::Symptom::kNormal, repr};
  }
};

std::vector<Analyzer*>* GetAnalyzers() {
  static std::vector<Analyzer*>* analyzers = new std::vector<Analyzer*>{
      new MetricFrequency("CompileTime", 0.5f, 10),
      new MetricFrequency("TransferFromDeviceTime", 0.5f),
      new MetricTime("CompileTime", 300e9),
      new MetricTime("ExecuteTime", 30e9),
      new UnloweredOp(),
      new XrtMetricFrequency({{"XrtTryFreeMemory", 0.1f},
                              {"XrtCompaction", 0.1f},
                              {"XrtExecutorEvict", 0.1f}},
                             10),
  };
  return analyzers;
}

}  // namespace

std::string CreatePerformanceReport(
    const std::map<std::string, torch_xla::runtime::Metric>& xrt_metrics) {
  std::stringstream ss;
  std::vector<Analyzer*>* analyzers = GetAnalyzers();
  for (auto const& analyzer : *analyzers) {
    Analysis result = analyzer->Run(xrt_metrics);
    if (result.symptom != Analysis::Symptom::kNormal) {
      ss << result.repr << std::endl;
    }
  }
  return ss.str();
}

}  // namespace metrics
}  // namespace runtime
}  // namespace torch_xla
