#ifndef TENSORFLOW_COMPILER_XLA_RPC_METRICS_H_
#define TENSORFLOW_COMPILER_XLA_RPC_METRICS_H_

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace metrics {

struct Sample {
  Sample() = default;
  Sample(int64 timestamp_ns, double value)
      : timestamp_ns(timestamp_ns), value(value) {}

  int64 timestamp_ns = 0;
  double value = 0;
};

using MetricReprFn = std::function<string(double)>;

// This class can host two different kind of sampling. A counter based one, in
// which the caller posts "counts" (number of requests, amount of data in bytes,
// etc...), and for which the internal counter always increases.
// It also supports a time-sample posting, where up to max_samples samples are
// held in a circular buffer. The latter can be used to extract percentile
// metrics.
class MetricData {
 public:
  MetricData(MetricReprFn repr_fn, size_t max_samples);

  double Counter() const;

  size_t TotalSamples() const;

  void AddSample(int64 timestamp_ns, double value);

  std::vector<Sample> Samples() const;

  string Repr(double value) const { return repr_fn_(value); }

 private:
  mutable std::mutex lock_;
  MetricReprFn repr_fn_;
  size_t count_ = 0;
  std::vector<Sample> samples_;
  double counter_ = 0.0;
};

// Emits the value in a to_string() conversion.
string MetricFnValue(double value);
// Emits the value in a humanized bytes representation.
string MetricFnBytes(double value);
// Emits the value in a humanized time representation. The value is expressed in
// nanoseconds EPOCH time.
string MetricFnTime(double value);

// The typical use of a Metric is one in which it gets created either in a
// global scope context:
//   static Metric* metric = new Metric("RpcCount");
// Or within a function scope:
//   void MyFunction(...) {
//     static Metric* metric = new Metric("RpcCount");
//     ...
//     metric->AddSample(ts_nanos, some_value);
//   }
class Metric {
 public:
  explicit Metric(string name, MetricReprFn repr_fn = MetricFnValue,
                  size_t max_samples = 1024);

  const string& Name() const { return name_; }

  double Counter() const;

  void AddSample(int64 timestamp_ns, double value);

  void AddSample(double value);

  std::vector<Sample> Samples() const;

  string Repr(double value) const;

 private:
  MetricData* GetData() const;

  string name_;
  MetricReprFn repr_fn_;
  size_t max_samples_;
  mutable std::shared_ptr<MetricData> data_;
};

// Retrieves the current EPOCH time in nanoseconds.
int64 NowNs();

// Creates a report with the current metrics statistics.
string CreateMetricReport();

// Scope based utility class to measure the time the code takes within a given
// C++ scope.
class TimedSection {
 public:
  explicit TimedSection(Metric* metric, Metric* counter_metric = nullptr)
      : metric_(metric), counter_metric_(counter_metric), start_(NowNs()) {}

  ~TimedSection() {
    int64 now = NowNs();
    metric_->AddSample(now, now - start_);
    if (counter_metric_ != nullptr) {
      counter_metric_->AddSample(1, now);
    }
  }

 private:
  Metric* metric_;
  Metric* counter_metric_;
  int64 start_;
};

}  // namespace metrics
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RPC_METRICS_H_
