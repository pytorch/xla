#ifndef XLA_CLIENT_METRICS_H_
#define XLA_CLIENT_METRICS_H_

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"

namespace xla {
namespace metrics {

struct Sample {
  Sample() = default;
  Sample(int64 timestamp_ns, double value)
      : timestamp_ns(timestamp_ns), value(value) {}

  int64 timestamp_ns = 0;
  double value = 0;
};

using MetricReprFn = std::function<std::string(double)>;

// Class used to collect time-stamped numeric samples. The samples are stored in
// a circular buffer whose size can be configured at constructor time.
class MetricData {
 public:
  // Creates a new MetricData object with the internal circular buffer storing
  // max_samples samples. The repr_fn argument allow to specify a function which
  // pretty-prints a sample value.
  MetricData(MetricReprFn repr_fn, size_t max_samples);

  // Returns the total values of all the samples being posted to this metric.
  double Accumulator() const;

  size_t TotalSamples() const;

  void AddSample(int64 timestamp_ns, double value);

  // Returns a vector with all the current samples, from the oldest to the
  // newer. If accumulator is not nullptr, it will receive the current value of
  // the metrics' accumulator (the sum of all posted values). If total_samples
  // is not nullptr, it will receive the count of the posted values.
  std::vector<Sample> Samples(double* accumulator, size_t* total_samples) const;

  std::string Repr(double value) const { return repr_fn_(value); }

 private:
  mutable std::mutex lock_;
  MetricReprFn repr_fn_;
  size_t count_ = 0;
  std::vector<Sample> samples_;
  double accumulator_ = 0.0;
};

// Counters are a very lightweight form of metrics which do not need to track
// sample time.
class CounterData {
 public:
  CounterData() : value_(0) {}

  void AddValue(xla::int64 value) { value_ += value; }

  xla::int64 Value() const { return value_; }

 private:
  std::atomic<xla::int64> value_;
};

// Emits the value in a to_string() conversion.
std::string MetricFnValue(double value);
// Emits the value in a humanized bytes representation.
std::string MetricFnBytes(double value);
// Emits the value in a humanized time representation. The value is expressed in
// nanoseconds EPOCH time.
std::string MetricFnTime(double value);

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
  explicit Metric(std::string name, MetricReprFn repr_fn = MetricFnValue,
                  size_t max_samples = 1024);

  const std::string& Name() const { return name_; }

  double Accumulator() const;

  void AddSample(int64 timestamp_ns, double value);

  void AddSample(double value);

  std::vector<Sample> Samples(double* accumulator, size_t* total_samples) const;

  std::string Repr(double value) const;

 private:
  MetricData* GetData() const;

  std::string name_;
  MetricReprFn repr_fn_;
  size_t max_samples_;
  mutable std::shared_ptr<MetricData> data_ptr_;
  mutable std::atomic<MetricData*> data_;
};

// A Counter is a lightweight form of metric which tracks an integer value which
// can increase or decrease.
// A typical use is as:
//   static Counter* counter = new Counter("MyCounter");
//   ...
//   counter->AddValue(+1);
class Counter {
 public:
  explicit Counter(std::string name);

  void AddValue(xla::int64 value) { GetData()->AddValue(value); }

  xla::int64 Value() const { return GetData()->Value(); }

 private:
  CounterData* GetData() const;

  std::string name_;
  mutable std::shared_ptr<CounterData> data_ptr_;
  mutable std::atomic<CounterData*> data_;
};

#define XLA_COUNTER(name, value)                \
  do {                                          \
    static ::xla::metrics::Counter* __counter = \
        new ::xla::metrics::Counter(name);      \
    __counter->AddValue(value);                 \
  } while (0)

#define XLA_FN_COUNTER(ns) XLA_COUNTER(absl::StrCat(ns, __FUNCTION__), 1)

#define XLA_VALUE_METRIC(name, value)                                    \
  do {                                                                   \
    static ::xla::metrics::Metric* __metric =                            \
        new ::xla::metrics::Metric(name, ::xla::metrics::MetricFnValue); \
    __metric->AddSample(value);                                          \
  } while (0)

// Creates a report with the current metrics statistics.
std::string CreateMetricReport();

// Returns the currently registered metric names. Note that the list can grow
// since metrics are usualy function intialized (they are static function
// variables).
std::vector<std::string> GetMetricNames();

// Retrieves the metric data of a given metric, or nullptr if such metric does
// not exist.
MetricData* GetMetric(const std::string& name);

// Returns the currently registered counter names. Note that the list can grow
// since counters are usualy function intialized (they are static function
// variables).
std::vector<std::string> GetCounterNames();

// Retrieves the counter data of a given counter, or nullptr if such counter
// does not exist.
CounterData* GetCounter(const std::string& name);

// Scope based utility class to measure the time the code takes within a given
// C++ scope.
class TimedSection {
 public:
  explicit TimedSection(Metric* metric)
      : metric_(metric), start_(sys_util::NowNs()) {}

  ~TimedSection() {
    int64 now = sys_util::NowNs();
    metric_->AddSample(now, now - start_);
  }

  double Elapsed() const {
    return 1e-9 * static_cast<double>(sys_util::NowNs() - start_);
  }

 private:
  Metric* metric_;
  int64 start_;
};

#define XLA_TIMED(name)                                           \
  static xla::metrics::Metric* timed_metric =                     \
      new xla::metrics::Metric(name, xla::metrics::MetricFnTime); \
  xla::metrics::TimedSection timed_section(timed_metric)

}  // namespace metrics
}  // namespace xla

#endif  // XLA_CLIENT_METRICS_H_
