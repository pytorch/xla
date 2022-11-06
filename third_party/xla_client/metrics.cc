#include "tensorflow/compiler/xla/xla_client/metrics.h"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "absl/memory/memory.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {
namespace metrics {
namespace {

const std::vector<double>* ReadEnvPercentiles() {
  std::string percentiles = sys_util::GetEnvString(
      "XLA_METRICS_PERCENTILES", "0.01:0.05:0.1:0.2:0.5:0.8:0.9:0.95:0.99");
  std::vector<std::string> percentiles_list = absl::StrSplit(percentiles, ':');
  std::unique_ptr<std::vector<double>> metrics_percentiles =
      absl::make_unique<std::vector<double>>();
  for (auto& pct_str : percentiles_list) {
    double pct = std::stod(pct_str);
    XLA_CHECK(pct > 0.0 && pct < 1.0) << pct;
    metrics_percentiles->push_back(pct);
  }
  std::sort(metrics_percentiles->begin(), metrics_percentiles->end());
  return metrics_percentiles.release();
}

const std::vector<double>& GetPercentiles() {
  static const std::vector<double>* metrics_percentiles = ReadEnvPercentiles();
  return *metrics_percentiles;
}

void EmitMetricInfo(const std::string& name, MetricData* data,
                    std::stringstream* ss) {
  double accumulator = 0.0;
  size_t total_samples = 0;
  std::vector<Sample> samples = data->Samples(&accumulator, &total_samples);
  (*ss) << "Metric: " << name << std::endl;
  (*ss) << "  TotalSamples: " << total_samples << std::endl;
  (*ss) << "  Accumulator: " << data->Repr(accumulator) << std::endl;
  if (!samples.empty()) {
    double total = 0.0;
    for (auto& sample : samples) {
      total += sample.value;
    }
    int64_t delta_time =
        samples.back().timestamp_ns - samples.front().timestamp_ns;
    if (delta_time > 0) {
      double value_sec = 1e6 * (total / (delta_time / 1000.0));
      (*ss) << "  ValueRate: " << data->Repr(value_sec) << " / second"
            << std::endl;
      double count_sec =
          1e6 * (static_cast<double>(samples.size()) / (delta_time / 1000.0));
      (*ss) << "  Rate: " << count_sec << " / second" << std::endl;
    }
  }

  const std::vector<double>& metrics_percentiles = GetPercentiles();
  std::sort(
      samples.begin(), samples.end(),
      [](const Sample& s1, const Sample& s2) { return s1.value < s2.value; });
  (*ss) << "  Percentiles: ";
  for (size_t i = 0; i < metrics_percentiles.size(); ++i) {
    size_t index = metrics_percentiles[i] * samples.size();
    if (i > 0) {
      (*ss) << "; ";
    }
    (*ss) << (metrics_percentiles[i] * 100.0)
          << "%=" << data->Repr(samples[index].value);
  }
  (*ss) << std::endl;
}

void EmitCounterInfo(const std::string& name, CounterData* data,
                     std::stringstream* ss) {
  (*ss) << "Counter: " << name << std::endl;
  (*ss) << "  Value: " << data->Value() << std::endl;
}

}  // namespace

MetricsArena* MetricsArena::Get() {
  static MetricsArena* arena = new MetricsArena();
  return arena;
}

void MetricsArena::RegisterMetric(const std::string& name, MetricReprFn repr_fn,
                                  size_t max_samples,
                                  std::shared_ptr<MetricData>* data) {
  std::lock_guard<std::mutex> lock(lock_);
  if (*data == nullptr) {
    *data = xla::util::MapInsert(&metrics_, name, [&]() {
      return std::make_shared<MetricData>(std::move(repr_fn), max_samples);
    });
  }
}

void MetricsArena::RegisterCounter(const std::string& name,
                                   std::shared_ptr<CounterData>* data) {
  std::lock_guard<std::mutex> lock(lock_);
  if (*data == nullptr) {
    *data = xla::util::MapInsert(
        &counters_, name, []() { return std::make_shared<CounterData>(); });
  }
}

void MetricsArena::ForEachMetric(
    const std::function<void(const std::string&, MetricData*)>& metric_func) {
  std::lock_guard<std::mutex> lock(lock_);
  for (auto& name_data : metrics_) {
    metric_func(name_data.first, name_data.second.get());
  }
}

void MetricsArena::ForEachCounter(
    const std::function<void(const std::string&, CounterData*)>& counter_func) {
  std::lock_guard<std::mutex> lock(lock_);
  for (auto& name_data : counters_) {
    counter_func(name_data.first, name_data.second.get());
  }
}

std::vector<std::string> MetricsArena::GetMetricNames() {
  std::vector<std::string> names;
  std::lock_guard<std::mutex> lock(lock_);
  for (auto& name_data : metrics_) {
    if (name_data.second->TotalSamples() > 0) {
      names.push_back(name_data.first);
    }
  }
  return names;
}

MetricData* MetricsArena::GetMetric(const std::string& name) {
  std::lock_guard<std::mutex> lock(lock_);
  auto it = metrics_.find(name);
  return (it != metrics_.end() && it->second->TotalSamples() > 0)
             ? it->second.get()
             : nullptr;
}

std::vector<std::string> MetricsArena::GetCounterNames() {
  std::vector<std::string> names;
  std::lock_guard<std::mutex> lock(lock_);
  for (auto& name_data : counters_) {
    if (name_data.second->Value() > 0) {
      names.push_back(name_data.first);
    }
  }
  return names;
}

CounterData* MetricsArena::GetCounter(const std::string& name) {
  std::lock_guard<std::mutex> lock(lock_);
  auto it = counters_.find(name);
  return (it != counters_.end() && it->second->Value() > 0) ? it->second.get()
                                                            : nullptr;
}

void MetricsArena::ClearCounters() {
  for (auto& counter : counters_) {
    counter.second->Clear();
  }
}

void MetricsArena::ClearMetrics() {
  for (auto& metrics : metrics_) {
    metrics.second->Clear();
  }
}

MetricData::MetricData(MetricReprFn repr_fn, size_t max_samples)
    : repr_fn_(std::move(repr_fn)), samples_(max_samples) {}

void MetricData::AddSample(int64_t timestamp_ns, double value) {
  std::lock_guard<std::mutex> lock(lock_);
  size_t position = count_ % samples_.size();
  ++count_;
  accumulator_ += value;
  samples_[position] = Sample(timestamp_ns, value);
}

double MetricData::Accumulator() const {
  std::lock_guard<std::mutex> lock(lock_);
  return accumulator_;
}

size_t MetricData::TotalSamples() const {
  std::lock_guard<std::mutex> lock(lock_);
  return count_;
}

void MetricData::Clear() {
  std::lock_guard<std::mutex> lock(lock_);
  count_ = 0;
  accumulator_ = 0.0;
  samples_ = std::vector<Sample>(samples_.size());
}

std::vector<Sample> MetricData::Samples(double* accumulator,
                                        size_t* total_samples) const {
  std::lock_guard<std::mutex> lock(lock_);
  std::vector<Sample> samples;
  if (count_ <= samples_.size()) {
    samples.insert(samples.end(), samples_.begin(), samples_.begin() + count_);
  } else {
    size_t position = count_ % samples_.size();
    samples.insert(samples.end(), samples_.begin() + position, samples_.end());
    samples.insert(samples.end(), samples_.begin(),
                   samples_.begin() + position);
  }
  if (accumulator != nullptr) {
    *accumulator = accumulator_;
  }
  if (total_samples != nullptr) {
    *total_samples = count_;
  }
  return samples;
}

Metric::Metric(std::string name, MetricReprFn repr_fn, size_t max_samples)
    : name_(std::move(name)),
      repr_fn_(std::move(repr_fn)),
      max_samples_(max_samples != 0
                       ? max_samples
                       : sys_util::GetEnvInt("XLA_METRICS_SAMPLES", 1024)),
      data_(nullptr) {}

double Metric::Accumulator() const { return GetData()->Accumulator(); }

void Metric::AddSample(int64_t timestamp_ns, double value) {
  GetData()->AddSample(timestamp_ns, value);
}

void Metric::AddSample(double value) {
  GetData()->AddSample(sys_util::NowNs(), value);
}

std::vector<Sample> Metric::Samples(double* accumulator,
                                    size_t* total_samples) const {
  return GetData()->Samples(accumulator, total_samples);
}

std::string Metric::Repr(double value) const { return GetData()->Repr(value); }

MetricData* Metric::GetData() const {
  MetricData* data = data_.load();
  if (TF_PREDICT_FALSE(data == nullptr)) {
    // The RegisterMetric() API is a synchronization point, and even if multiple
    // threads enters it, the data will be created only once.
    MetricsArena* arena = MetricsArena::Get();
    arena->RegisterMetric(name_, repr_fn_, max_samples_, &data_ptr_);
    // Even if multiple threads will enter this IF statement, they will all
    // fetch the same value, and hence store the same value below.
    data = data_ptr_.get();
    data_.store(data);
  }
  return data;
}

Counter::Counter(std::string name) : name_(std::move(name)), data_(nullptr) {}

CounterData* Counter::GetData() const {
  CounterData* data = data_.load();
  if (TF_PREDICT_FALSE(data == nullptr)) {
    // The RegisterCounter() API is a synchronization point, and even if
    // multiple threads enters it, the data will be created only once.
    MetricsArena* arena = MetricsArena::Get();
    arena->RegisterCounter(name_, &data_ptr_);
    // Even if multiple threads will enter this IF statement, they will all
    // fetch the same value, and hence store the same value below.
    data = data_ptr_.get();
    data_.store(data);
  }
  return data;
}

std::string MetricFnValue(double value) {
  std::stringstream ss;
  ss.precision(2);
  ss << std::fixed << value;
  return ss.str();
}

std::string MetricFnBytes(double value) {
  const int kNumSuffixes = 6;
  static const char* const kSizeSuffixes[kNumSuffixes] = {"B",  "KB", "MB",
                                                          "GB", "TB", "PB"};
  int sfix = 0;
  for (; (sfix + 1) < kNumSuffixes && value >= 1024.0; ++sfix) {
    value /= 1024.0;
  }
  std::stringstream ss;
  ss.precision(2);
  ss << std::fixed << value << kSizeSuffixes[sfix];
  return ss.str();
}

std::string MetricFnTime(double value) {
  static struct TimePart {
    const char* suffix;
    double scaler;
    int width;
    int precision;
    char fill;
  } const time_parts[] = {
      {"d", 86400.0 * 1e9, 2, 0, '0'}, {"h", 3600.0 * 1e9, 2, 0, '0'},
      {"m", 60.0 * 1e9, 2, 0, '0'},    {"s", 1e9, 2, 0, '0'},
      {"ms", 1e6, 3, 0, '0'},          {"us", 1e3, 7, 3, '0'},
  };
  int count = 0;
  std::stringstream ss;
  for (size_t i = 0; i < TF_ARRAYSIZE(time_parts); ++i) {
    const TimePart& part = time_parts[i];
    double ctime = value / part.scaler;
    if (ctime >= 1.0 || count > 0 || i + 1 == TF_ARRAYSIZE(time_parts)) {
      ss.precision(part.precision);
      ss.width(part.width);
      ss.fill(part.fill);
      ss << std::fixed << ctime << part.suffix;
      value -= std::floor(ctime) * part.scaler;
      ++count;
    }
  }
  return ss.str();
}

std::string CreateMetricReport() {
  MetricsArena* arena = MetricsArena::Get();
  std::stringstream ss;
  arena->ForEachMetric([&ss](const std::string& name, MetricData* data) {
    if (data->TotalSamples() > 0) {
      EmitMetricInfo(name, data, &ss);
    }
  });
  arena->ForEachCounter([&ss](const std::string& name, CounterData* data) {
    if (data->Value() > 0) {
      EmitCounterInfo(name, data, &ss);
    }
  });
  return ss.str();
}

std::string CreateMetricReport(const std::vector<std::string>& counter_names,
                               const std::vector<std::string>& metric_names) {
  MetricsArena* arena = MetricsArena::Get();
  std::stringstream ss;
  for (const std::string& metric_name : metric_names) {
    MetricData* data = arena->GetMetric(metric_name);
    if (data && data->TotalSamples() > 0) {
      EmitMetricInfo(metric_name, data, &ss);
    }
  }
  for (const std::string& counter_name : counter_names) {
    CounterData* data = arena->GetCounter(counter_name);
    if (data && data->Value() > 0) {
      EmitCounterInfo(counter_name, data, &ss);
    }
  }
  static std::string fall_back_counter_prefix = "aten::";
  arena->ForEachCounter([&ss](const std::string& name, CounterData* data) {
    if (name.rfind(fall_back_counter_prefix, 0) == 0 && data->Value() > 0) {
      // it might emit duplicated counter if user also specified exact aten
      // counter in the `counter_names` but it should be very rare.
      EmitCounterInfo(name, data, &ss);
    }
  });
  return ss.str();
}

std::vector<std::string> GetMetricNames() {
  return MetricsArena::Get()->GetMetricNames();
}

MetricData* GetMetric(const std::string& name) {
  return MetricsArena::Get()->GetMetric(name);
}

std::vector<std::string> GetCounterNames() {
  return MetricsArena::Get()->GetCounterNames();
}

CounterData* GetCounter(const std::string& name) {
  return MetricsArena::Get()->GetCounter(name);
}

void ClearCounters() { MetricsArena::Get()->ClearCounters(); }

void ClearMetrics() { MetricsArena::Get()->ClearMetrics(); }

}  // namespace metrics
}  // namespace xla
