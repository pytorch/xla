#include "tensorflow/compiler/xla/xla_client/metrics.h"

#include "tensorflow/core/platform/default/logging.h"

#include <map>
#include <cmath>
#include <sstream>

namespace xla {
namespace metrics {
namespace {

class MetricsArena {
 public:
  static MetricsArena* Get();

  // Register a new metric in the global arena.
  void RegisterMetric(const string& name, MetricReprFn repr_fn,
                      size_t max_samples, std::shared_ptr<MetricData>* data);

  void ForEachMetric(
      const std::function<void(const string&, MetricData*)>& metric_func);

 private:
  std::mutex lock_;
  std::map<string, std::shared_ptr<MetricData>> metrics_;
};

MetricsArena* MetricsArena::Get() {
  static MetricsArena* arena = new MetricsArena();
  return arena;
}

void MetricsArena::RegisterMetric(const string& name, MetricReprFn repr_fn,
                                  size_t max_samples,
                                  std::shared_ptr<MetricData>* data) {
  std::lock_guard<std::mutex> lock(lock_);
  if (*data == nullptr) {
    std::shared_ptr<MetricData> new_data =
        std::make_shared<MetricData>(std::move(repr_fn), max_samples);
    auto it = metrics_.emplace(name, new_data).first;
    *data = it->second;
  }
}

void MetricsArena::ForEachMetric(
    const std::function<void(const string&, MetricData*)>& metric_func) {
  std::lock_guard<std::mutex> lock(lock_);
  for (auto& name_data : metrics_) {
    metric_func(name_data.first, name_data.second.get());
  }
}

void EmitMetricInfo(const string& name, MetricData* data,
                    std::stringstream* ss) {
  double counter = data->Counter();
  std::vector<Sample> samples = data->Samples();
  (*ss) << "Metric: " << name << std::endl;
  (*ss) << "  TotalSamples: " << data->TotalSamples() << std::endl;
  (*ss) << "  Counter: " << data->Repr(counter) << std::endl;
  if (!samples.empty()) {
    double total = 0.0;
    for (auto& sample : samples) {
      total += sample.value;
    }
    int64 delta_time =
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

  const int kNumPercentiles = 9;
  static double const kPercentiles[kNumPercentiles] = {
      0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99};
  std::sort(
      samples.begin(), samples.end(),
      [](const Sample& s1, const Sample& s2) { return s1.value < s2.value; });
  (*ss) << "  Percentiles: ";
  for (int i = 0; i < kNumPercentiles; ++i) {
    size_t index = kPercentiles[i] * samples.size();
    if (i > 0) {
      (*ss) << "; ";
    }
    (*ss) << (kPercentiles[i] * 100.0)
          << "%=" << data->Repr(samples[index].value);
  }
  (*ss) << std::endl;
}

}  // namespace

MetricData::MetricData(MetricReprFn repr_fn, size_t max_samples)
    : repr_fn_(std::move(repr_fn)), samples_(max_samples) {}

void MetricData::AddSample(int64 timestamp_ns, double value) {
  std::lock_guard<std::mutex> lock(lock_);
  size_t position = count_ % samples_.size();
  ++count_;
  counter_ += value;
  samples_[position] = Sample(timestamp_ns, value);
}

double MetricData::Counter() const {
  std::lock_guard<std::mutex> lock(lock_);
  return counter_;
}

size_t MetricData::TotalSamples() const {
  std::lock_guard<std::mutex> lock(lock_);
  return count_;
}

std::vector<Sample> MetricData::Samples() const {
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
  return samples;
}


Metric::Metric(string name, MetricReprFn repr_fn, size_t max_samples)
    : name_(std::move(name)),
      repr_fn_(std::move(repr_fn)),
      max_samples_(max_samples) {}

double Metric::Counter() const {
  return GetData()->Counter();
}

void Metric::AddSample(int64 timestamp_ns, double value) {
  GetData()->AddSample(timestamp_ns, value);
}

void Metric::AddSample(double value) {
  GetData()->AddSample(NowNs(), value);
}

std::vector<Sample> Metric::Samples() const {
  return GetData()->Samples();
}

string Metric::Repr(double value) const {
  return GetData()->Repr(value);
}

MetricData* Metric::GetData() const {
  if (data_ == nullptr) {
    MetricsArena* arena = MetricsArena::Get();
    arena->RegisterMetric(name_, repr_fn_, max_samples_, &data_);
  }
  return data_.get();
}


string MetricFnValue(double value) {
  std::stringstream ss;
  ss.precision(2);
  ss << std::fixed << value;
  return ss.str();
}

string MetricFnBytes(double value) {
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

string MetricFnTime(double value) {
  static struct TimePart {
    const char* suffix;
    double scaler;
    int width;
    int precision;
    char fill;
  } const time_parts[] = {
      {"d", 86400.0 * 1e9, 2, 0, '0'},
      {"h", 1440.0 * 1e9, 2, 0, '0'},
      {"m", 60.0 * 1e9, 2, 0, '0'},
      {"s", 1e9, 2, 0, '0'},
      {"ms", 1e6, 3, 0, '0'},
      {"us", 1e3, 3, 3, '0'},
  };
  int count = 0;
  std::stringstream ss;
  for (auto& part : time_parts) {
    double ctime = value / part.scaler;
    if (ctime >= 1.0 || count > 0) {
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

int64 NowNs() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      now.time_since_epoch()).count();
}

string CreateMetricReport() {
  MetricsArena* arena = MetricsArena::Get();
  std::stringstream ss;
  arena->ForEachMetric([&ss](const string& name, MetricData* data) {
    EmitMetricInfo(name, data, &ss);
  });
  return ss.str();
}

}  // namespace metrics
}  // namespace xla
