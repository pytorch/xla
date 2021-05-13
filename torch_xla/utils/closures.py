import abc
import queue
import threading

class ClosureHandler(abc.ABC):

  def __init__(self):
    pass

  @abc.abstractmethod
  def run(self, closure):
    """Run closure function
    
    Args:
      closure: callable function to run
    """
    pass

  def run_all(self, closures):
    for closure in closures:
      self.run(closure)

class AsyncClosureHandler(ClosureHandler):
  """Handler for Asynchronous Step Closures

  Args:
    max_queue_size: The maximum length of the closure queue after which
      the training loop will block until closures are evaluated.
      By default, no limit on queue size
  """

  def __init__(self, max_queue_size=-1):
    super().__init__()
    self._closure_queue = queue.Queue(max_queue_size)
    self._closure_exception = queue.Queue()
    self._closure_event_loop = None

  def start_event_loop(self):
    """Start closure event loop if not started"""
    if self._closure_event_loop is None:
      def event_loop():
        # Run loop until closure event is set and closure queue is empty
        while not self._closure_queue.empty():
          try:
            closure = self._closure_queue.get(block=True, timeout=3)
            closure()
            self._closure_queue.task_done()
          except queue.Empty:
            pass
          except Exception as e:
            self._closure_exception.put(e)
            break

      self._closure_event_loop = threading.Thread(target=event_loop)
      self._closure_event_loop.start()

  def run(self, closure):
    if (
      self._closure_event_loop is None
      or not self._closure_event_loop.is_alive()
    ):
      try:
        e = self._closure_exception.get(block=False)
        raise RuntimeError(
          "Cannot run asynchronous closure due to previously raised exception"
        ) from e
      except queue.Empty:
        self._closure_event_loop = None
        self.start_event_loop()

    self._closure_queue.put(closure, block=True)
    return