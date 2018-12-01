from __future__ import print_function

import collections
import threading


def _notify_cv(cv):
    cv.acquire()
    cv.notify()
    cv.release()


class KeydQueue(object):
    def __init__(self, maxsize=1024):
        self._maxsize = maxsize
        self._items = dict()
        self._waited_keys = set()
        self._ready_cv = threading.Condition()
        self._space_available_cv = threading.Condition()
        self._close_read = False
        self._close_write = False

    def close(self):
        self._close_read = True
        self._close_write = True
        _notify_cv(self._ready_cv)
        _notify_cv(self._space_available_cv)

    def close_write(self):
        self._close_write = True
        _notify_cv(self._ready_cv)

    def put(self, key, item):
        self._space_available_cv.acquire()
        # Wait for space available, unless there is a waiter for the incoming
        # key.
        while (len(self._items) >= self._maxsize and
               key not in self._waited_keys and not self._close_read):
            self._space_available_cv.wait()
        self._space_available_cv.release()
        if not self._close_read:
            self._ready_cv.acquire()
            self._items[key] = item
            if key in self._waited_keys:
                self._ready_cv.notify()
            self._ready_cv.release()

    def get(self, key):
        self._ready_cv.acquire()
        while key not in self._items and not self._close_write:
            self._waited_keys.add(key)
            self._ready_cv.wait()
            self._waited_keys.discard(key)
        item = self._items.pop(key, None)
        self._ready_cv.release()
        if item is not None and not self._close_write:
            _notify_cv(self._space_available_cv)
        return item


class Queue(object):
    def __init__(self, maxsize=1024):
        self._maxsize = maxsize
        self._items = collections.deque()
        self._ready_cv = threading.Condition()
        self._space_available_cv = threading.Condition()
        self._close_read = False
        self._close_write = False

    def close(self):
        self._close_read = True
        self._close_write = True
        _notify_cv(self._ready_cv)
        _notify_cv(self._space_available_cv)

    def close_write(self):
        self._close_write = True
        _notify_cv(self._ready_cv)

    def put(self, item):
        self._space_available_cv.acquire()
        while (len(self._items) >= self._maxsize and not self._close_read):
            self._space_available_cv.wait()
        self._space_available_cv.release()
        if not self._close_read:
            self._ready_cv.acquire()
            self._items.append(item)
            self._ready_cv.notify()
            self._ready_cv.release()

    def get(self):
        self._ready_cv.acquire()
        while not self._items and not self._close_write:
            self._ready_cv.wait()
        item = self._items.popleft() if self._items else None
        self._ready_cv.release()
        if item is not None and not self._close_write:
            _notify_cv(self._space_available_cv)
        return item
